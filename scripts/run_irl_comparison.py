"""Reuse validated experts and execute the existing comparison plan resumably.

Run with the installed benchmark environment and python -I. This wrapper does
not change learning algorithms. Resume skips verified, completed commands; it
archives and restarts an interrupted command, not its optimizer checkpoint.
"""
import argparse
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

from active_sensing_bench.compare import commands, summarize


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def atomic_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".progress-",
                                     suffix=".tmp", delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def runtime_provenance():
    sources = {"launcher": sha256(__file__)}
    for package in ("active_sensing_bench", "airl_runner"):
        spec = importlib.util.find_spec(package)
        directory = Path(spec.origin).parent
        sources[package] = {str(p.relative_to(directory)): sha256(p)
                            for p in sorted(directory.rglob("*.py"))}
    return dict(python=sys.version, executable=sys.executable, sources=sources,
                versions={p: importlib.metadata.version(p) for p in
                          ("numpy", "torch", "gymnasium", "minigrid", "stable-baselines3", "imitation", "scipy")})


def validate_experts(config, directory):
    for key, value in dict(horizon=400, num_distractors=0, action_count=7,
                           input_dim=71, hidden_sizes=[128, 128]).items():
        if config[key] != value:
            raise ValueError(f"This benchmark requires {key}={value}")
    if not 0 <= config["minimum_control_success"] <= 1 or config["weighting"] != "transition":
        raise ValueError("Need a control threshold in [0,1] and matched transition weighting")
    experts, provenance = {}, {}
    for representation in ("full", "belief"):
        expert = (directory / f"{representation}_seed0").resolve()
        files = [expert / name for name in ("policy.zip", "config.json", "evaluation.json")]
        if not all(p.is_file() and p.stat().st_size for p in files):
            raise ValueError(f"Missing expert policy/config/evaluation in {expert}")
        if (expert / "policy.pt").exists():
            raise ValueError(f"Ambiguous expert: both policy.pt and policy.zip in {expert}")
        training, evaluation = read_json(files[1]), read_json(files[2])
        condition = dict(representation=representation, view_size=config["view_size"],
                         horizon=config["horizon"], collection_limit=None)
        if training.get("condition") != condition or training.get("reward") != "true" or training.get("seed") != 0:
            raise ValueError(f"Need a condition-matched, true-reward seed-0 expert: {expert}")
        if evaluation.get("actual_interactions", 0) < config["fresh_policy_interactions"]:
            raise ValueError(f"Expert has less than the declared training budget: {expert}")
        success = evaluation.get("success_rate", float("nan"))
        if not math.isfinite(success) or not config["minimum_control_success"] <= success <= 1:
            raise ValueError(f"Expert success {success} fails the quality check: {expert}")
        if len(evaluation.get("episodes", [])) < config["evaluation_episodes"] or not evaluation.get("deterministic"):
            raise ValueError(f"Expert needs the configured deterministic evaluation: {expert}")
        experts[representation] = str(expert)
        provenance[representation] = dict(directory=str(expert),
                                          files={p.name: sha256(p) for p in files})
    return experts, provenance


def argument(command, name):
    return command[command.index(name) + 1]


def stage_paths(command):
    output = Path(argument(command, "--output"))
    if argument(command, "-m") == "active_sensing_bench.evaluate":
        return [output, output.with_suffix(".coverage.npz")]
    return [output]


def artifacts(command, root):
    output = stage_paths(command)[0]
    module = argument(command, "-m")
    markers = {
        "active_sensing_bench.generate_demos": ("manifest.json", "train.npz", "validation.npz", "test.npz"),
        "active_sensing_bench.retrain_policy": ("config.json", "policy.zip", "evaluation.json"),
        "active_sensing_bench.train_bc": ("config.json", "selection.json", "policy.pt"),
        "active_sensing_bench.train_iq": ("config.json", "selection.json", "policy.pt", "reward.pt"),
        "airl_runner.train": ("config.json", "selection.json", "policy.zip", "reward.pt"),
        "active_sensing_bench.train_mce": ("config.json", "manifest.json", "reward.json", "evaluation.json"),
    }
    required = [output / p for p in markers[module]] if module in markers else [output]
    if not all(p.is_file() for p in required):
        raise RuntimeError(f"Stage exited without its completion artifacts: {output}")
    files = sorted(p for p in output.rglob("*") if p.is_file()) if output.is_dir() else [
        p for p in stage_paths(command) if p.is_file()]
    return {str(p.relative_to(root)): sha256(p) for p in files}


def verify_artifacts(root, files):
    for name, expected in files.items():
        path = root / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Completed artifact changed or is missing; refusing resume: {path}")


def check_control(command, config):
    if "--reward" in command and argument(command, "--reward") == "true":
        report = Path(argument(command, "--output")) / "evaluation.json"
        success = read_json(report).get("success_rate", float("nan"))
        if not math.isfinite(success) or not config["minimum_control_success"] <= success <= 1:
            raise RuntimeError(f"True-reward learnability gate failed ({success}): {report}")


def archive_interrupted(command, root, index):
    existing = [path for path in stage_paths(command) if path.exists()]
    if not existing:
        return
    archive = root / "interrupted" / f"stage-{index + 1:03d}-{uuid.uuid4().hex}"
    archive.mkdir(parents=True)
    for path in existing:
        path.rename(archive / path.name)
    print(f"Preserved interrupted stage artifacts in {archive}", flush=True)


def run(args):
    config = read_json(args.config)
    root = Path(args.output).resolve()
    experts, expert_provenance = validate_experts(config, Path(args.experts))
    plan = commands(config, root, experts, ["bc", "iq", "airl"])
    plan.append([sys.executable, "-I", "-m", "active_sensing_bench.train_mce",
                 "--seed", "0", "--output", str(root / "mce")])
    expected = dict(schema=1, config=config, experts=expert_provenance,
                    runtime=runtime_provenance(), commands=plan)
    if args.resume:
        if not (root / "progress.json").is_file():
            raise ValueError(f"No resumable progress file in {root}")
    else:
        root.mkdir(parents=True, exist_ok=False)
    with (root / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another launcher is already running in {root}") from error
        progress_file = root / "progress.json"
        if args.resume:
            if read_json(root / "launch.json") != expected:
                raise ValueError("Config, experts, installed code, or runtime changed; start a new comparison")
            progress = read_json(progress_file)
        else:
            atomic_json(root / "launch.json", expected)
            atomic_json(root / "config.json", config)
            atomic_json(root / "commands.json", plan)
            progress = dict(completed=[], active=None, finished=False)
            atomic_json(progress_file, progress)
        completed = len(progress["completed"])
        if (completed > len(plan) or progress["active"] not in (None, completed)
                or (progress["finished"] and completed != len(plan))):
            raise ValueError("Inconsistent progress journal; inspect before resuming")
        for command, entry in zip(plan, progress["completed"]):
            verify_artifacts(root, entry)
            check_control(command, config)
        print(f"Comparison directory: {root}\nVerified {completed}/{len(plan)} completed stages", flush=True)
        if args.plan_only:
            print("Plan only; no training launched", flush=True)
            return
        if progress["finished"]:
            verify_artifacts(root, progress["summary"])
            print(f"Already complete: {root / 'comparison.csv'}", flush=True)
            return
        for index in range(completed, len(plan)):
            command = plan[index]
            if progress["active"] == index:
                archive_interrupted(command, root, index)
            elif any(p.exists() for p in stage_paths(command)):
                raise ValueError(f"Unexpected output for an unstarted stage: {stage_paths(command)[0]}")
            progress["active"] = index
            atomic_json(progress_file, progress)
            print(f"Stage {index + 1}/{len(plan)}: {command}", flush=True)
            subprocess.run(command, check=True)
            progress["completed"].append(artifacts(command, root))
            progress["active"] = None
            atomic_json(progress_file, progress)
            check_control(command, config)
        summarize(root)
        progress["summary"] = {"comparison.csv": sha256(root / "comparison.csv")}
        progress["finished"] = True
        atomic_json(progress_file, progress)
        print(f"Completed. Comparison: {root / 'comparison.csv'}", flush=True)
        print(f"MCE results: {root / 'mce' / 'evaluation.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--experts", required=True, help="Parent of full_seed0/ and belief_seed0/")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
