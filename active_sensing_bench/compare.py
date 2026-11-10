"""Run the explicit-state comparison from two condition-matched expert checkpoints.

Use an installed wheel in the pinned AIRL environment for every method to keep
dependency versions fixed. Each invocation writes a new output directory.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from .data import write_json


def commands(config, output, experts, methods):
    root = Path(output).resolve()
    python = [sys.executable, "-I", "-m"]
    def command(module, **kwargs):
        result = python + [module]
        for key, value in kwargs.items():
            result.append("--" + key.replace("_", "-"))
            result.extend(str(v) for v in value) if isinstance(value, list) else result.append(str(value))
        return result
    plan = []
    for representation, expert in experts.items():
        data = root / f"data_{representation}"
        plan.append(command("active_sensing_bench.generate_demos", policy=Path(expert).resolve(),
                    representation=representation, view_size=config["view_size"], counts=config["demonstrations"],
                    seed=config["data_seed"], output=data))
    # True-reward controls precede recovered-reward optimization with matched seeds.
    for representation in experts:
        for fresh_seed in config["fresh_policy_seeds"]:
            plan.append(command("active_sensing_bench.retrain_policy", reward="true", representation=representation,
                view_size=config["view_size"], seed=fresh_seed, timesteps=config["fresh_policy_interactions"],
                envs=config["parallel_environments"], steps_per_env=config["steps_per_environment"],
                batch_size=config["ppo_batch_size"], epochs=config["ppo_epochs"],
                gamma=config["gamma"], lr=config["learning_rate"],
                eval_episodes=config["evaluation_episodes"], eval_seed=config["evaluation_seed"],
                output=root / f"true_{representation}_seed{fresh_seed}"))
    for method in methods:
        for representation in experts:
            data = root / f"data_{representation}"
            for seed in config["learner_seeds"]:
                run = root / f"{method}_{representation}_seed{seed}"
                common = dict(data=data, output=run, seed=seed)
                if method == "airl":
                    plan.append(command("airl_runner.train", **common, rounds=config["airl_rounds"],
                        envs=config["parallel_environments"], steps_per_env=config["steps_per_environment"],
                        demo_batch_size=config["discriminator_batch_size_per_class"],
                        disc_updates=config["discriminator_updates_per_round"], ppo_batch_size=config["ppo_batch_size"],
                        ppo_epochs=config["ppo_epochs"], gamma=config["gamma"], lr=config["learning_rate"]))
                else:
                    plan.append(command(f"active_sensing_bench.train_{method}", **common,
                        steps=config["offline_updates"], eval_every=config["offline_eval_every"],
                        batch_size=config["offline_batch_size"], weighting=config["weighting"],
                        temperature=config["temperature"], chi_lambda=config["chi_lambda"], gamma=config["gamma"],
                        lr=config["learning_rate"]))
                plan.append(command("active_sensing_bench.evaluate", run=run, data=data,
                    episodes=config["evaluation_episodes"], seed=config["evaluation_seed"], output=run / "evaluation.json"))
                if method != "bc":
                    for fresh_seed in config["fresh_policy_seeds"]:
                        plan.append(command("active_sensing_bench.retrain_policy", reward=run / "reward.pt",
                            representation=representation, view_size=config["view_size"], seed=fresh_seed,
                            timesteps=config["fresh_policy_interactions"], envs=config["parallel_environments"],
                            steps_per_env=config["steps_per_environment"], batch_size=config["ppo_batch_size"],
                            epochs=config["ppo_epochs"], eval_episodes=config["evaluation_episodes"],
                            gamma=config["gamma"], lr=config["learning_rate"],
                            eval_seed=config["evaluation_seed"], output=run / f"fresh_seed{fresh_seed}"))
    return plan


def summarize(output):
    root, rows = Path(output), []
    for path in sorted(root.glob("*/selection.json")):
        run = path.parent
        config = json.loads((run / "config.json").read_text())
        selection = json.loads(path.read_text())
        evaluation = json.loads((run / "evaluation.json").read_text())
        representation = config["condition"]["representation"]
        fresh = sorted(run.glob("fresh_seed*/evaluation.json"))
        for fresh_path in fresh or [None]:
            recovery = json.loads(fresh_path.read_text()) if fresh_path else {}
            fresh_seed = int(fresh_path.parent.name.removeprefix("fresh_seed")) if fresh_path else None
            control = json.loads((root / f"true_{representation}_seed{fresh_seed}" / "evaluation.json").read_text()) if fresh_path else {}
            rows.append(dict(method=config["method"], representation=representation, training_seed=config["seed"],
                fresh_seed=fresh_seed, expert_id=config["expert_id"], demonstration_count=config["demonstration_count"],
                training_transition_count=config["training_transition_count"],
                online_interactions=selection.get("online_interactions", 0),
                reward_family=config["reward_family"], model_access=config["model_access"],
                test_episode_nll=evaluation["imitation"]["episode_mean_nll"],
                test_step_nll=evaluation["imitation"]["step_mean_nll"],
                independent_return=evaluation["rollouts"]["mean_true_return"],
                independent_success=evaluation["rollouts"]["success_rate"],
                fresh_reward_policy_return=recovery.get("mean_true_return"),
                true_reward_control_return=control.get("mean_true_return"),
                reward_freeze_check=recovery.get("reward_freeze_check")))
    if not rows:
        raise ValueError("No completed comparison runs")
    with (root / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expert-full", required=True)
    parser.add_argument("--expert-belief", required=True)
    parser.add_argument("--methods", nargs="+", choices=("bc", "iq", "airl"), default=["bc", "iq", "airl"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    for key, expected in dict(horizon=400, num_distractors=0, action_count=7, input_dim=71, hidden_sizes=[128, 128]).items():
        if config[key] != expected:
            raise ValueError(f"This implementation requires {key}={expected}")
    if not 0 <= config["minimum_control_success"] <= 1:
        raise ValueError("Declare a learnability success threshold in [0,1]")
    if config["weighting"] != "transition" and "airl" in args.methods:
        raise ValueError("This AIRL runner uses transition weighting; use the matched comparison weighting")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    plan = commands(config, root, {"full": args.expert_full, "belief": args.expert_belief}, args.methods)
    write_json(root / "config.json", config)
    write_json(root / "commands.json", plan)
    if args.plan_only:
        return
    for command in plan:
        print("Running:", command, flush=True)
        subprocess.run(command, check=True)
        if "--reward" in command and command[command.index("--reward") + 1] == "true":
            control_path = Path(command[command.index("--output") + 1]) / "evaluation.json"
            control = json.loads(control_path.read_text())
            if control["success_rate"] < config["minimum_control_success"]:
                raise RuntimeError(f"True-reward learnability gate failed: {control_path}. Increase the shared forward budget or investigate optimization before inverse-method comparisons.")
    summarize(root)


if __name__ == "__main__":
    main()
