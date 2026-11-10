"""Shared offline training, splits, weighting, and validation selection for BC/IQ."""
import argparse
import copy
import importlib.metadata
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import digest, load_split, transition_arrays, write_json
from .evaluate import MLPPolicy, likelihood
from .networks import iq_loss, mlp
from .reward_exports import FrozenReward, code_version


def train(args):
    if min(args.steps, args.eval_every, args.batch_size, args.target_every) < 1 or args.temperature <= 0 or args.chi_lambda <= 0:
        raise ValueError("Positive budgets, temperature and chi regularization required")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    training, manifest = load_split(args.data, "train")
    validation, _ = load_split(args.data, "validation")
    arrays = transition_arrays(training, args.weighting)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), "condition": manifest["condition"], "code_version": code_version(),
              "manifest_sha256": digest(Path(args.data) / "manifest.json"),
              "checkpoint_rule": "minimum validation episode-mean NLL", "online_interactions": 0,
              "training_transition_count": len(arrays["action"]), "demonstration_count": len(training),
              "expert_id": manifest["expert_id"], "reward_family": "implicit neural Q" if args.method == "iq" else None,
              "model_access": "known transition model for reward export" if args.method == "iq" else "none",
              "versions": {p: importlib.metadata.version(p) for p in ("torch", "numpy", "minigrid", "gymnasium")}}
    write_json(out / "config.json", config)
    write_json(out / "split_manifest.json", manifest)
    network = mlp()
    target = copy.deepcopy(network).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    x, nx = torch.from_numpy(arrays["x"]), torch.from_numpy(arrays["next_x"])
    actions, terminal = torch.from_numpy(arrays["action"]), torch.from_numpy(arrays["terminal"])
    temperature = args.temperature if args.method == "iq" else 1.
    best, best_step, best_state, history = float("inf"), 0, None, []
    for step in range(args.steps + 1):
        if step:
            index = rng.choice(len(x), args.batch_size, replace=True, p=arrays["sampling_probability"])
            q = network(x[index])
            if args.method == "bc":
                loss = F.cross_entropy(q, actions[index])
            else:
                loss, _ = iq_loss(q, actions[index], target(nx[index]), terminal[index],
                                  args.gamma, args.temperature, args.chi_lambda)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % args.target_every == 0:
                target.load_state_dict(network.state_dict())
        if step % args.eval_every == 0 or step == args.steps:
            metrics = likelihood(MLPPolicy(network, temperature), validation)
            score = metrics["episode_mean_nll"]
            history.append(dict(step=step, **{k: v for k, v in metrics.items() if k != "episodes"}))
            if score < best:
                best, best_step, best_state = score, step, copy.deepcopy(network.state_dict())
            print(f"{args.method} step={step} validation_episode_nll={score:.6f}", flush=True)
    network.load_state_dict(best_state)
    torch.save(dict(network=best_state, temperature=temperature), out / "policy.pt")
    if args.method == "iq":
        FrozenReward(network, method="iq", representation=manifest["condition"]["representation"],
                     view_size=manifest["condition"]["view_size"], gamma=args.gamma,
                     temperature=args.temperature).save(out / "reward.pt")
    report = dict(selected_step=best_step, best_validation_nll=best, history=history,
                  train_episode_ids=[e["episode_id"] for e in training])
    write_json(out / "selection.json", report)
    return report


def main(method=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if method is None:
        parser.add_argument("--method", choices=("bc", "iq"), required=True)
    else:
        parser.set_defaults(method=method)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--weighting", choices=("transition", "episode"), default="transition")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--temperature", type=float, default=.001)
    parser.add_argument("--chi-lambda", type=float, default=.5)
    parser.add_argument("--target-every", type=int, default=4)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
