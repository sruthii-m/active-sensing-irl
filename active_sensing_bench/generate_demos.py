"""Generate neutral 71-D episodes on preassigned disjoint complete layouts."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .data import SCHEMA, digest, layout_id, save_episodes, split_layouts, write_json
from .env import DecisionEnv
from .evaluate import load_policy


def collect_episode(env, policy, layout, episode_id, expert_id, condition, rng, deterministic):
    x, _ = env.reset(options={"layout": layout})
    observations, actions, rewards, flags = [x], [], [], []
    while True:
        action = int(rng.integers(7)) if policy is None else policy.act(x, rng, deterministic)
        x, reward, terminated, truncated, info = env.step(action)
        observations.append(x)
        actions.append(action)
        rewards.append(reward)
        flags.append(info)
        if terminated or truncated:
            break
    episode = dict(episode_id=episode_id, layout_id=layout_id(layout), expert_id=expert_id,
                   condition=condition, observations=np.asarray(observations, np.float32),
                   actions=np.asarray(actions, np.int64), length=len(actions))
    for field in ("terminated", "truncated", "task_terminal"):
        episode[field] = np.asarray([row[field] for row in flags], dtype=bool)
    sidecar = dict(episode_id=episode_id, layout=list(layout), true_rewards=rewards)
    return episode, sidecar


def generate(args):
    torch.set_num_threads(1)
    condition = dict(representation=args.representation, view_size=args.view_size,
                     horizon=400, collection_limit=args.collection_limit)
    policy = None
    if args.policy:
        policy, config = load_policy(args.policy)
        for key in ("representation", "view_size", "horizon"):
            if config["condition"][key] != condition[key]:
                raise ValueError("Expert was trained under a different condition")
        policy_path = Path(args.policy) / ("policy.pt" if (Path(args.policy) / "policy.pt").exists() else "policy.zip")
        expert_id = digest(policy_path)
    elif args.random_smoke:
        expert_id = "random-policy-SMOKE-ONLY"
    else:
        raise ValueError("Supply --policy or explicitly select --random-smoke")
    layouts = split_layouts(tuple(args.counts), args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema=SCHEMA, condition=condition, expert_id=expert_id, seed=args.seed,
                    deterministic=not args.sample_actions, expert_temperature=None,
                    split_unit="complete reset pose, direction, goal", splits={},
                    scope="demonstration/probe splits; does not guarantee expert-training layout holdout")
    rng = np.random.default_rng(args.seed)
    env = DecisionEnv(args.representation, args.view_size, args.collection_limit)
    try:
        for split, specs in layouts.items():
            episodes, sidecars = [], []
            for i, layout in enumerate(specs):
                ep, sidecar = collect_episode(env, policy, layout, f"{split}-{i}", expert_id,
                                              condition, rng, not args.sample_actions)
                episodes.append(ep)
                sidecars.append(sidecar)
            path = out / f"{split}.npz"
            save_episodes(path, episodes)
            write_json(out / f"{split}.evaluation-only.json", sidecars)
            manifest["splits"][split] = dict(sha256=digest(path), episode_ids=[e["episode_id"] for e in episodes],
                                            layout_ids=[e["layout_id"] for e in episodes])
    finally:
        env.close()
    write_json(out / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--policy")
    source.add_argument("--random-smoke", action="store_true")
    parser.add_argument("--representation", choices=("full", "belief"), required=True)
    parser.add_argument("--view-size", type=int, default=3)
    parser.add_argument("--collection-limit", type=int)
    parser.add_argument("--counts", type=int, nargs=3, default=[400, 100, 100])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-actions", action="store_true")
    parser.add_argument("--output", required=True)
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
