"""Full-state expert: PPO (Stable-Baselines3, MlpPolicy) on the FullyObsWrapper
foraging env. Validate success rate before generating demonstrations."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from perception_irl.environment import make_full_obs_learning_env


def evaluate_success_rate(model: PPO, *, n_episodes: int = 50, env_kwargs: dict | None = None) -> float:
    """Fraction of episodes that reach the goal (MiniGrid gives reward > 0
    only on success, 0 on timeout), under the deterministic policy."""
    env = make_full_obs_learning_env(**(env_kwargs or {}))
    successes = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        episode_reward = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
        if episode_reward > 0:
            successes += 1
    env.close()
    return successes / n_episodes


def train(
    *,
    total_timesteps: int,
    seed: int = 0,
    save_path: str | None = None,
    env_kwargs: dict | None = None,
    learning_rate: float = 4e-4,
    n_steps: int = 256,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.0,
    net_arch: list[int] | None = None,
) -> PPO:
    """Train the full-state PPO expert and optionally save it at ``save_path``."""
    env = make_full_obs_learning_env(**(env_kwargs or {}))
    model = PPO(
        "MlpPolicy", env, seed=seed, verbose=1,
        learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size,
        n_epochs=n_epochs, gamma=gamma, gae_lambda=gae_lambda,
        clip_range=clip_range, ent_coef=ent_coef,
        policy_kwargs={"net_arch": net_arch or [64, 64]},
    )
    model.learn(total_timesteps=total_timesteps)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
    env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the full-state foraging expert.")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", default="artifacts/experts/forage_full_state")
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--net-arch", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--eval-episodes", type=int, default=50)
    args = parser.parse_args()
    trained = train(
        total_timesteps=args.timesteps, seed=args.seed, save_path=args.save_path,
        learning_rate=args.learning_rate, n_steps=args.n_steps, batch_size=args.batch_size,
        n_epochs=args.n_epochs, gamma=args.gamma, gae_lambda=args.gae_lambda,
        clip_range=args.clip_range, ent_coef=args.ent_coef, net_arch=args.net_arch,
    )
    success_rate = evaluate_success_rate(trained, n_episodes=args.eval_episodes)
    print(f"Success rate over {args.eval_episodes} episodes: {success_rate:.2%}")
