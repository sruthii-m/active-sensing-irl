"""Optuna hyperparameter sweep for the full-state PPO expert.
Usage:
    python -m experts.sweep --n-trials 40 --timesteps 30000
    python -m experts.sweep --n-trials 200 --n-jobs 4 \
        --storage sqlite:///artifacts/sweeps/forage_full_state.db --study-name forage_full_state
"""

from __future__ import annotations

import argparse
from pathlib import Path

import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from environment import make_full_obs_learning_env
from experts.train_full_state import evaluate_success_rate

_NET_ARCHES = {
    "small": [64, 64],
    "medium": [128, 128],
    "large": [256, 256],
}


class SuccessRatePruningCallback(BaseCallback):
    """Periodically evaluates success rate and reports it to the Optuna
    trial, stopping training early if the trial should be pruned."""

    def __init__(self, trial: optuna.Trial, eval_freq: int, n_eval_episodes: int, env_kwargs: dict | None):
        super().__init__()
        self.trial = trial
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.env_kwargs = env_kwargs
        self._last_eval_step = 0
        self.last_success_rate = 0.0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True
        self._last_eval_step = self.num_timesteps
        self.last_success_rate = evaluate_success_rate(
            self.model, n_episodes=self.n_eval_episodes, env_kwargs=self.env_kwargs
        )
        self.trial.report(self.last_success_rate, self.num_timesteps)
        if self.trial.should_prune():
            raise optuna.TrialPruned()
        return True


def objective(trial: optuna.Trial, *, total_timesteps: int, seed: int, env_kwargs: dict | None) -> float:
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024])
    n_minibatches = trial.suggest_categorical("n_minibatches", [4, 8, 16])
    batch_size = max(1, n_steps // n_minibatches)

    model = PPO(
        "MlpPolicy",
        make_full_obs_learning_env(**(env_kwargs or {})),
        seed=seed,
        verbose=0,
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=trial.suggest_int("n_epochs", 3, 20),
        gamma=trial.suggest_float("gamma", 0.9, 0.999),
        gae_lambda=trial.suggest_float("gae_lambda", 0.8, 0.999),
        clip_range=trial.suggest_float("clip_range", 0.1, 0.4),
        ent_coef=trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True),
        policy_kwargs={"net_arch": _NET_ARCHES[trial.suggest_categorical("net_arch", list(_NET_ARCHES))]},
    )

    callback = SuccessRatePruningCallback(
        trial, eval_freq=max(total_timesteps // 10, n_steps), n_eval_episodes=20, env_kwargs=env_kwargs
    )
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.env.close()

    final_success_rate = evaluate_success_rate(model, n_episodes=50, env_kwargs=env_kwargs)
    return final_success_rate


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna sweep for the full-state PPO expert.")
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--timesteps", type=int, default=100_000, help="Per-trial timestep budget (should match the real training run).")
    parser.add_argument("--n-warmup-evals", type=int, default=3, help="Eval checkpoints to skip before pruning can kick in.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel trials (only helps with a shared --storage).")
    parser.add_argument("--study-name", default="forage_full_state")
    parser.add_argument("--storage", default=None, help="e.g. sqlite:///artifacts/sweeps/forage_full_state.db to persist/resume.")
    args = parser.parse_args()

    if args.storage and args.storage.startswith("sqlite:///"):
        Path(args.storage.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)

    # Pruner's n_warmup_steps is measured in the same units as the `step` passed to
    # trial.report (num_timesteps), so convert "skip N eval checkpoints" into a
    # timestep threshold using the same eval_freq formula the callback uses.
    eval_freq = max(args.timesteps // 10, 1)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=args.n_warmup_evals * eval_freq
        ),
    )
    starting_params = {
    "learning_rate": 4e-4,
    "n_steps": 256,
    "batch_size": 64
    }

    study.enqueue_trial(starting_params)

    study.optimize(
        lambda trial: objective(trial, total_timesteps=args.timesteps, seed=args.seed, env_kwargs=None),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        # SQLite storage under concurrent workers can race two "tell" calls onto the
        # same pruned trial; skip that trial instead of aborting the whole sweep.
        catch=(ValueError,),
    )

    print(f"\nBest success rate: {study.best_value:.2%}")
    print("Best hyperparameters:")
    best = dict(study.best_params)
    n_steps, n_minibatches = best.pop("n_steps"), best.pop("n_minibatches")
    net_arch = _NET_ARCHES[best.pop("net_arch")]
    batch_size = max(1, n_steps // n_minibatches)
    cli_flags = " ".join(
        [
            f"--learning-rate {best['learning_rate']}",
            f"--n-steps {n_steps}",
            f"--batch-size {batch_size}",
            f"--n-epochs {best['n_epochs']}",
            f"--gamma {best['gamma']}",
            f"--gae-lambda {best['gae_lambda']}",
            f"--clip-range {best['clip_range']}",
            f"--ent-coef {best['ent_coef']}",
            f"--net-arch {' '.join(str(x) for x in net_arch)}",
        ]
    )
    print(f"\npython -m experts.train_full_state {cli_flags} --timesteps 100000")


if __name__ == "__main__":
    main()
