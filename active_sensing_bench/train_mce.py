"""Scalar MCE calibration with likelihood profile and fresh exact reward planning."""
import argparse
import json
from pathlib import Path

import numpy as np

from .data import digest, write_json
from .mce_calibration import evaluate, fit_cost, generate_trajectories, log_probabilities, trajectory_nll
from .numerical_reference import cue_model, soft_dp


def expected_return(policy, true_cost, reliability):
    transition, reward, _ = cue_model(true_cost, reliability)
    value = np.zeros(4)
    for t in range(1, -1, -1):
        value = (policy[t] * (reward[t] + np.einsum("san,n->sa", transition, value))).sum(-1)
    return float(value[0])


def run(output, seed=0):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    temperature, true_cost = .1, .15
    contexts = {"train": (.8,), "validation": (.8,), "test": (.65, .95)}
    counts = dict(train=400, validation=100, test=100)
    datasets = {name: generate_trajectories(counts[name], cost=true_cost, temperature=temperature,
                seed=seed + i * 10000, reliabilities=contexts[name]) for i, name in enumerate(counts)}
    estimate = fit_cost(datasets["train"], temperature=temperature)
    config = dict(seed=seed, temperature=temperature, goal_reward_anchor=1., prior=.5,
                  contexts=contexts, cost_bounds=[1e-5, .6], model_access="exact cue transition model",
                  fitting="mean trajectory-summed action NLL", reward_family="single positive sensing cost",
                  scope="synthetic structured recovery reference; stronger reward-family assumptions than neural methods")
    write_json(out / "config.json", config)
    write_json(out / "reward.json", dict(method="mce", sensing_cost=estimate, correct_reward=1., incorrect_reward=0.))
    before = digest(out / "reward.json")
    profile_costs = np.linspace(1e-5, .6, 601)
    profile_nll = np.array([trajectory_nll(datasets["train"], cost, temperature) for cost in profile_costs])
    np.savez_compressed(out / "likelihood_profile.npz", cost=profile_costs, trajectory_nll=profile_nll)
    admissible = profile_costs[2 * counts["train"] * (profile_nll - trajectory_nll(datasets["train"], estimate, temperature)) <= 3.841459]
    report = dict(fitted_cost=estimate, true_cost_evaluator_only=true_cost, absolute_cost_error=abs(estimate-true_cost),
                  profile_likelihood_ci95=[float(admissible.min()), float(admissible.max())],
                  interval_method="asymptotic likelihood ratio, grid resolution .001", splits={}, fresh_policy={})
    manifest = {}
    for name, data in datasets.items():
        np.savez_compressed(out / f"{name}.npz", decisions=data)
        manifest[name] = dict(sha256=digest(out / f"{name}.npz"), count=counts[name], reliabilities=contexts[name])
        report["splits"][name] = {**evaluate(data, estimate, temperature),
                                   "step_mean_nll": float(-log_probabilities(data, estimate, temperature).mean())}
    for reliability in (.65, .8, .95):
        saved = json.loads((out / "reward.json").read_text())
        solution = soft_dp(*cue_model(saved["sensing_cost"], reliability), temperature=temperature)
        control = soft_dp(*cue_model(true_cost, reliability), temperature=temperature)
        np.savez_compressed(out / f"policy_q{reliability}.npz", q=solution.q, policy=solution.policy)
        report["fresh_policy"][str(reliability)] = dict(
            expected_true_return=expected_return(solution.policy, true_cost, reliability),
            true_reward_control_return=expected_return(control.policy, true_cost, reliability),
            sensing_probability=float(solution.policy[0, 0, 2]))
    report["reward_freeze_check"] = before == digest(out / "reward.json")
    write_json(out / "manifest.json", manifest)
    write_json(out / "evaluation.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.seed), indent=2))
