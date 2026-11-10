"""Executable scalar-reward calibration, not the project's comparative experiment.

Run from the bundle root: python -m reference.demo_mce --seed 0
Train reward labels are never passed to fit_cost: it consumes only chosen actions,
visited belief-state IDs, and the known sensor model. Goal reward=1 and temperature
are explicitly supplied structural assumptions. No IQ/AIRL performance is measured.
"""
import argparse
import json
import numpy as np
from scipy.optimize import minimize_scalar
from .numerical_reference import cue_model, soft_dp, episode_mean_nll


def generate_trajectories(n, *, cost, temperature, seed, reliabilities=(.8,)):
    rng=np.random.default_rng(seed)
    rows=[]
    for episode in range(n):
        reliability=float(reliabilities[episode % len(reliabilities)])
        p,r,m=cue_model(cost,reliability)
        solution=soft_dp(p,r,m,temperature=temperature)
        action=int(rng.choice(3,p=solution.policy[0,0]))
        rows.append((episode,0,0,action,reliability))
        if action==2:
            state=int(rng.choice(4,p=p[0,2]))
            action=int(rng.choice(3,p=solution.policy[1,state]))
            rows.append((episode,1,state,action,reliability))
    return np.asarray(rows,dtype=float)


def log_probabilities(data, cost, temperature):
    output=np.empty(len(data))
    for reliability in np.unique(data[:,4]):
        selected=np.flatnonzero(data[:,4]==reliability)
        solution=soft_dp(*cue_model(cost,reliability),temperature=temperature)
        rows=data[selected,:4].astype(int)
        probabilities=solution.policy[rows[:,1],rows[:,2],rows[:,3]]
        output[selected]=np.log(np.maximum(probabilities,np.finfo(float).tiny))
    return output


def trajectory_nll(data,cost,temperature):
    # True trajectory likelihood sums conditional action log probabilities.
    # Dividing by trajectory count preserves the maximum-likelihood optimum.
    return float(-log_probabilities(data,cost,temperature).sum()/len(np.unique(data[:,0])))


def fit_cost(training_actions, *, temperature, cost_bounds=(1e-5,.6)):
    result=minimize_scalar(lambda c:trajectory_nll(training_actions,c,temperature),
                           bounds=cost_bounds,method='bounded',
                           options={'xatol':1e-10})
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.x)


def evaluate(data,cost,temperature):
    lp=log_probabilities(data,cost,temperature)
    return {'trajectories':int(len(np.unique(data[:,0]))),'steps':int(len(data)),
            'trajectory_sum_nll':float(-lp.sum()/len(np.unique(data[:,0]))),
            'episode_mean_step_nll':episode_mean_nll(lp,data[:,0].astype(int))}


def run(seed=0):
    true_cost=.15;temperature=.1
    train=generate_trajectories(400,cost=true_cost,temperature=temperature,seed=seed)
    validation=generate_trajectories(100,cost=true_cost,temperature=temperature,seed=seed+10000)
    test=generate_trajectories(100,cost=true_cost,temperature=temperature,seed=seed+20000,
                               reliabilities=(.65,.95))
    estimated=fit_cost(train,temperature=temperature)
    return {'scope':'synthetic scalar MCE calibration only; not IQ/AIRL comparison',
            'seed':seed,'temperature':temperature,'goal_reward_anchor':1.0,
            'prior_left':0.5,'fit_cost_bounds':[1e-5,0.6],
            'true_cost_evaluator_only':true_cost,'fitted_cost':estimated,
            'absolute_cost_error':abs(estimated-true_cost),
            'train_reliability':.8,'held_out_test_reliabilities':[.65,.95],
            'fit_objective':'mean trajectory-summed action NLL; no reward labels',
            'train':evaluate(train,estimated,temperature),
            'validation':evaluate(validation,estimated,temperature),
            'test':evaluate(test,estimated,temperature)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    print(json.dumps(run(args.seed),indent=2))
