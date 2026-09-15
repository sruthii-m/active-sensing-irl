# Active Sensing in Inverse Reinforcement Learning

Can information-seeking behavior emerge from a learned task reward, without adding an explicit reward for reducing uncertainty?

This project studies that question in a partially observable foraging task. An agent observes the environment from an egocentric viewpoint, infers a history-dependent internal state, and learns a reward and policy from expert demonstrations with inverse reinforcement learning (IRL). Its actions are then compared with those of a reward-free uncertainty-minimization policy.

## Research question

Animals act from partial, first-person observations rather than direct access to the true state of the world. They nevertheless maintain internal representations that guide where they look and move. This project asks whether similar active sensing can arise because resolving uncertainty helps an agent optimize a learned task reward.

The central comparison is:

1. **Reward-driven policy:** choose actions using a learned reward and a belief-like representation of observation history.
2. **Uncertainty baseline:** choose actions solely to maximize expected information gain, with task reward set to zero.

If the policies select similar actions, information-seeking may arise as a consequence of planning for reward. If they diverge, the comparison can identify when explicit uncertainty reduction explains behavior beyond reward optimization.

No information-gain bonus is added to the IRL objective. The uncertainty policy remains a separate null-hypothesis baseline.

## Experimental setting

The environment is a MiniGrid foraging task containing:

- one reward-bearing goal at a randomized location;
- reward-irrelevant objects with varied colors and shapes;
- a privileged full-state representation for initial validation; and
- a partial egocentric observation for the POMDP setting.

Distractors allow the analysis to distinguish selective sensing of reward-relevant features from generic exploration. The full-state and partial-observation agents operate on the same underlying task so that observability is the main experimental difference.

## Method

### 1. Validate IRL with full state

A PPO expert is trained using privileged simulator state. Successful trajectories provide demonstrations for a standard, single-mode IRL baseline.

The project adapts [IQ-Learn](https://arxiv.org/abs/2106.12142), which learns a soft Q-function that implicitly represents both reward and policy. The recovered reward is computed from

```text
r(s, a) = Q(s, a) - gamma * E[V(s')]
```

Because multiple rewards can rationalize the same behavior, reward correlation alone is insufficient. The full-state stage uses two validation gates:

- correlation between recovered and ground-truth reward; and
- success of the policy induced by the learned Q-function in the environment.

### 2. Move to egocentric partial observations

A recurrent PPO expert acts from the egocentric observation while carrying memory across the episode. Its successful trajectories are used to train a recurrent offline IQ-Learn agent.

The IRL agent combines a jointly trained CNN or MLP observation encoder with a GRU. The recurrent hidden state summarizes observation history and serves as a learned belief-like state for the Q-network. This tests whether a useful reward and policy can be recovered when a single observation is not Markov.

### 3. Measure active sensing

Reward-driven trajectories are evaluated using pre-specified behavioral measures:

- visited-state entropy;
- path length relative to the shortest path;
- time to first fixation on the goal;
- dwell time near the goal; and
- fixation and dwell time around reward-irrelevant distractors.

The separate uncertainty baseline scores an action by its expected reduction in Shannon entropy:

```text
information_gain(a) = H[b_t] - E_o'[H[b_{t+1} | a, o']]
```

Action agreement and behavioral metrics can then compare reward-driven sensing with explicit uncertainty minimization.

## World-model extension

The planned extension replaces the recurrent encoder with an action-conditioned stochastic latent world model:

```text
posterior:   q(z_t | observation history)
transition:  p(z_{t+1} | z_t, a_t)
```

A VAE-style latent is the initial choice because it represents uncertainty directly. A deterministic JEPA-style representation would require an ensemble or probabilistic prediction head to support the same uncertainty analysis.

The latent representation will be validated for **reward sufficiency**: it should preserve features needed to infer reward and choose actions, rather than merely predict future pixels. Active-sensing scores will be computed over a reward-relevant projection of the latent state so that reducing irrelevant uncertainty does not count as task-directed sensing.

In the corresponding belief MDP, planning uses the expected learned reward

```text
r_bar(b, a) = E_{s ~ b}[r(s, a)]

V*(b) = max_a [r_bar(b, a)
               + gamma * E_{o' ~ p(o | b, a)} V*(tau(b, a, o'))]
```

where `tau` is the belief update after an action and new observation.

## Current status

| Component | Status |
|---|---|
| MiniGrid foraging environment with randomized goal and distractors | Implemented |
| Full-state PPO expert and evaluation | Implemented |
| Recurrent PPO expert for egocentric observations | Implemented |
| Full-state IQ-Learn training and validation gates | Implemented |
| CNN/MLP + GRU recurrent IQ-Learn agent | Implemented |
| Demonstration generation for both observation settings | Implemented |
| Active-sensing metrics and Shannon-entropy baseline | Implemented and unit tested |
| Action-conditioned stochastic latent world model | Planned; interface scaffolded |
| Reward-relevance auxiliary head | Planned; interface scaffolded |
| Final policy-agreement experiments and conclusions | In progress |

The repository does not yet claim that active sensing has emerged; the final comparison depends on completing the world-model and policy-agreement experiments.

## Repository structure

```text
environment/    MiniGrid task, observation wrappers, and render checks
experts/        PPO and recurrent PPO expert training and evaluation
imitation/      Demonstration generation and full-state/recurrent IQ-Learn
analysis/       Behavioral metrics, training curves, and entropy baseline
models/         World-model and reward-relevance extension scaffolds
iq_learn/       Attributed subset of the upstream IQ-Learn implementation
scripts/        Cluster training entry points
tests/          Deterministic tests for analysis utilities
```

## Installation

Python 3.10+ is recommended.

```bash
git clone https://github.com/sruthii-m/active-sensing-irl.git
cd active-sensing-irl
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Example workflow

Train and evaluate the full-state expert:

```bash
python -m experts.train_full_state \
  --timesteps 100000 \
  --eval-episodes 50
```

Generate full-state demonstrations and fit IQ-Learn:

```bash
python -m imitation.generate_full_state_demos
python -m imitation.train_iq_full_state
```

Train the recurrent expert from egocentric observations:

```bash
python -m experts.train_partial_obs \
  --timesteps 250000 \
  --eval-episodes 50
```

Generate partial-observation demonstrations and train recurrent IQ-Learn:

```bash
python -m imitation.generate_partial_obs_demos
python -m imitation.train_iq_recurrent \
  --env partial_obs \
  --demo-path artifacts/demos/forage_partial_obs_demos.pkl \
  --grid-shape 7 7 3
```

Run the analysis tests:

```bash
python -m unittest tests.test_metrics
```

Training scripts expose additional command-line options for seeds, architectures, batch sizes, learning rates, and evaluation budgets. `experts/sweep.py` provides an Optuna sweep for the full-state PPO baseline.

## Research progression

The project follows a staged design so failures can be localized:

1. establish that the task and expert are learnable with full state;
2. verify that IQ-Learn recovers a behaviorally sufficient solution;
3. replace true state with egocentric observations and recurrent memory;
4. compare reward-driven actions with explicit uncertainty minimization; and
5. introduce a stochastic latent world model and restrict uncertainty analysis to reward-relevant features.

This progression is inspired by perception-based active sensing and by the view that perception can be formulated as learning an internal world model over observation history and actions.

## References and attribution

- Garg et al., [IQ-Learn: Inverse Soft-Q Learning for Imitation](https://arxiv.org/abs/2106.12142)
- [Perception-grounded active sensing framework](https://www.sciencedirect.com/science/article/pii/S2666389924000977)
- The algorithm core under `iq_learn/` is vendored from [Div99/IQ-Learn](https://github.com/Div99/IQ-Learn) at commit `1f5492dd26348ef11dea3467037baf4eff65c178`. See `iq_learn/README_VENDORED.md` for the retained files and project-specific adaptations.
