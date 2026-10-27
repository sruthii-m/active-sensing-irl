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

The exact Bayesian evaluator (`python -m analysis.evaluate_belief_alignment`) compares
the policy's executed argmax over all seven MiniGrid actions with information gain
over left, right, and forward. Its CSV `agreement` and `q_in_ig_maximizers` fields
use that actual policy action; pickup, drop, toggle, and done count as disagreement.
`iq_argmax`, `iq_argmax_all`, and `executed_action` identify the actual action,
while `iq_argmax_navigation` identifies the best navigation-only Q value.

JSON summaries report these metrics overall, by belief resolution, and by candidate
count. Use strict `agreement_unique_ig` (steps with a unique IG maximum) and strict
`q_in_ig_maximizers` (steps where IG is not tied across all navigation actions)
to assess policy-level active sensing. `agreement` includes all steps, using the
deterministic first IG maximizer, and
`policy_argmax_outside_navigation_fraction` uses all steps in each group.
Metrics ending in `_navigation_conditional` retain the previous comparison after
restricting the Q argmax to navigation, evaluated on the same steps; they do not
filter out non-navigation policy choices and are diagnostic only.
`spearman_navigation`, `mean_spearman_navigation`, and `rank_steps_navigation`
explicitly label the navigation-only rank comparison; constant rankings remain
undefined and are excluded from the mean. Empty summary denominators yield null.
This changes the meaning of the former navigation-only `agreement`,
`q_in_ig_maximizers`, and `iq_argmax` fields; rerun older evaluations before using
them for policy-level claims.

Schema-v2 reports also expose explicit `strict_*` and
`navigation_conditional_*` names for these metrics. The primary endpoint is
`unresolved.strict_agreement_unique_ig`, equivalent to `agreement_unique_ig`.
`actual_outside_navigation_fraction` aliases
`policy_argmax_outside_navigation_fraction`. These aliases preserve both report
interfaces; agreement including arbitrary IG tie-breaking remains diagnostic.

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

The repository does not yet claim that active sensing has emerged. The exact external Bayesian baseline below enables policy-agreement experiments without a world model.

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

## Exact state-uncertainty baseline (Experiment A)

This stage does **not** require the world-model extension. The external Bayesian
baseline is analysis only; no entropy bonus enters IQ-Learn, its encoder, or
demonstrations. Use zero distractors first. Full observation with distractors
(Experiment B) and partial observation with distractors (Experiment C) remain
separate future experimental conditions. Existing distractors have visibly
different identities from the known green goal: their presence alone does not
establish reward-identity ambiguity. Experiment B will need its own task design.

### Objects and assumptions

- `analysis/discrete_goal_belief.py` represents the conditional distribution
  `b(G | observed cells, known pose/map)` over a single static goal location.
  At reset, MiniGrid rejection sampling places the goal uniformly over interior
  cells excluding the agent's spawn. The goal precedes distractors in placement.
  The implemented exact baseline requires **zero distractors**, a known empty
  rectangular room, and known pose. It is not a belief over unknown geometry or
  localization. A singleton can be resolved by elimination without ever seeing
  the goal; `goal_seen` and `resolved` are therefore logged separately.
- Visibility comes from `gen_obs_grid()` and `get_view_coords()`, including
  occlusion. Counterfactual left/right/forward transitions use copied simulators.
  In this condition Goal and empty floor both allow overlap and transmit sight,
  so the visibility geometry does not depend on the unknown goal position.
  Pickup/drop/toggle/done cannot change geometry with no movable objects.
- The posterior is uniform on surviving candidates. Entropy is `ln(N)` and
  one-step IG is `ln(N) - (N-k)/N * ln(N-k)`, with `0 ln 0 = 0`.
  The existing generic entropy module supplies entropy and a reference explicit
  outcome enumeration. Its API remains in **bits**; new reports convert to
  **nats**.
- `analysis/alignment_metrics.py` measures action ordering, not equality of Q
  and IG magnitudes. `analysis/policy_io.py` adapts existing checkpoint APIs
  without changing training. The current GRU consumes **observations only**,
  not previous actions; it is not literally an action-conditioned history model.
- `analysis/expert_likelihood.py` computes teacher-forced
  `mean_t log_softmax(Q(h_t)/alpha)[a_expert]`, resetting recurrence per episode.
  It also reports NLL, top-1 accuracy, policy entropy, per-episode log sums and
  both step-weighted and episode-weighted means. This is conditional action
  likelihood, not normalized joint trajectory probability. Initial-state
  probability is omitted; recorded transitions are assumed valid/deterministic.
- `analysis/reward_maps.py` evaluates every action from each visited history,
  using `Q(h,a) - gamma*(1-terminated)*V(h'_a)` and the actual counterfactual
  next observation. Truncations bootstrap, matching demonstration `dones` and
  IQ loss. Maps average over sampled histories/directions at a cell, not over a
  unique recurrent state associated with that cell. They use the online Q/V
  network, as in existing post-training reward inference.

### Run the isolated condition

Environment factories and expert/demo/IQ training CLIs accept `--size`,
`--num-distractors`, and `--view-size` (odd, at least 3). Defaults remain 10,
3, and 7. `make_full_obs` provides the full symbolic grid;
`make_full_obs_learning_env` keeps the existing privileged five-feature view.
For Experiment A, use **identical settings throughout**:

```bash
python -m experts.train_partial_obs --num-distractors 0 --view-size 5 \
  --save-path artifacts/experts/state_only_fov5
python -m imitation.generate_partial_obs_demos --num-distractors 0 --view-size 5 \
  --model-path artifacts/experts/state_only_fov5 --seed 0 \
  --out-path artifacts/demos/state_only_fov5_train.pkl
python -m imitation.train_iq_recurrent --env partial_obs \
  --num-distractors 0 --view-size 5 --grid-shape 5 5 3 \
  --demo-path artifacts/demos/state_only_fov5_train.pkl \
  --save-path artifacts/iq_learn/state_only_fov5.pt
```

Use 3, 5, or 7 for the FOV sweep and train matching checkpoints for each shape.
Full-state equivalents expose the same condition arguments. New demo files
record environment settings and the reset seed interval (`seed_start` through
`seed_start + attempts - 1`). Generate held-out demos from a nonoverlapping seed
interval, excluded from fitting and model selection. Existing demos do not
contain provenance, so their held-out status cannot be established automatically.
By default generators retain successful episodes (and partial demos apply a length filter):
likelihood on these datasets measures fidelity conditional on that selection.

```bash
# Trained reward-only policy rollout, zero distractors enforced
python -m analysis.evaluate_belief_alignment \
  --checkpoint artifacts/iq_learn/state_only_fov5.pt \
  --init-temp 0.001 --grid-shape 5 5 3 --view-size 5 \
  --episodes 100 --seed 10000 --output artifacts/analysis/alignment_fov5.csv

# Recurrent partial-observation likelihood on a separately generated held-out file
python -m analysis.expert_likelihood \
  --checkpoint artifacts/iq_learn/state_only_fov5.pt \
  --init-temp 0.001 --grid-shape 5 5 3 \
  --heldout-demos artifacts/demos/state_only_fov5_heldout.pkl \
  --output artifacts/analysis/partial_likelihood.json
# Full-state feedforward (or recurrent MLP) likelihood
python -m analysis.expert_likelihood \
  --checkpoint artifacts/iq_learn/forage_full_state_qnet.pt --init-temp 0.001 \
  --heldout-demos artifacts/demos/full_state_heldout.pkl \
  --output artifacts/analysis/full_likelihood.json

python -m analysis.reward_maps --env partial_obs \
  --checkpoint artifacts/iq_learn/state_only_fov5.pt \
  --init-temp 0.001 --grid-shape 5 5 3 --view-size 5 --seed 10000 \
  --output artifacts/analysis/reward_fov5
# For a full-state checkpoint use --env full_state and omit --grid-shape.
```

Legacy checkpoints contain weights only: supply the actual training temperature
(`--init-temp` is required), CNN shape if applicable, and training discount
(`--gamma`, default .99). Architecture widths are inferred from weights.
Evaluating a distractor-trained checkpoint here is a distribution-shift test,
not a clean Experiment A result. The scripts do not certify training provenance.

### Outputs and interpretation

Alignment writes per-step CSV and a sibling JSON summary with run settings,
success rate, unresolved/resolved summaries, and candidate-count bins. CSV logs
pose, privileged goal, seen/resolved status, entropy, all seven Q values,
three IG values, ties, candidate and full-policy argmax, executed action,
Spearman correlation, and new visible cells per candidate action. The actual
rollout executes the **seven-action** learned argmax. The primary endpoint is
`unresolved.strict_agreement_unique_ig`: the actual policy argmax must match
the unique IG maximizer. Pickup/drop/toggle/done count as disagreement.
`actual_outside_navigation_fraction` reports their frequency over all steps in
each summary group (including IG ties); its count and denominator are included.
`navigation_conditional_agreement_unique_ig` retains the old comparison after
restricting Q to left/right/forward, on the same timesteps. This is an action-space
conditional diagnostic, not conditioning on the policy having chosen navigation.
Spearman is explicitly `navigation_conditional_mean_spearman` and is also only a
diagnostic. Strict and navigation-conditional maximizing-set membership are
reported separately on informative states.

New reports use `schema_version: 2` and identify the primary metric explicitly.
CSV `iq_argmax` and `iq_argmax_all` now both denote the executed seven-action
argmax; `iq_argmax_navigation` denotes the restricted argmax. Explicit strict
and navigation-conditional metric names are preferred; the merged remote names
remain as compatibility aliases. Re-run evaluation to obtain the new schema;
do not mix older navigation-only agreement reports with these strict metrics.

Unique-IG agreement excludes every tied IG maximum. Membership of the Q argmax
in the IG maximizing set is reported separately on informative states; states with all IG values tied are excluded. Spearman uses average ranks and is undefined (`null`)
when either vector is constant. Counts accompany denominators. Q ties are
logged; deterministic first-action selection makes them reproducible, but those
choices should not be interpreted as a strong preference. Resolved states have
zero IG, so their agreement/correlation is structurally uninformative, not a
policy failure. The one-step entropy policy is a local comparator and can stall
on ties; it is not an optimal multi-step search policy.

Likelihood writes JSON. Reward visualization writes PNG (occupancy, mean of all
seven actions, seven individual action maps, goal marker) and NPZ (maps,
occupancy, individual pose/reward samples, goal, settings). Each map uses one
fixed episode/layout; repeat seeds separately rather than averaging moving goal
locations into a misleading spatial map. Blank cells are unsampled.

High success establishes task completion; held-out likelihood measures expert
policy reproduction; reward maps show sampled spatial structure, not unique
reward identification. Unresolved Q/IG ordering agreement supports behavior
consistent with information seeking under this state-uncertainty model. None
proves that the deterministic GRU contains calibrated Bayesian uncertainty.
There is no `H(h_t)` calculation.

Before large runs, settle with the PI: known map/pose assumptions; whether to
include action history in a later encoder comparison; how to construct genuine
reward-identity ambiguity for B; temperature/provenance of older checkpoints;
held-out seed ranges and success-filtering bias; and whether the main agreement
endpoint should require a unique Q maximum as well as a unique IG maximum.
Pre-register entropy bins, episode/seed-level confidence intervals, and matched
layouts across FOVs. Lower FOV need not monotonically increase entropy along
policy-dependent trajectories, so test that manipulation rather than assume it.

## PACE workflow for Experiment A

`scripts/state_uncertainty.sbatch` follows the existing batch scripts: account
`gts-awu36`, QoS `embers`, one GPU, eight CPUs, 32 GB, eight hours, modules
`anaconda3 cuda`, and Conda environment `active-sensing-irl`. These are copied
from this repository, not independently verified against current PACE settings.
Submit from the repository root **after syncing these code changes**. Create
`logs` before submission: Slurm opens its output files before the script runs.

Start with FOV 7 and training seed 0. Submit each stage after the previous one
finishes successfully:

```bash
mkdir -p logs
sbatch scripts/state_uncertainty.sbatch expert partial_obs 7 0
# Inspect the expert's final success rate before continuing.
sbatch scripts/state_uncertainty.sbatch demos partial_obs 7 0
sbatch scripts/state_uncertainty.sbatch iq partial_obs 7 0
sbatch scripts/state_uncertainty.sbatch evaluate partial_obs 7 0

squeue -u "$USER"
# Replace JOBID with the ID returned by sbatch.
tail -f logs/state-uncertainty-JOBID.out
cat logs/state-uncertainty-JOBID.err
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode
```

Do not submit all four commands together without dependencies. After accepting
the expert, the remaining stages can be chained:

```bash
demos_job=$(sbatch --parsable scripts/state_uncertainty.sbatch demos partial_obs 7 0)
iq_job=$(sbatch --parsable --dependency=afterok:"$demos_job" scripts/state_uncertainty.sbatch iq partial_obs 7 0)
sbatch --dependency=afterok:"$iq_job" scripts/state_uncertainty.sbatch evaluate partial_obs 7 0
```

The stages perform:

1. **Expert:** train recurrent PPO for 250,000 timesteps, then evaluate 100 episodes.
2. **Demos:** generate 400 successful training episodes and 100 unfiltered held-out episodes.
   Training seeds start at 100000; held-out seeds start after all training reset
   attempts plus a gap. The existing partial-demo length filter remains 100 steps for training; held-out episodes retain all outcomes and lengths.
3. **IQ-Learn:** use the existing 400-demo batch configuration: CNN, latent width
   128, GRU width 256, up to 200,000 updates, evaluations every 3,000 updates,
   temperature .001, discount .99, and the existing plateau rule.
4. **Evaluation:** calculate held-out likelihood, 100 reward-policy alignment
   rollouts, and reward maps for three layouts. Evaluation seeds start at 10000,
   shared across FOV conditions and separate from demo-generation seeds.

Files go under `artifacts/state_uncertainty/partial_obs_fov7_seed0/`:
`expert.zip`, `train.pkl`, `heldout.pkl`, `iq.pt`, `likelihood.json`,
`alignment.csv`, `alignment.json`, and `reward_0` through `reward_2` PNG/NPZ
pairs. Stage commit files record the repository HEAD; commit your changes for
those hashes to identify the actual code. Stage reruns reuse these paths.
Download PNGs to inspect them, or open them through your remote IDE.

For FOVs 5 and 3, repeat all stages with that FOV argument; do not reuse the
FOV-7 checkpoint or demos. Repeat training seeds 1 and 2 after the first pipeline
works. Use `full_state 7 0` in the same commands for a full-state control: that
branch trains PPO and feedforward IQ-Learn, and computes likelihood/reward maps.
It does not run the partial-observation alignment evaluator. Full-state training
prints its own success estimate. The single batch template reserves a GPU even
for the CPU analysis stage for consistency with the existing scripts; tune that
allocation to your PACE partition if desired.

### How to judge the run

- **Job execution:** require successful Slurm exit status and nonempty output
  files. An eight-hour timeout is not a completed experiment; these training
  scripts do not resume optimizer state. Increase requested time if needed.
- **Expert quality:** inspect the 100-episode success rate and the number of demo
  attempts needed to retain 400 episodes. A practical pilot target is at least
  90% expert success, not a validated scientific cutoff. If the expert is weak,
  improve its training budget/settings first. Demo generation retries until it
  fills the requested count, so a weak expert can consume the whole allocation.
- **Imitation fidelity:** compare IQ success with expert success, then inspect
  `expert_top1_accuracy` and `negative_log_likelihood` in `likelihood.json`.
  A uniform seven-action policy has NLL `ln(7) ≈ 1.946` and expected accuracy
  `1/7 ≈ 14.3%`. This is only a sanity reference; compare against an empirical
  action-frequency baseline too. Temperature .001 makes confident mistakes
  expensive in NLL, so high greedy success can coexist with poor likelihood.
  Evaluate held-out data once the checkpoint/configuration is selected; repeated
  tuning against that report turns it into validation data.
- **Belief correctness in rollouts:** within each CSV episode, candidates and
  entropy should never increase; entropy should equal `ln(candidates)`.
  Seeing the goal must leave one candidate. All three IG values must become
  zero once resolved. The belief can also resolve by elimination before sight.
  These checks assess the analysis mechanism, not whether the learned policy
  senses information.
- **The research endpoint:** examine `unresolved.strict_agreement_unique_ig`
  and `actual_outside_navigation_fraction`, together with candidate-count bins
  and denominator counts. Strict agreement uses the action actually executed.
  Navigation-conditional agreement and Spearman are secondary diagnostics and
  cannot by themselves support policy-level active-sensing claims. `resolved`
  correlations remain null because IG is constant. Few informative states mean
  limited alignment evidence, even if conditional scores look strong.
- **Controls and replication:** do not treat 1/3 agreement as a sufficient null
  test when action preferences are imbalanced. Compare with action-frequency or
  shuffled pairings that preserve phase/uncertainty bins, and aggregate uncertainty
  across independent episodes/training seeds, not correlated timesteps. The
  current JSON reports descriptive point estimates, not confidence intervals or
  a significance test. Check actual entropy distributions before claiming that
  smaller FOV induced more uncertainty.
- **Reward maps:** look for coherent action-specific structure along sampled
  paths; unsampled cells are blank. A plausible goal-related pattern is useful
  evidence to inspect, not proof of unique reward recovery. Three pictures are
  examples, not an aggregate statistical result.

### Paired expert/IQ policy fidelity

Add `--expert-model artifacts/experts/forage_partial_obs.zip` to
`analysis.expert_likelihood` to score the saved RecurrentPPO expert and IQ policy
on exactly the same recorded observation histories. For a feedforward PPO
expert also pass `--expert-kind ppo`. Each model's recurrent state resets per
episode. Neither policy generates alternative histories during this comparison.
The loaded expert must be the one that generated the demonstrations, using the
same observation preprocessing.

The JSON retains the previous IQ-only metrics and adds `paired` and
`paired_by_outcome` (success/failure/unknown). These contain step-weighted expert
log likelihood, the expert-minus-IQ log likelihood gap, exact categorical
`KL(expert || IQ)`, and greedy policy agreement. Individual episode entries
contain the corresponding means and trajectory log likelihood sums. A sibling
CSV records each timestep, both policies' action probabilities and chosen-action
log likelihoods, likelihood gap, cumulative gap, KL, and greedy actions.
Plot CSV `cumulative_likelihood_gap` versus `timestep`, grouped by `episode`, to
locate divergence; its slope is more comparable across lengths than its total.
All log quantities are in nats.

A gap near zero alone does not imply equal policies. KL zero means distributions
match on evaluated histories; high greedy agreement means preferred actions
match. For deterministic expert demonstrations, the empirical action likelihood
gap can be negative and is not a KL estimate. The exact distribution KL compares
the underlying stochastic PPO distribution; greedy agreement and demonstrated
action likelihood assess reproduction of the deterministic demonstrator.
Neither establishes agreement on histories absent from the evaluation set.
Temperature must remain the actual IQ training temperature for this comparison.

To assess expert reproduction independently of expert task success, generate
unfiltered evaluation episodes (the training-demo default remains filtered):

```bash
python -m imitation.generate_partial_obs_demos \
  --model-path artifacts/experts/forage_partial_obs.zip \
  --num-episodes 20 --seed 200000 --num-distractors 3 --view-size 7 \
  --keep-all --out-path artifacts/demos/forage_partial_obs_eval_all.pkl
python -m analysis.expert_likelihood \
  --checkpoint artifacts/iq_learn/forage_partial_obs_qnet.pt \
  --init-temp 0.001 --grid-shape 7 7 3 \
  --expert-model artifacts/experts/forage_partial_obs.zip \
  --heldout-demos artifacts/demos/forage_partial_obs_eval_all.pkl \
  --output artifacts/analysis/paired_likelihood.json
```

Use seeds excluded from fitting and model selection. The PACE `demos` stage now
uses `--keep-all` for held-out episodes and its `evaluate` stage loads the saved
expert for paired metrics. Existing held-out files remain filtered until you
regenerate them; their JSON selection metadata is reported when available.

### Local batch pipeline

Run the complete zero-distractor partial-observation pipeline locally with:

```bash
bash scripts/run_state_uncertainty_local.sh 3 0
```

Arguments are FOV (3/5/7, default 3), training seed (default 0), and an optional
new output directory. Activate your project Python environment first, or set
`PYTHON=/path/to/python`. The script trains the expert, generates 400 filtered
training demos and 100 unfiltered held-out episodes on separate seed intervals,
trains recurrent IQ-Learn on CPU, and produces paired likelihood, strict
alignment, and a reward map. It uses the same training budgets/settings as the
local FOV-7 run. Evaluation layouts start at seed 10000 across conditions.

Stage progress is printed; full output is in stage-specific logs under
`artifacts/state_uncertainty/partial_obs_fov3_seed0_local/`. `status.txt` records
the current stage or completion/failure; `commands.sh` records invoked commands.
Keep the terminal open and the computer awake. Training-demo generation retries
until 400 successful short episodes are collected, so weak experts can make
that stage slow. Existing run directories are refused, and failed commands stop
the pipeline. This script does not resume interrupted training. Change FOV/seed
for each replication; use the third argument for a deliberately separate rerun.

### Action-frequency reference and episode confidence intervals

After generating alignment CSVs, run this analysis without retraining:

```bash
python -m analysis.alignment_uncertainty
```

It reads `artifacts/state_uncertainty/*/alignment.csv` and writes
`artifacts/analysis/alignment_uncertainty.json`. Options include `--root`,
`--output`, `--resamples` (default 5000), and `--seed` (default 0).

The eligible population is the same as the primary strict metric: unresolved
states with one navigation IG maximizer. Within candidate-count bins 2–3, 4–7,
8–15, 16–31, 32–63, and 64+, the reference is
`sum_a p(actual action=a) * p(IG argmax=a)`. Bin references are weighted by their
eligible timestep counts. All seven actual-action frequencies are retained;
non-navigation actions cannot match a navigation IG maximizer. This is the exact
expected agreement under uniform within-bin permutation of actual-action labels.
It controls marginal action preferences and coarse uncertainty level, but not
pose, visibility geometry, detailed history, or multi-step information value.
Sparse bins and their episode/action counts are included in the report.

The script reports observed agreement, reference agreement, and their difference
(in percentage points when printed). Confidence intervals are 95% percentile
intervals from resampling whole episodes with replacement, including episodes
with no eligible steps. Each resample recomputes both the step-weighted metric
and reference frequencies. It does not treat timesteps as independent and does
not report a timestep-shuffle p-value. Zero-denominator draws are omitted and
counted; runs with no eligible states report null estimates. Episodes with longer
eligible phases still carry more weight, matching the original strict metric.

These are exploratory diagnostics: bins were chosen after inspecting pilot
results, the intervals condition on each fixed trained policy, and no correction
for multiple comparisons or training-seed uncertainty is applied. An excess
interval above zero supports association beyond this specific marginal-frequency
reference; it does not establish causal information seeking or calibrated GRU
beliefs. Do not infer a FOV effect from overlapping or nonoverlapping per-run
intervals alone; a matched-layout contrast and more training seeds are needed.
The report evaluates each checkpoint separately rather than pooling trajectories
from different trained policies as independent replications.

### Decode goal-location belief from frozen recurrent memory

Run this after the zero-distractor experiment; it uses the saved IQ checkpoint
without retraining the policy or expert:

```bash
bash scripts/run_belief_decoding_local.sh 3 0
bash scripts/run_belief_decoding_local.sh 3 1
```

Defaults: 300 on-policy episodes starting at reset seed 200000, 100 probe epochs,
CPU, and one probe initialization. Set `PYTHON` to choose an interpreter,
`CHECKPOINT` to use a different saved policy, or `EPISODES`, `EPOCHS`, and
`PROBE_SEED` to change the probe run. A third positional argument selects a new
output directory; existing directories are refused. The module also exposes
`--rollout-seed`, `--width`, `--batch-size`, `--lr`, and environment options:

```bash
python -m analysis.evaluate_belief_decoding --help
```

The target is the complete external posterior over interior goal cells, including
zero probability on eliminated cells. The policy is frozen. Three small MLP
probes predict the same posterior from (1) the current flattened observation,
(2) the frozen GRU evaluated with zero memory on that observation, and (3) the
actual recurrent hidden state. Every probe additionally receives the same known
position and direction so a world-coordinate target does not unfairly penalize
observation-only features. The actual goal is never a probe input. Reset-memory
features are a diagnostic; rollout actions always use the original memory.

Train/validation/test splits are 60/20/20 by unique initial configuration (goal,
spawn position, direction), keeping repeated layouts and entire episodes together.
These layouts are held out from **probe fitting**, not necessarily original IQ
training. Input normalization uses training samples only. Training weights each
episode equally. The best decoder epoch minimizes validation unresolved-state KL;
test data do not select the model. Probes have the same hidden width and optimizer,
but observation and hidden-state input dimensions differ, so parameter counts are
reported. The two GRU probes have matching capacity. A negative result can also
reflect insufficient probe capacity or fitting, not absence of belief information.

Outputs under `artifacts/belief_decoding/fov3_seed0_probe0/`:

- `summary.json`: phase-specific KL and cross-entropy, split counts, selected
  epochs, policy hash, and paired baseline-minus-history KL intervals.
- `belief_decoding.png` / `.pdf`: held-out comparison and paired improvements.
- `rollouts.npz`: observations, hidden/reset-hidden states, pose, posterior,
  episode/timestep/action, layout IDs, and split labels (0=train, 1=validation,
  2=test). `cells` gives the world-coordinate order of posterior entries.
- `test_predictions.npz` and `test_episode_metrics.csv`: test posterior predictions
  and per-episode phase-specific KL for inspection and reanalysis.
- `*_probe.pt` / `*_training.json`: decoder weights, normalization, and validation
  curves for all three inputs.

The primary metric is unresolved-state `KL(b || decoded_b)`, averaged within each
held-out episode and then across episodes. Lower is better; zero means identical
posteriors. Cross-entropy equals target entropy plus KL, so raw cross-entropy is
not directly comparable across different uncertainty levels. Resolved states are
reported separately. Positive baseline-minus-history KL favors recurrent memory.
The 95% percentile intervals use 2000 paired layout-bootstrap draws, preserving
within-episode dependence and repeated layouts. They condition on this policy,
probe fit, and split; repeat probe/policy seeds to assess training variability.

This tests whether belief information is **decodable**, not whether the policy
causally uses it. Stronger evidence requires matched-history behavior tests or
controlled memory interventions. Entropy prediction alone is insufficient because
different spatial posteriors can have the same entropy.

#### Repeat decoders while keeping the policy and evaluation split fixed

The initial 100-epoch probes are exploratory. Check fitting sensitivity before
interpreting a small advantage as reliable memory evidence:

```bash
MPLCONFIGDIR=/tmp/active-sensing-mpl python -u -m analysis.repeat_belief_decoding
```

This reuses the two completed FOV 3 datasets above, fits probe initializations
0/1/2 for 500 epochs each, selecting weights by validation error. Early stopping
is disabled by default because validation error can improve after a long plateau.
It never retrains IQ, recollects trajectories, or changes the saved
layout splits. `--sources`, `--probe-seeds`, `--epochs`, `--patience`, and
`--output-dir` configure the repeat. Existing output directories are refused.

The combined `artifacts/belief_decoding/robustness_fixed_budget/robustness.png` (also PDF),
`runs.csv`, and `summary.json` report every initialization separately. Each fit
retains validation curves, selected epoch, epochs run, and a flag when the chosen
epoch reaches the requested budget. `validation_curves.png` / `.pdf` show every
validation curve with its selected epoch; a best epoch near the budget means
convergence is not established. Report all fits rather than choosing the one
with the largest test advantage. Per-fit intervals use the same held-out layouts;
repeating initializations does not multiply the number of independent test cases.
These follow-up comparisons reuse an already inspected test set, so remain
exploratory. Independent new episodes/layouts and additional policy seeds are
needed for confirmation; capacity/optimization limitations remain possible.

The underlying evaluator also accepts `--cached-rollouts path/to/rollouts.npz`,
requiring its adjacent `summary.json`. It checks the checkpoint fingerprint,
environment/rollout configuration, and split isolation before fitting. The new
`--split-seed` (default 0) controls data splitting independently of `--probe-seed`.
When using cached rollouts the original split seed must match; legacy datasets
use their original probe seed as the split seed. All normalization is still fit
using training samples only.

### Matched histories: does remembered evidence change decisions?

Run the controlled-history evaluation on existing checkpoints:

```bash
bash scripts/run_matched_histories_local.sh 3 0
bash scripts/run_matched_histories_local.sh 3 1
```

`analysis.evaluate_matched_histories` starts both histories from the same reset.
One traverses a square with left turns, the other a square with right turns
(side lengths 1, 2, 3). Both must return to the same junction. An identical forward
step then moves the query to a nearby observation (`--approach-steps 1`, default).
Without this shared approach, the closed loops can exhaust all next-step IG at
the junction; use `--approach-steps 0` to reproduce that diagnostic. No teleporting
or hidden-state replacement is used. Forced actions are chosen independently of
the policy's Q values and the actual goal location.

Matched pairs must have exactly the same final observation, position, direction,
and elapsed timestep, and reach the intended endpoint of the shared approach.
Terminal/truncated histories and endpoint mismatches (e.g. from wall collisions)
are excluded and counted. Primary analysis requires both Bayesian beliefs to be
unresolved and their candidate sets different. All seven actual policy actions
are retained. Belief and hidden state are updated from each observation once,
including the final observation before querying Q.

A separate control rotates left versus right for the same total number of steps,
then takes the same approach. Controls must have the same final Bayesian belief
but distinct observation sequences. This tests sensitivity to observation order
without changing the Bayesian goal posterior. Rotation and movement histories
have different coverage and motion patterns, so comparing their aggregate scores
is not an isolated causal effect of belief.

Outputs under `artifacts/matched_histories/fov3_seed0_approach1/`:

- `summary.json`: matched/excluded counts, distinct-reset counts, policy total
  variation (TV), actual-argmax change rate, and directional IG results.
- `pairs.csv`: every accepted pair, including resolved and tied-IG cases, with
  explicit eligibility flags and per-pair results.
- `matched_traces.npz`: both action/observation sequences, poses, endpoint hidden
  states, seven Q values, three IG scores, and candidate sets for every accepted
  pair. Keys start with `pair{pair_id}_a_` or `pair{pair_id}_b_`.
- `matched_histories.png` / `.pdf`: changes and directional IG comparison.

Policy TV is `0.5 * sum(abs(pi_A - pi_B))`; 0 means identical distributions and
1 means disjoint support. Where both histories have different unique IG winners
`a_A` and `a_B`, the directional score is
`0.5 * (pi_A(a_A) - pi_B(a_A) + pi_B(a_B) - pi_A(a_B))`.
Positive values mean each history shifts probability toward its own IG winner;
the report separately counts whether both actual argmax actions match those
winners. No eligible pairs produces null values, not a zero effect. The reference
is one-step information gain, not a reward-maximizing belief-space planner.

Metrics average distances within each reset seed, then weight reset seeds equally.
95% percentile intervals resample 2000 reset-seed clusters, keeping repeated
history lengths together. Default: 300 fresh reset seeds starting at 300000.
These are exploratory fixed-policy estimates. The shared approach was introduced
after inspecting the zero-approach geometry; independent layouts should be used
for confirmation. Forced histories can be outside the policy's training/rollout
distribution. Action changes establish history sensitivity, not causal use of
the decoded Bayesian belief. Substantial changes in the same-belief control are
evidence that this posterior alone does not explain policy behavior in the test.

### Reward-maximizing Bayesian reference

Compare the saved matched histories against a reward objective rather than
one-step information gain (no IQ retraining):

```bash
MPLCONFIGDIR=/tmp/active-sensing-mpl python -u -m analysis.evaluate_belief_reward
```

Defaults read both `artifacts/matched_histories/fov3_seed{0,1}_approach1` runs,
evaluate horizons 3 and 5, and write `artifacts/belief_reward/policy{0,1}/`.
Use `--sources`, `--horizons`, and `--output-dir` to select other traces or a new
run. Existing output roots are refused. Each output includes `summary.json`,
`pairs.csv` (all seven planner Q values, tied maximizers, and IQ actions), and
`belief_reward.png` / `.pdf`.

`analysis.belief_reward_planner.BeliefRewardPlanner` solves the finite-horizon
belief recursion for the known empty room and uniform goal-location posterior:

```text
Q_H(pose,b,a) = E[r + gamma * V_(H-1)(next_pose,next_b)]
V_H = max over all seven actions Q_H; V_0 = 0
```

The reward is the actual MiniGrid goal-entry reward,
`1 - 0.9 * arrival_step / max_steps`. Goal entry terminates; reaching the time
limit prevents further continuation but still allows reward on that final step.
Other actions receive zero immediate reward. Pickup/drop/toggle/done are identical
no-ops in this zero-distractor environment. The planner branches on seeing each
possible goal, seeing no goal, or entering the goal. Observations use MiniGrid's
visibility mask on an independent goal-free geometry; the actual hidden goal is
not an input. Once the goal is known, shortest orientation-aware paths give exact
remaining values within the horizon. Unknown-goal states use cached belief-tree
recursion. An independent test constructs observation branches directly from
MiniGrid simulator transitions and checks the planner's expected Q values.

This is exact **within the specified finite horizon**, with zero terminal value,
not a full-episode or infinite-horizon optimal policy. Goals beyond the planning
horizon may receive no value, creating ties or changing preferred actions as the
horizon increases. Report horizon and unique-winner coverage with every result.
The goal is a reward-planning reference, not a claim that the IQ network must
match an optimal reference or that differing behavior demonstrates absence of
belief use.

Primary directional comparisons require two different, unique reward-maximizing
actions for the two histories. The report separates:

- Probability support shift toward each history's own reward winner versus the
  other history's winner (positive is a favorable shift).
- Whether both actual seven-action IQ argmax choices match those winners.
- Expected finite-horizon action regret: `max(Q_B) - Q_B(actual IQ action)`.
- Reward-reference versus IG disagreement when both winners are unique.
- Tied/unique-reference coverage, including all-zero short-horizon cases.

The source contains 178 unresolved different-belief pairs across 131 reset seeds.
At horizon 3 there are no pairs with different unique reward winners. At horizon
5 there are 43 such pairs across 40 resets. For those pairs, seed-weighted both-
actions agreement is 7.5% for policy seed 0 and 0% for seed 1. Probability support
shift is +0.168 (95% interval +0.054 to +0.279) and -0.064 (-0.162 to +0.042),
respectively. This does not establish consistent reward-planner alignment across
the two policies. Intervals resample reset seeds, preserve multiple distances
within a seed, and condition on the fixed policy and reused forced histories.

### Natural matched histories and development memory interventions

The next diagnostic removes forced histories and probes a candidate mechanism:

```bash
bash scripts/run_belief_intervention_local.sh 0
bash scripts/run_belief_intervention_local.sh 1
```

The wrapper uses existing FOV 3 IQ checkpoints and the preselected initialization-0
500-epoch belief decoders; it does not retrain them. `PYTHON`, `EPISODES`, and
`PROBE` are configurable. A second argument chooses a new output root; existing
roots are refused. The modules can also be run separately via
`python -m analysis.evaluate_natural_histories --help` and
`python -m analysis.intervene_belief_memory --help`.

Natural-history selection collects 600 greedy on-policy episodes starting at seed
400000, excluding initial layouts present in the earlier belief-probe dataset.
It requires exactly identical current observations, known pose, and elapsed time,
but distinct observation sequences from different episodes. Pair selection does
not inspect IQ actions or planner agreement. Episodes are disjoint within each
comparison group. At most 100 different-belief pairs and 100 same-belief controls
are selected using a fixed matching seed. Controls and primary pairs can share
an episode across groups, so their separate intervals do not establish a causal
between-group contrast. The resulting matched samples are conditional selections,
not estimates over all rollouts.

Split 2 is reserved: it is excluded from matching, planning, and intervention
selection. Development uses novel layouts from splits 0/1. Each candidate action
must be a unique reward maximizer at horizons 5 and 7, agree between those
horizons, and have a Q gap of at least 0.0001. Interventions require different
stable winners for the two histories. This is stability at the tested horizons,
not a guarantee of full-episode optimality. Saved rollouts, pair lists, reference
values, policy hashes, and plots make the selection auditable.

For each eligible pair, the intervention runs in both directions. It changes
only the hidden input to the frozen belief decoder, optimizing cross-entropy
against the other history's full posterior. The externally supplied pose stays
fixed. The policy is frozen and its outputs are never used in the optimization
loss. The edit is bounded to RMS 0.1 in training-standardized hidden coordinates,
uses 200 optimization steps, and keeps the best decoder-loss iterate including
the zero edit. Each edit has 20 random controls and 20 controls projected into
the local nullspace of the decoder's centered-logit Jacobian, all matched in
standardized L2 norm. The nullspace control suppresses first-order decoder effects;
it is not guaranteed to preserve the posterior through nonlinear boundaries.

The action outcome is the change in `pi(target reference action) -
pi(source reference action)`. Positive means the predicted directional shift.
The report also gives the targeted effect minus each control mean. Bootstrap
intervals average both directions within a pair and resample pairs, not individual
random controls. Decoder CE improvement is reported independently. Restoring the
original memory is checked for deterministic reproducibility; this is not a
navigation-level behavioral rescue experiment.

Initial development outputs are saved in:

- `artifacts/natural_histories/fov3_seed{0,1}/`: selection, complete rollouts,
  `intervention_candidates.json`, `reference_values.npz`, and natural-history plots.
- `artifacts/belief_interventions/fov3_seed{0,1}/`: `summary.json`, direction-level
  results, before/after memories, control vectors, and `interventions.png` / `.pdf`.

There were 9 eligible development pairs for policy seed 0 and 8 for seed 1;
120 novel episodes per policy remained reserved. Target decoder cross-entropy
improved by 0.608 and 0.798 nats, respectively, but the target-versus-source action
probability margin shifted by -0.029 and -0.018. Effects beyond random and
local decoder-null controls were also negative in these small development samples.
The method changed what the decoder reports without producing the hypothesized
policy shift. No edit parameters were retuned to reverse that result.

These are preliminary mechanistic diagnostics, not proof of either belief use or
its absence. Edits can exploit the decoder, leave the natural hidden-state
manifold, or change other latent information. Keeping pose inputs fixed does not
prove pose information inside the GRU is preserved, and navigation competence
has not been tested after edits. Candidate counts are small. A stronger claim
requires validated, selective interventions and prespecified evaluation on the
reserved data; that confirmation has not been run.

### Behavioral explanations and belief-update consistency

Run the frozen-policy analysis on existing belief-decoding rollouts:

```bash
bash scripts/run_belief_consistency_local.sh 3 0
bash scripts/run_belief_consistency_local.sh 3 1
```

`ROLLOUTS` overrides the cached `rollouts.npz`; `EPOCHS` (default 100),
`PROBE_SEED`, `RANDOM_SEED`, and `PYTHON` are configurable. A third positional
argument selects a new output directory. Existing output directories are refused.
No environment rollouts or IQ/expert training are required. Checkpoint provenance
and the original layout-disjoint train/validation/test split are checked.

`analysis.evaluate_belief_consistency` fits four action models: observation;
observation plus four previous actions and visited-position mask; Bayesian
posterior; and observation/history plus posterior. All receive known pose and
elapsed time. Targets are the recorded **greedy IQ actions**, not expert actions.
The primary behavioral comparison is history-only NLL minus history-plus-belief
NLL on unresolved test steps: positive favors additional belief information.
This measures predictive explanation, not causal mediation. Equal hidden widths
are used, but input dimensions and parameter counts differ; capacity is not
exactly matched. All model parameter counts are reported.

Posterior decoders receive observations, reset trained memory, full trained
memory, or an entirely untrained encoder of the same architecture (including
random CNN/MLP/GRU). The random encoder sees precisely the same observations and
resets at episode boundaries. Each decoder also receives known pose and elapsed
time. Normalization uses training data; validation chooses the checkpoint;
test data do not affect fitting. One random initialization is not representative
of all random networks; repeat seeds before drawing general conclusions.

For consecutive nonterminal records with an unresolved prior, the update metric
is `sum(abs((decoded_b[t] - decoded_b[t-1]) - (b[t] - b[t-1])))`.
It is reported separately when the true posterior changes and when it does not.
Episode boundaries and transitions whose prior is resolved are excluded. A
stationary decoder reference reveals how much error results from making no
update. Exact posterior predictions give zero error as a metric sanity check;
this is **not** an independently trained positive-control agent. Increment
tracking is a descriptive temporal test, not a direct test of the Bayesian update
operator, posterior calibration, or causal use. Cached data omit the final
terminal observation.

Outputs under `artifacts/belief_consistency/` include `summary.json`, fitted
models with training histories, held-out predictions, and `consistency.png` /
`.pdf`. Scores weight episodes equally; paired intervals bootstrap initial
layouts. Intervals omit training-seed variability. These test layouts have been
used in earlier exploratory analyses, so they are not a fresh confirmatory test.
Selection at the epoch budget should motivate a prespecified convergence study,
not tuning against test performance. These diagnostics do not replace selective
interventions, a separate explicit-belief agent control, or new held-out layouts.

### Positive-control validation and fitting-seed repeats

```bash
bash scripts/run_belief_validation_local.sh 0
bash scripts/run_belief_validation_local.sh 1
```

These commands reuse the FOV-3 cached histories and run fitting seeds 0, 1, and 2,
with 200 epochs and an approximately 20,000-parameter budget per MLP. They do not
retrain IQ or the expert. `ROLLOUTS`, `EPOCHS`, `PARAMETER_BUDGET`, and `PYTHON`
override defaults; a second positional argument chooses a fresh output directory.

`analysis.validate_belief_consistency` adds an explicit reference agent whose
seven-action probabilities are a softmax of Bayesian expected information gain
(temperature 0.25 nats). It computes observation geometry without accessing the
actual hidden goal or IQ outputs. This is a **constructed information-seeking
agent**, not a learned expert or an optimal reward policy. It is queried on the
same cached IQ histories, so action history features remain those of the IQ
rollout. No new reference-agent trajectories are collected. The control tests
whether the explanatory models detect known belief dependence on these inputs.
Its action probabilities, rather than sampled actions, are soft training targets;
compare within-target cross-entropy reductions, not absolute cross entropy
between the IQ and reference targets. Resolved posteriors produce uniform
reference probabilities; primary reporting excludes resolved steps.

For each target (IQ greedy action or reference action distribution), fit:

- Observation, pose, elapsed time, four previous actions, and visited positions.
- The same inputs plus the Bayesian posterior.
- The same inputs plus a posterior shuffled separately inside each split.

The shuffled control has identical input dimension, width, and parameter count
to the real-posterior model. It preserves marginal posterior statistics but is
not a conditional randomization test and may produce implausible pose/belief
combinations. Hidden widths for other models are rounded to the nearest shared
parameter budget; exact counts are saved. Equal counts do not make function
classes identical. Validation selects checkpoints; fitting and model selection
never inspect test scores. The 200-epoch budget is fixed across conditions.

Posterior decoding is repeated for trained GRU memory, an untrained encoder,
and explicit posterior memory. The last control trains a decoder rather than
copying its input: failures therefore reveal limitations of the learned probe.
Random encoder and probe initialization both vary with the repetition seed.
Their individual contributions to variance are not isolated. Update tests retain
the stationary baseline and separate new-evidence from no-new-evidence steps.

Outputs in `artifacts/belief_validation/fov3_seed*/` contain reference action
probabilities and IG scores, all fitted models and training histories, held-out
predictions per fitting seed, summaries, and `validation.png` / `.pdf`.
Each paired effect has a layout-bootstrap interval. Across-seed means and ranges
are descriptive: the runs reuse test layouts, so they are not independent test
samples. Positive effects mean adding belief improves prediction, explicit/trained
memory beats random decoding, or decoded updates beat the stationary baseline,
as labeled. Report both positive controls and IQ results, including failures.

Successful positive controls would demonstrate sensitivity to this particular
known mechanism, not validate every possible belief representation. Failure to
find an IQ effect cannot establish absence of belief use: the history baseline
can already carry belief information. Causal interventions, independent held-out
layouts, and convergence checks remain necessary for stronger claims.

The initial 200-epoch, three-seed validation runs are complete for both policies.
Mean held-out cross-entropy reductions from adding belief were -0.0866 / -0.1143
nats for IQ and +0.00319 / -0.00177 nats for the explicit reference actor
(policy 0 / policy 1). The reference actor benefit was positive in all three fits
on policy-0 histories but inconsistent on policy-1 histories. Thus this behavioral
comparison does not reliably detect known belief dependence across the tested
datasets; IQ's negative effect is not evidence of no belief use.

The explicit-memory decoding control did work: mean posterior KL was
0.0463 / 0.0659 nats, versus trained-memory 0.3428 / 0.4845 and random-memory
0.3630 / 0.5014. Explicit decoded updates beat the stationary baseline by
0.3005 / 0.2971 L1 units; trained-memory updates were worse than stationary by
0.2119 / 0.2558. Many decoders selected epochs near the fixed budget, so these
are budget-dependent fits, not convergence claims. Repeated seeds share the same
held-out layouts and do not supply independent replications of the task.

### Matched-history positive control on new reset seeds

```bash
bash scripts/run_matched_validation_local.sh 0
bash scripts/run_matched_validation_local.sh 1
```

The completed runs live under `artifacts/matched_validation/fov3_seed{0,1}/`.
To rerun, provide a fresh output root as the second argument. `EPISODES`,
`RESET_SEED`, `CHECKPOINT`, and `PYTHON` can be overridden. Default reset seeds
are 500000–500299. New reset seeds do not guarantee novel layouts or states
unseen during IQ training. Both policies receive identical forced histories.

The collector retains its original left/right loop and same-belief rotation
protocol. Endpoint image, pose, and elapsed time must match. The primary subset
requires unresolved beliefs and different unique navigation IG maximizers;
selection never depends on IQ actions. The validation module independently
recomputes IG from posterior and observation geometry, verifies agreement with
the collector, and queries the explicit seven-action IG actor. Strict agreement
uses the actual all-action argmax; non-navigation choices count as failures.

For histories A/B with IG winners a/b, the directional probability effect is
`0.5 * ((pi_A[a] - pi_B[a]) + (pi_B[b] - pi_A[b]))`. Positive means policy mass
shifts toward each history's own information-seeking action. IQ retains its
checkpoint temperature (0.001); the reference uses temperature 0.25. Therefore,
effect magnitude is not a ranking of which policy is more belief-driven.
Reference agreement is guaranteed by the eligible-pair construction: it is a
measurement calibration check, not a new scientific discovery.

Each policy had 71 eligible pairs from 51 reset seeds. Reset-weighted strict
both-history agreement was 17.65% for IQ seed 0, 0% for IQ seed 1, and 100% for
the explicit control. Directional IQ effects were +0.1417 (95% reset-cluster
interval [0.0092, 0.2675]) and +0.0080 ([-0.0362, 0.0471]), respectively.
The explicit control effect was +0.0919 ([0.0892, 0.0948]).

The result supports history-dependent IG alignment in one policy under this
forced-history protocol; it is not consistent across both policies. Same-belief
rotation histories can also change IQ decisions. Movement and coverage differ
between control and primary histories, so no causal difference-of-groups claim
is justified. This experiment does not isolate a neural belief representation
or show reward-optimal belief use. Pair-level data, source hashes, summaries,
and `matched_validation.png` / `.pdf` are saved in each `comparison/` directory.

### Posterior-predictive subspace transfers

```bash
bash scripts/run_belief_transfer_local.sh 0 artifacts/belief_transfer/fov3_seed0_new
bash scripts/run_belief_transfer_local.sh 1 artifacts/belief_transfer/fov3_seed1_new
```

Completed outputs with manipulation checks are in
`artifacts/belief_transfer/fov3_seed{0,1}_with_checks/`. The first outputs without
those checks are retained separately. The checked run also avoids platform BLAS
floating-point warning artifacts; its policy effects reproduce the first run.

`analysis.transfer_belief_subspace` fits an episode-weighted ridge regression
from standardized memory to the posterior using only the original probe training
split. Linear pose/time effects are residualized from both inputs and targets.
The top eight left singular vectors of the regression coefficients define the
candidate subspace; rank 8 and ridge 0.01 were fixed before inspecting transfer
outcomes. No action labels or policy outputs are used to learn the directions.
For matched histories A/B, it applies `h_A + P(h_B - h_A)` in standardized memory
coordinates, preserving the orthogonal complement. Each pair is tested both ways.
Twenty fixed random subspaces have the same rank; projected random donor transfers
are scaled to the candidate transfer's norm. Their effects are averaged before
bootstrapping reset clusters. This differs from a full donor-memory replacement.

All initial layouts in the original probe dataset are excluded from the matched
histories. After exclusions there are 71 primary pairs / 51 reset seeds and
657 same-belief rotation pairs / 219 reset seeds per policy. These matched
histories were inspected in the preceding experiment, so this is exploratory.

The primary outcome is the change in the donor-IG-versus-source-IG action
probability margin. Positive would support the proposed directional effect.
Mean effects beyond random controls were -0.00450 (95% interval
[-0.00831, -0.00161]) for policy 0 and -0.00323 ([-0.00549, -0.00120]) for policy 1.
Transferring the explicit control's full posterior instead gives +0.18387;
this calibrates the direction but does not validate the learned IQ subspace.

The manipulation check measures raw linear decoder MSE reduction toward the
donor's actual posterior. It improved slightly for policy 0 (+0.00000675), but
not clearly for policy 1 (-0.00000251; interval includes zero). These linear
outputs are unconstrained scores, not normalized distributions. In particular,
policy 1's negative action effect cannot be treated as the effect of a successful
belief manipulation. Even for policy 0, off-manifold states or other encoded
features can explain the small opposite-direction policy change. No navigation
rescue, necessity, or latent-feature specificity has been established. These
results do not support the proposed causal belief mechanism and do not prove
that IQ never uses belief information.

Saved outputs include regression coefficients, normalization, candidate/random
bases, edited memories, policy probabilities, direction-level metrics, clustered
summaries, and `belief_transfer.png` / `.pdf`. Existing output directories are
never overwritten. `COMPARISON`, `ROLLOUTS`, and `PYTHON` override script inputs.

### Known-mechanism subspace recovery benchmark

```bash
bash scripts/run_subspace_recovery_local.sh 0
bash scripts/run_subspace_recovery_local.sh 1
```

The completed outputs are in `artifacts/subspace_recovery/fov3_seed{0,1}/`.
Use a second positional argument for a fresh output directory; `ROLLOUTS` and
`PYTHON` override inputs. This benchmark uses cached posterior distributions,
not IQ actions, and trains no navigation policy.

`analysis.validate_subspace_recovery` constructs frozen networks with a known
mechanism. Training-only posterior PCA provides either eight retained belief
coordinates or all 63 independent coordinates of the 64-cell posterior. These
coordinates are scaled using training data, combined with independent Gaussian
nuisance features, and orthogonally mixed into a 128-dimensional hidden vector.
The planted action head reverses the mixing and reads only the belief block.
A second negative-control action head reads only eight nuisance coordinates.
These are deliberately favorable linear encodings, not emergent representations.

The existing ridge/subspace estimator receives the mixed hidden states and full
posterior labels, using only the original training split. It receives neither
the planted mixing matrix nor action outputs. Three conditions were fixed:

- Eight belief coordinates, rank-eight recovery.
- Full 63-dimensional belief, rank-eight recovery (deliberate under-capacity).
- Full belief, rank-63 recovery (known-dimension control).

Evaluation selects one unresolved point from each held-out episode (54 per
source dataset), with circular donors from other test episodes. Only the
candidate hidden component is transferred. Unlike the navigation experiment,
these synthetic donor pairs need not share pose: the planted action head depends
only on the known belief block. The exact oracle replaces that block while
preserving all nuisance coordinates. Additional controls fit shuffled training
posterior labels or use 20 norm/rank-matched random donor transfers. Three mixing
seeds are reported separately; they reuse episodes and are not independent task
replications. The shuffled estimator is a separate negative control, not a
norm-matched transfer.

Policy recovery is `1 - TV(edited_policy, donor_policy) /
TV(original_policy, donor_policy)`, with numerator and denominator averaged over
held-out source episodes before division. The donor policy is the oracle target
because it reads only the donor belief. One means exact recovery, zero means no
improvement, and negative means the edit moved farther from the target. The
nuisance-only policy's TV change measures unwanted effects on irrelevant features.

Mean fractions of the oracle policy change recovered across three mixing seeds:

| Planted mechanism / fitted rank | Dataset 0 | Dataset 1 |
| --- | ---: | ---: |
| 8 belief dimensions / rank 8 | 0.9180 | 0.9049 |
| 63 belief dimensions / rank 8 | 0.0842 | 0.0253 |
| 63 belief dimensions / rank 63 | 0.9899 | 0.9927 |

Matched-rank fits recovered about 99.9% of the true subspace by normalized
projection overlap. Their nuisance-only policy TV changes were approximately
0.0006–0.0043. Random transfers recovered little of the desired policy change;
shuffled-label fits moved policies away from their targets. The exact oracle
recovered the target policy to numerical precision and preserved nuisance outputs.

This validates recovery for a known linear mechanism when rank is adequate.
It also demonstrates that a rank-eight transfer can give a weak or failed effect
even when decisions are explicitly belief-driven. It does **not** establish the
rank or encoding used by IQ, explain IQ's opposite-direction effects, or validate
nonlinear/off-manifold interventions. Select any future IQ rank using development
reconstruction/manipulation criteria, not the sign of held-out action effects.

Saved artifacts include construction and fitted parameters, source/donor indices,
edited hidden states, per-seed metrics, `summary.json`, and `recovery.png` / `.pdf`.
Unit tests check that held-out targets cannot affect network construction, oracle
transfers reproduce donor actions without changing nuisance outputs, and donors
come from different held-out episodes.

### Validation-selected IQ subspace rank

```bash
bash scripts/run_selected_rank_local.sh 0
bash scripts/run_selected_rank_local.sh 1
```

Completed selection files are under `artifacts/selected_rank/fov3_seed{0,1}/selection/`;
choose a fresh output root as the second argument to rerun. The wrapper selects
rank before collecting any new matched histories. If rank zero wins, it stops
without collection or intervention. Otherwise, the frozen selection is passed to
the transfer evaluator, which checks policy, rollout, and ridge provenance.

The candidate ranks are 0, 2, 4, 8, 16, 32, and 63, with ridge fixed at 0.01.
The existing training-only ridge fit supplies nested singular-vector subspaces.
Selection minimizes episode-weighted posterior MSE on unresolved validation
steps, with smaller ranks winning ties. Rank zero is the fitted pose/time-only
linear reconstruction baseline. No action outputs or test targets participate.
Outputs are unconstrained linear scores, so this criterion is not posterior KL.

Both IQ policies selected **rank zero**. Validation MSE for policy 0 was
0.0002630 at rank zero, 0.0003021 at rank eight, and 0.0003943 at rank 63.
Policy 1 values were 0.0002521, 0.0003255, and 0.0004573. No tested memory
subspace improved on the nuisance-only reconstruction. Therefore no new history
collection or selected-rank transfer was performed. Increasing rank blindly is
not supported by these validation results.

This is a failure to validate the current linear posterior-predictive subspace,
not evidence that IQ has no belief information. It is compatible with nonlinear
encoding, inappropriate regression/regularization, or a representation of only
task-relevant uncertainty. Earlier opposite-direction rank-eight effects should
not be presented as interventions on an established belief variable. A known
linear-belief unit test selects a nonzero rank, and another verifies that changing
test data or all action labels cannot change the selection. Figures are saved as
`rank_selection.png` / `.pdf` alongside `selection.json` and the training fit.

### Consolidated decoder comparison on fresh layouts

```bash
bash scripts/run_decoder_comparison_local.sh 0
bash scripts/run_decoder_comparison_local.sh 1
```

The comparison uses only the original training/validation splits for decoder
fitting and selection. For each of three fitting seeds, each input condition and
each target gets four candidates: linear or single-hidden-layer nonlinear, with
AdamW weight decay 0.001 or 0.1. Nonlinear candidates have approximately 20,000
parameters; linear parameter counts vary with input dimension. Counts are saved.
Validation selects architecture, regularization, and epoch. The cap is 300 epochs,
with stopping after 40 epochs without validation improvement. Budget-boundary
selection is reported rather than interpreted as convergence.

Inputs are observation/pose/time; those inputs plus four previous actions and a
visited-position mask; history plus trained memory; and history plus untrained
recurrent memory. Thus the key comparisons measure the **additional** predictive
information from trained memory, beyond common inputs. Random encoder and probe
seeds vary together, so their separate variance contributions are not isolated.
Targets are the full posterior (primary) and four fixed world-coordinate quadrant
masses (secondary). Both use normalized probability predictions and KL scores.

The protocol and exclusion list are written before fitting. Only after all choices
are frozen does the evaluator collect a fresh test set, retaining 200 distinct
layouts per policy from 300 candidate resets starting at 700000. It excludes
layouts in available prior rollout caches and reconstructs old forced-history
initial layouts for exclusion as well. Fresh means unseen in these analysis
artifacts, not necessarily unseen during IQ training. The exclusion snapshot and
held-out rollouts are saved. Tests check train-only normalization/selection,
region-probability conservation, and exclusion of duplicate/previous layouts.

For each target, the decision rule requires trained memory to beat both history
and history-plus-random controls: positive lower 95% layout-bootstrap bounds for
the seed-averaged KL differences, plus positive mean differences for every fitting
seed. A qualifying signal is labeled budget-limited if any relevant selected fit
lands in the final 10% of the epoch cap. Otherwise it is `signal_detected` or
`not_established`. The region result is secondary and does not replace a failed
primary result. Intervals are conditional on fitted models; repeated seeds do not
create independent datasets. The two policies are reported separately.

Outputs under `artifacts/decoder_comparison/` include the protocol, all candidate
validation histories, selected models, fresh rollouts, predictions, a consolidated
summary, and `decoder_comparison.png` / `.pdf`. A detected signal establishes
predictive information, not causal belief use or a validated intervention target.
Existing outputs are refused. A second positional argument selects a fresh output
directory; `ROLLOUTS`, `PYTHON`, `EPOCHS`, `TEST_EPISODES`, and `TEST_SEED` are
configurable. Changing the protocol after inspecting test results requires a new
test set rather than interpreting the old one as untouched confirmation.

The completed run retained 200 fresh layouts per policy, with 182 unresolved
layout/episodes contributing to primary scoring. Models were selected before
collecting or scoring those histories. Full-posterior test KL, averaged over the
three fitting seeds, was:

| Decoder inputs | Policy 0 | Policy 1 |
| --- | ---: | ---: |
| Observation + pose/time | 0.3591 | 0.4344 |
| Above + simple history | 0.4205 | 0.3408 |
| History + random recurrent memory | 0.3952 | 0.3683 |
| History + trained recurrent memory | 0.3685 | 0.3935 |

Lower is better. Policy 0's mean trained-memory improvement over history was
+0.0520 nats (95% layout interval [0.0247, 0.0889]), but one fitting seed had a
negative effect. Its improvement over random memory was +0.0267 with an interval
including zero ([-0.0198, 0.0631]). Also, its simple observation decoder outperformed
its history baseline: adding features does not guarantee improved generalization
for these finite fitted models. Two selected full-posterior memory fits were near
the epoch cap. Policy 1's corresponding improvements were -0.0527
([-0.0749, -0.0310]) and -0.0252 ([-0.0489, -0.0059]). Neither policy passed the
predeclared primary rule; neither passed the secondary quadrant rule either.

This is not a validated intervention target. It leaves open redundant belief
information already present in the controls, nonlinear encoding not recovered by
these decoders, and fitting/generalization limitations. It does not demonstrate
absence of belief use. No additional intervention was launched or hyperparameters
retuned against these test results.
