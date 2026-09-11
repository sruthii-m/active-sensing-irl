# Vendored from Div99/IQ-Learn

Source: https://github.com/Div99/IQ-Learn, commit `1f5492dd26348ef11dea3467037baf4eff65c178`.

Only the algorithm core is vendored here, not the full hydra/wandb/tensorboardX
training harness (`train_iq.py`, `conf/`, `make_envs.py`, `utils/logger.py`) —
that harness targets Atari/MuJoCo/classic-control envs and old-gym's 4-tuple
`step()`/`env.seed()` API. This project's training loop and agent wrapper
(`imitation/iq_learn_agent.py`, `imitation/train_iq_full_state.py`) reimplement
`agent/softq.py`'s update logic directly against gymnasium's 5-tuple API and a
plain dict-based config instead of Hydra, but call the unmodified files below.

Files kept as-is (see their own headers for Div Garg's original copyright):
- `agent/softq_models.py` — `SimpleQNetwork`/`SoftQNetwork` architectures.
- `iq.py` — the `iq_loss` function (the actual IQ-Learn objective).
- `dataset/expert_dataset.py` — `ExpertDataset` / `load_trajectories`, defining
  the `{states, next_states, actions, rewards, dones, lengths}` demo format
  that `imitation/generate_full_state_demos.py` writes to.
