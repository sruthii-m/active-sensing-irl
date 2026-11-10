"""NumPy reference definitions for a finite-horizon IRL comparison.

These functions specify numerical contracts and provide analytical test targets.
They are NOT PyTorch/autograd implementations or complete IQ/AIRL trainers.
Temperature is explicit in soft planning/IQ; AIRL below uses its standard unit
reward-temperature convention. Export AIRL's unshaped base reward g, not its
discriminator logits. A production IQ target-value calculation must stop gradients.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class SoftSolution:
    q: np.ndarray       # [H,S,A], invalid actions are -inf
    v: np.ndarray       # [H+1,S], final/terminal values are exactly zero
    policy: np.ndarray  # [H,S,A], terminal rows and invalid actions are zero


def _discount(gamma):
    if not np.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must be finite and in [0,1]")


def _temperature(temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")


def soft_dp(transition, reward, allowed, *, gamma=1.0, temperature=0.1):
    """Exact backward soft planning for a finite tabular decision process.

    transition: P[s,a,s_next], known, independent of time.
    reward: expected immediate reward r[t,s,a]. No sampled hidden state is needed.
    allowed: boolean [H,S,A]. A state with no valid actions has V=0 and policy=0.
    Every (s,a) allowed at any time must have a normalized transition row.
    Future terminal reward/entropy is zero. gamma=1 is valid at finite horizon.
    """
    _discount(gamma); _temperature(temperature)
    p = np.asarray(transition, dtype=float)
    r = np.asarray(reward, dtype=float)
    mask = np.asarray(allowed, dtype=bool)
    if r.ndim != 3 or r.shape != mask.shape or r.shape[0] < 1:
        raise ValueError("reward and allowed must have the same nonempty [H,S,A] shape")
    h, s, a = r.shape
    if p.shape != (s, a, s) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("transition must be finite nonnegative [S,A,S]")
    if not np.isfinite(r).all():
        raise ValueError("reward must be finite")
    used = mask.any(axis=0)
    if not np.allclose(p.sum(axis=-1)[used], 1.0, atol=1e-12):
        raise ValueError("transition rows for allowed actions must sum to one")
    q = np.full_like(r, -np.inf)
    v = np.zeros((h + 1, s))
    policy = np.zeros_like(r)
    for t in range(h - 1, -1, -1):
        candidate = r[t] + gamma * np.einsum('san,n->sa', p, v[t + 1])
        q[t] = np.where(mask[t], candidate, -np.inf)
        active = mask[t].any(axis=-1)
        # Only evaluate active rows; all-invalid logsumexp/softmax is undefined.
        logits = q[t, active] / temperature
        norm = logsumexp(logits, axis=-1)
        v[t, active] = temperature * norm
        policy[t, active] = np.exp(logits - norm[:, None])
    return SoftSolution(q, v, policy)


def episode_mean_nll(selected_log_probs, episode_ids):
    """Average -log p(action) within each episode, then equally across episodes."""
    logp = np.asarray(selected_log_probs, dtype=float)
    ids = np.asarray(episode_ids)
    if logp.ndim != 1 or ids.shape != logp.shape or logp.size == 0:
        raise ValueError("nonempty 1D log probabilities and matching episode IDs required")
    if not np.isfinite(logp).all() or (logp > 1e-12).any():
        raise ValueError("selected log probabilities must be finite and <=0")
    return float(np.mean([-logp[ids == i].mean() for i in np.unique(ids)]))


def expected_implicit_reward(q, transition, next_value, *, gamma=1.0):
    """Q[s,a] - gamma E[V_next|s,a], with terminal next values already set to 0.

    This is model-conditioned reward extraction. Replacing the transition kernel
    when exporting changes the reward; freeze/document that kernel for transfer.
    """
    _discount(gamma)
    q = np.asarray(q, dtype=float)
    p = np.asarray(transition, dtype=float)
    v = np.asarray(next_value, dtype=float)
    if q.ndim != 2 or p.shape != (*q.shape, v.size) or v.ndim != 1:
        raise ValueError("expected Q[S,A], P[S,A,S_next], V_next[S_next]")
    if not (np.isfinite(q).all() and np.isfinite(p).all() and np.isfinite(v).all()):
        raise ValueError("inputs must be finite; select valid Q entries first")
    return q - gamma * np.einsum('san,n->sa', p, v)


def sampled_iq_loss(q_values, expert_actions, target_next_value, terminated,
                    *, gamma=0.99, temperature=0.001, chi_lambda=0.5):
    """Project's all-expert value_expert + chi objective, averaged over steps.

    target_next_value must be computed without gradients in a production trainer.
    'terminated' means the experiment's intrinsic terminal condition, not an
    arbitrary data collection cutoff. Arrays are length N; q_values has [N,A].
    No environmental reward is an input to this objective.
    """
    _discount(gamma); _temperature(temperature)
    if not np.isfinite(chi_lambda) or chi_lambda <= 0:
        raise ValueError("chi_lambda must be positive")
    q = np.asarray(q_values, dtype=float)
    actions = np.asarray(expert_actions)
    nxt = np.asarray(target_next_value, dtype=float)
    done = np.asarray(terminated, dtype=bool)
    if q.ndim != 2 or not q.shape[0] or actions.shape != (len(q),) or nxt.shape != actions.shape or done.shape != actions.shape:
        raise ValueError("expected Q[N,A] and matching [N] arrays")
    if not np.issubdtype(actions.dtype, np.integer) or (actions < 0).any() or (actions >= q.shape[1]).any():
        raise ValueError("expert action indexes out of range")
    if not np.isfinite(q).all() or not np.isfinite(nxt).all():
        raise ValueError("Q and next values must be finite")
    v = temperature * logsumexp(q / temperature, axis=-1)
    qa = q[np.arange(len(q)), actions]
    y = gamma * (~done) * nxt
    residual = qa - y
    imitation = float(np.mean(v - qa))
    penalty = float(np.mean(residual ** 2) / (4 * chi_lambda))
    return {'loss': imitation + penalty, 'temperature_times_nll': imitation,
            'chi_penalty': penalty, 'sampled_residual': residual}


def airl_logits(base_reward, potential, next_potential, terminated, log_policy,
                *, gamma=0.99):
    """Standard unit-temperature AIRL logit f - log pi for transitions.

    f=g + gamma*(1-terminated)*Phi_next - Phi. g is the unshaped reward to
    export. These logits are not a portable reward. No recurrent history is
    reconstructed by this scalar kernel; callers must compute correct log pi.
    """
    _discount(gamma)
    g, phi, nxt, logpi = [np.asarray(x, dtype=float) for x in
                          (base_reward, potential, next_potential, log_policy)]
    done = np.asarray(terminated, dtype=bool)
    if not all(x.shape == g.shape for x in (phi, nxt, logpi, done)):
        raise ValueError("AIRL transition arrays must have equal shapes")
    if not all(np.isfinite(x).all() for x in (g, phi, nxt, logpi)) or (logpi > 1e-12).any():
        raise ValueError("finite inputs and valid log policy probabilities required")
    return g + gamma * (~done) * nxt - phi - logpi


def balanced_airl_loss(expert_logits, generator_logits):
    """Equal mass to expert-positive and generator-negative logistic losses."""
    e = np.asarray(expert_logits, dtype=float)
    g = np.asarray(generator_logits, dtype=float)
    if not e.size or not g.size or not np.isfinite(e).all() or not np.isfinite(g).all():
        raise ValueError("nonempty finite logits required")
    return float(0.5 * np.logaddexp(0, -e).mean() + 0.5 * np.logaddexp(0, g).mean())


def cue_model(cost, reliability=0.8, prior=0.5):
    """Two-decision calibration POMDP as an exact belief MDP.

    State IDs: 0=initial belief; 1=left cue; 2=right cue; 3=terminal.
    Actions: 0=commit left; 1=commit right; 2=sense (initial step only).
    Correct terminal choice pays 1; incorrect choice pays 0; sensing costs cost.
    Known context (prior,reliability,availability) is encoded in this model;
    state IDs are only valid within that fixed model, not across contexts.
    """
    if not np.isfinite(cost) or cost < 0 or not 0 < prior < 1 or not 0.5 < reliability < 1:
        raise ValueError("cost>=0, prior in (0,1), reliability in (0.5,1) required")
    py_left = prior * reliability + (1 - prior) * (1 - reliability)
    post_left = prior * reliability / py_left
    post_right = prior * (1 - reliability) / (1 - py_left)
    p = np.zeros((4, 3, 4)); p[:, :2, 3] = 1
    p[0, 2, 1:3] = [py_left, 1 - py_left]
    reward = np.zeros((2, 4, 3))
    for state, belief in enumerate([prior, post_left, post_right]):
        reward[:, state, :2] = [belief, 1 - belief]
    reward[:, 0, 2] = -cost
    allowed = np.zeros_like(reward, dtype=bool)
    allowed[0, 0] = True
    allowed[1, 1:3, :2] = True
    return p, reward, allowed
