"""Tie-aware action ordering metrics; undefined correlations stay missing."""
import numpy as np


def action_alignment(q, scores, ig_action, ig_maximizers, actions=None):
    """Separate actual-policy agreement from navigation-only diagnostics.

    The conditional comparison restricts the action space, not the sampled
    timesteps. Non-navigation actual actions always disagree with the baseline.
    """
    navigation = tuple(scores) if actions is None else tuple(actions)
    actual = int(np.argmax(q))
    conditional = max(navigation, key=lambda a: q[a])
    outside = actual not in navigation
    result = dict(
        executed_action=actual,
        ig_tie=len(ig_maximizers) > 1, ig_all_tied=len(ig_maximizers) == len(navigation),
        iq_argmax=actual, iq_argmax_all=actual,
        iq_argmax_navigation=conditional,
        actual_outside_navigation=outside,
        strict_agreement=not outside and actual == ig_action,
        navigation_conditional_agreement=conditional == ig_action,
        strict_q_in_ig_maximizers=not outside and actual in ig_maximizers,
        navigation_conditional_q_in_ig_maximizers=conditional in ig_maximizers,
        navigation_conditional_spearman=rank_correlation(
            [q[a] for a in navigation], [scores[a] for a in navigation]),
        q_tie=bool(np.sum(np.abs(np.asarray(q) - q[actual]) <= 1e-10) > 1),
        navigation_q_tie=sum(abs(q[a] - q[conditional]) <= 1e-10 for a in navigation) > 1,
    )
    # Keep the remote report names as aliases of the explicit schema-v2 names.
    result.update(
        policy_argmax_outside_navigation=result['actual_outside_navigation'],
        agreement=result['strict_agreement'],
        agreement_navigation_conditional=result['navigation_conditional_agreement'],
        q_in_ig_maximizers=result['strict_q_in_ig_maximizers'],
        q_in_ig_maximizers_navigation_conditional=result['navigation_conditional_q_in_ig_maximizers'],
        spearman_navigation=result['navigation_conditional_spearman'],
    )
    return result


def rank_correlation(q, ig):
    def ranks(v):
        v = np.asarray(v)
        return np.array([np.sum(v < x) + (np.sum(v == x) - 1) / 2 for x in v])
    x, y = ranks(q), ranks(ig)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def summarize(rows):
    def group(items):
        unique = [r for r in items if not r['ig_tie']]
        informative = [r for r in items if not r['ig_all_tied']]
        ranks = [r['navigation_conditional_spearman'] for r in items
                 if r['navigation_conditional_spearman'] is not None]
        def mean(key, subset):
            return float(np.mean([r[key] for r in subset])) if subset else None
        result = dict(steps=len(items), unique_ig_steps=len(unique),
                    strict_agreement_unique_ig=mean('strict_agreement', unique),
                    actual_outside_navigation_steps=sum(r['actual_outside_navigation'] for r in items),
                    actual_outside_navigation_fraction=mean('actual_outside_navigation', items),
                    navigation_conditional_agreement_unique_ig=mean('navigation_conditional_agreement', unique),
                    informative_steps=len(informative),
                    strict_q_in_ig_maximizers=mean('strict_q_in_ig_maximizers', informative),
                    navigation_conditional_q_in_ig_maximizers=mean('navigation_conditional_q_in_ig_maximizers', informative),
                    navigation_conditional_rank_steps=len(ranks),
                    navigation_conditional_mean_spearman=float(np.mean(ranks)) if ranks else None)
        result.update(
            agreement=mean('strict_agreement', items),
            agreement_navigation_conditional=mean('navigation_conditional_agreement', items),
            policy_argmax_outside_navigation_fraction=result['actual_outside_navigation_fraction'],
            agreement_unique_ig=result['strict_agreement_unique_ig'],
            agreement_unique_ig_navigation_conditional=result['navigation_conditional_agreement_unique_ig'],
            q_in_ig_maximizers=result['strict_q_in_ig_maximizers'],
            q_in_ig_maximizers_navigation_conditional=result['navigation_conditional_q_in_ig_maximizers'],
            rank_steps_navigation=result['navigation_conditional_rank_steps'],
            mean_spearman_navigation=result['navigation_conditional_mean_spearman'],
        )
        return result
    return dict(overall=group(rows),
                unresolved=group([r for r in rows if not r['resolved']]),
                resolved=group([r for r in rows if r['resolved']]),
                by_candidate_count={str(n): group([r for r in rows if r['candidates'] == n])
                                    for n in sorted({r['candidates'] for r in rows})})


def reward_action_alignment(values, executed_action, tolerance=1e-10):
    """Compare the actual seven-action policy choice with reference reward Q."""
    values = np.asarray(values, dtype=float)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError('Need seven finite reference reward Q-values.')
    if executed_action not in range(7):
        raise ValueError('Executed action must be one of the seven MiniGrid actions.')
    best = values.max()
    winners = tuple(int(a) for a in np.flatnonzero(np.abs(values - best) <= tolerance))
    return dict(argmax=winners[0], maximizers=winners,
                tie=len(winners) > 1, all_tied=len(winners) == 7,
                strict_agreement_unique_reward=(executed_action == winners[0]
                                                if len(winners) == 1 else None),
                q_in_reward_maximizers=executed_action in winners,
                action_regret=float(best - values[executed_action]))


def summarize_reward_alignment(rows, horizons):
    """Step-weighted reward comparisons, with unique/tied-state denominators."""
    def group(items, prefix):
        unique = [r for r in items if not r[prefix + 'tie']]
        informative = [r for r in items if not r[prefix + 'all_tied']]

        def mean(key, subset):
            return float(np.mean([r[prefix + key] for r in subset])) if subset else None

        return dict(steps=len(items), episodes=len({r['episode'] for r in items}),
                    unique_reward_steps=len(unique),
                    unique_reward_episodes=len({r['episode'] for r in unique}),
                    reward_tie_steps=len(items) - len(unique),
                    all_reward_tied_steps=len(items) - len(informative),
                    informative_reward_steps=len(informative),
                    informative_reward_episodes=len({r['episode'] for r in informative}),
                    strict_agreement_unique_reward=mean('strict_agreement_unique_reward', unique),
                    q_in_reward_maximizers=mean('q_in_reward_maximizers', informative),
                    mean_action_regret=mean('action_regret', items),
                    mean_action_regret_informative=mean('action_regret', informative))

    result = {}
    for horizon in horizons:
        prefix = f'reward_h{horizon}_'
        result[str(horizon)] = dict(
            overall=group(rows, prefix),
            unresolved=group([r for r in rows if not r['resolved']], prefix),
            resolved=group([r for r in rows if r['resolved']], prefix))
    return result
