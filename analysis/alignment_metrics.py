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


def information_value_alignment(q, informed, withheld, immediate, tolerance=1e-10):
    """Visual evidence value conditional on a forced first action, over all seven actions."""
    q, informed, withheld, immediate = map(lambda v: np.asarray(v, dtype=float),
                                          (q, informed, withheld, immediate))
    if any(v.shape != (7,) or not np.isfinite(v).all() for v in (q, informed, withheld, immediate)):
        raise ValueError('Need seven finite action values for each comparator.')
    value = informed - withheld
    if value.min() < -tolerance:
        raise ValueError('Access to visual evidence reduced optimal expected reward.')
    value = np.maximum(value, 0.)
    varying = np.ptp(value) > tolerance
    winners = np.flatnonzero(np.abs(value - value.max()) <= tolerance)
    # Conditional association: remove linear dependence on reward obtainable
    # without vision and one-step reward. Seven actions only; diagnostic, not causal.
    design = np.column_stack([np.ones(7), withheld, immediate])
    residuals = [v - design @ np.linalg.lstsq(design, v, rcond=None)[0] for v in (q, value)]
    partial = (float(np.corrcoef(*residuals)[0, 1])
               if all(np.std(v) > tolerance for v in residuals) else None)
    result = dict(max_value=float(value.max()), chosen_value=float(value[q.argmax()]),
                  useful=bool(value.max() > tolerance), action_varying=bool(varying),
                  unique_maximizer=len(winners) == 1,
                  strict_agreement=bool(q.argmax() in winners) if len(winners) == 1 else None,
                  q_spearman=rank_correlation(q, value) if varying else None,
                  controlled_q_correlation=partial if varying else None)
    result.update({f'value_{a}': float(v) for a, v in enumerate(value)})
    result.update({f'withheld_q_{a}': float(v) for a, v in enumerate(withheld)})
    return result


def summarize_information_value(rows, horizons):
    """Fixed uncertainty bins and episode-bootstrap intervals for descriptive associations."""
    bins = [(2,3),(4,7),(8,15),(16,31),(32,63),(64,float('inf'))]

    def estimate(items, key):
        eligible = [r for r in items if r[key] is not None]
        if not eligible:
            return dict(mean=None, ci95=None, steps=0, episodes=0)
        episodes = sorted({r['episode'] for r in eligible})
        sums = np.array([sum(float(r[key]) for r in eligible if r['episode'] == e) for e in episodes])
        counts = np.array([sum(r['episode'] == e for r in eligible) for e in episodes])
        draws = np.random.default_rng(0).integers(len(episodes), size=(2000, len(episodes)))
        boot = sums[draws].sum(1) / counts[draws].sum(1)
        return dict(mean=float(sums.sum()/counts.sum()),
                    ci95=np.quantile(boot,[.025,.975]).tolist() if len(episodes)>1 else None,
                    steps=len(eligible), episodes=len(episodes))

    def group(items, prefix):
        unique_ig = [r for r in items if not r['resolved'] and not r['ig_tie']]
        # Chance reference preserves actual/IG action frequencies within uncertainty bins.
        expected = 0.
        for low, high in bins:
            subset = [r for r in unique_ig if low <= r['candidates'] <= high]
            if subset:
                expected += sum(sum(r['executed_action']==a for r in subset) *
                                sum(r['ig_argmax']==a for r in subset) for a in range(7)) / len(subset)
        baseline = expected/len(unique_ig) if unique_ig else None
        agreement = estimate(unique_ig, 'strict_agreement')
        return dict(steps=len(items), episodes=len({r['episode'] for r in items}),
                    strict_ig_agreement=agreement,
                    uncertainty_binned_action_frequency_reference=baseline,
                    excess_ig_agreement=agreement['mean']-baseline if baseline is not None else None,
                    max_information_value=estimate(items, prefix+'max_value'),
                    strict_value_agreement=estimate(items, prefix+'strict_agreement'),
                    q_value_spearman=estimate(items, prefix+'q_spearman'),
                    controlled_q_value_correlation=estimate(items, prefix+'controlled_q_correlation'))

    result = {}
    for horizon in horizons:
        prefix = f'information_h{horizon}_'
        unresolved = [r for r in rows if not r['resolved']]
        def split(items):
            return {label: group([r for r in items if r[prefix+'useful'] == useful], prefix)
                    for label, useful in [('positive_visual_value', True), ('zero_visual_value', False)]}
        result[str(horizon)] = dict(
            unresolved=split(unresolved), resolved=group([r for r in rows if r['resolved']], prefix),
            by_candidate_count={f'{low}-{high if np.isfinite(high) else "plus"}': split(
                [r for r in unresolved if low <= r['candidates'] <= high]) for low, high in bins})
    return result
