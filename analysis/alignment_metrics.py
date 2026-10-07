"""Tie-aware action ordering metrics; undefined correlations stay missing."""
import numpy as np


def action_alignment(q, scores, ig_action, ig_maximizers):
    """Separate actual-policy agreement from navigation-only diagnostics.

    The conditional comparison restricts the action space, not the sampled
    timesteps. Non-navigation actual actions always disagree with the baseline.
    """
    navigation = tuple(scores)
    actual = int(np.argmax(q))
    conditional = max(navigation, key=lambda a: q[a])
    outside = actual not in navigation
    return dict(
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
        return dict(steps=len(items), unique_ig_steps=len(unique),
                    strict_agreement_unique_ig=mean('strict_agreement', unique),
                    actual_outside_navigation_steps=sum(r['actual_outside_navigation'] for r in items),
                    actual_outside_navigation_fraction=mean('actual_outside_navigation', items),
                    navigation_conditional_agreement_unique_ig=mean('navigation_conditional_agreement', unique),
                    informative_steps=len(informative),
                    strict_q_in_ig_maximizers=mean('strict_q_in_ig_maximizers', informative),
                    navigation_conditional_q_in_ig_maximizers=mean('navigation_conditional_q_in_ig_maximizers', informative),
                    navigation_conditional_rank_steps=len(ranks),
                    navigation_conditional_mean_spearman=float(np.mean(ranks)) if ranks else None)
    return dict(overall=group(rows),
                unresolved=group([r for r in rows if not r['resolved']]),
                resolved=group([r for r in rows if r['resolved']]),
                by_candidate_count={str(n): group([r for r in rows if r['candidates'] == n])
                                    for n in sorted({r['candidates'] for r in rows})})
