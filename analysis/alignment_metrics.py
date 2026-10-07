"""Tie-aware action ordering metrics; undefined correlations stay missing."""
import numpy as np


def action_alignment(q, scores, ig_action, ties, actions):
    """Compare the executed all-action argmax and a navigation-only diagnostic."""
    reward_action = int(np.argmax(q))
    candidate_action = max(actions, key=lambda a: q[a])
    outside_navigation = reward_action not in actions
    return dict(
        iq_argmax=reward_action, iq_argmax_all=reward_action,
        iq_argmax_navigation=candidate_action, executed_action=reward_action,
        policy_argmax_outside_navigation=outside_navigation,
        agreement=not outside_navigation and reward_action == ig_action,
        agreement_navigation_conditional=candidate_action == ig_action,
        ig_tie=len(ties) > 1, ig_all_tied=len(ties) == len(actions),
        q_tie=sum(abs(q[a] - q[candidate_action]) <= 1e-10 for a in actions) > 1,
        q_in_ig_maximizers=not outside_navigation and reward_action in ties,
        q_in_ig_maximizers_navigation_conditional=candidate_action in ties,
        spearman_navigation=rank_correlation(
            [q[a] for a in actions], [scores[a] for a in actions]),
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
        ranks = [r['spearman_navigation'] for r in items if r['spearman_navigation'] is not None]
        def mean(key, subset):
            return float(np.mean([r[key] for r in subset])) if subset else None
        return dict(steps=len(items), unique_ig_steps=len(unique),
                    agreement=mean('agreement', items),
                    policy_argmax_outside_navigation_fraction=mean('policy_argmax_outside_navigation', items),
                    agreement_navigation_conditional=mean('agreement_navigation_conditional', items),
                    agreement_unique_ig=mean('agreement', unique),
                    agreement_unique_ig_navigation_conditional=mean('agreement_navigation_conditional', unique),
                    informative_steps=len(informative),
                    q_in_ig_maximizers=mean('q_in_ig_maximizers', informative),
                    q_in_ig_maximizers_navigation_conditional=mean('q_in_ig_maximizers_navigation_conditional', informative),
                    rank_steps_navigation=len(ranks),
                    mean_spearman_navigation=float(np.mean(ranks)) if ranks else None)
    return dict(overall=group(rows),
                unresolved=group([r for r in rows if not r['resolved']]),
                resolved=group([r for r in rows if r['resolved']]),
                by_candidate_count={str(n): group([r for r in rows if r['candidates'] == n])
                                    for n in sorted({r['candidates'] for r in rows})})
