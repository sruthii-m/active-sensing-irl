"""Tie-aware action ordering metrics; undefined correlations stay missing."""
import numpy as np

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
        ranks = [r['spearman'] for r in items if r['spearman'] is not None]
        return dict(steps=len(items), unique_ig_steps=len(unique),
                    agreement_unique_ig=np.mean([r['agreement'] for r in unique]).item() if unique else None,
                    informative_steps=len(informative),
                    q_in_ig_maximizers=np.mean([r['q_in_ig_maximizers'] for r in informative]).item() if informative else None,
                    rank_steps=len(ranks), mean_spearman=float(np.mean(ranks)) if ranks else None)
    return dict(unresolved=group([r for r in rows if not r['resolved']]),
                resolved=group([r for r in rows if r['resolved']]),
                by_candidate_count={str(n): group([r for r in rows if r['candidates'] == n])
                                    for n in sorted({r['candidates'] for r in rows})})
