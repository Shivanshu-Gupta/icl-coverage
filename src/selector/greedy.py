import numpy as np

def set_broadcasted_max(a: list[dict[int, float]], b: dict[int, float]):
    res = [dict() for _ in a]
    for i, a_i in enumerate(a):
        for k in a_i:
            res[i][k] = max(a_i[k], b[k]) if k in b else a_i[k]
        for k in b:
            if k not in a_i:
                res[i][k] = b[k]
    return res

def decomposed_coverage_greedy(
    k, cand_partial_scores, add_cand_score=False, cand_score_discount=1, candidates=None,
    cand_lens=None, max_len=-1,
):
    # partial_scores: cd
    state = dict(
        comb=[],
        set_partial_scores=-np.inf,    # d
        set_score=-np.inf,
    )
    stats = dict(n_reset=0)
    rem_len = max_len if max_len > 0 else 1000000
    while len(state['comb']) < k:
        if isinstance(cand_partial_scores[0], dict):
            if state['set_score'] == -np.inf:
                candset_partial_scores = cand_partial_scores
            else:
                candset_partial_scores = set_broadcasted_max(cand_partial_scores, state['set_partial_scores'])  # cd, d -> cd
            candset_scores = np.array([sum(s.values()) for s in candset_partial_scores]) # cd -> c
        else:
            candset_partial_scores = np.maximum(cand_partial_scores, state['set_partial_scores'])  # cd, d -> cd
            candset_scores = candset_partial_scores.sum(axis=-1) # cd -> c
            if add_cand_score:
                candset_scores += cand_partial_scores.sum(axis=-1) / cand_score_discount
        candset_scores[state['comb']] = -np.inf
        best_idx = np.argmax(candset_scores)
        if candset_scores[best_idx] > state['set_score']:
            if cand_lens:
                rem_len -= cand_lens[best_idx]
                if rem_len < 0: break
            state = dict(
                comb=state['comb'] + [best_idx],
                set_partial_scores=candset_partial_scores[best_idx],
                set_score=candset_scores[best_idx],
            )
        else:
            stats['n_reset'] += 1
            state['set_partial_scores'] = -np.inf
            state['set_score'] = -np.inf
    state['comb'].reverse()
    return state['comb'], stats | dict(score=state['set_score'])