"""Three-state HMM (H, A, D) decoded with Viterbi, independently per half.

Sticky transitions encode that possession is an interval: a switch is accepted only when the
accumulated (tempered) evidence outweighs the switch cost. Decoding is offline, so evidence after
a hand-over helps place its boundary. Ties resolve to the lowest state index, deterministically.
"""
import math

import numpy as np


def transition_log_matrix(cfg, fps):
    leave_team = 1.0 / (cfg.mean_team_state_s * fps)
    turnover = leave_team * cfg.turnover_fraction
    stop = leave_team * (1 - cfg.turnover_fraction)
    leave_dead = 1.0 / (cfg.mean_dead_s * fps)
    m = np.array([
        [1 - turnover - stop, turnover, stop],             # from H
        [turnover, 1 - turnover - stop, stop],             # from A
        [leave_dead / 2, leave_dead / 2, 1 - leave_dead],  # from D
    ])
    return np.log(m)


def viterbi(log_emit, log_trans, log_init=None):
    """Most likely state path for an (T, K) emission matrix."""
    n, k = log_emit.shape
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    if log_init is None:
        log_init = np.full(k, -math.log(k))
    back = np.zeros((n, k), dtype=np.int8)
    score = log_init + log_emit[0]
    trans_t = log_trans.T  # [to, from]
    for t in range(1, n):
        cand = score[None, :] + trans_t            # [to, from]
        best = np.argmax(cand, axis=1)
        back[t] = best
        score = cand[np.arange(k), best] + log_emit[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_halves(log_emit, blocks, cfg, fps, n_frames):
    """State codes 0/1/2 inside each half, -1 elsewhere. State is reset at every half."""
    codes = np.full(n_frames, -1, dtype=np.int8)
    log_trans = transition_log_matrix(cfg, fps)
    for _half, first, last in blocks:
        codes[first:last + 1] = viterbi(log_emit[first:last + 1], log_trans)
    return codes
