"""Per-frame features and soft H/A/D evidence.

Factorisation per frame:  P(D) = p_dead,  P(H) = (1 - p_dead) * p_home,  P(A) = (1 - p_dead) * (1 - p_home).
p_home is 0.5 when there is no information (ball missing, contested, in flight) so the temporal
model keeps the previous team -- the "loose ball stays with the last possessor" rule.
Missing ball alone never raises p_dead.
"""
import warnings

import numpy as np

from .loaders import AWAY, HOME

STATES = ("H", "A", "D")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _rolling_nanmean(values, half_window):
    """Centred mean over finite values only; NaN where the window has none."""
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    csum = np.concatenate(([0.0], np.cumsum(filled)))
    ccount = np.concatenate(([0], np.cumsum(finite)))
    idx = np.arange(len(values))
    lo = np.maximum(0, idx - half_window)
    hi = np.minimum(len(values), idx + half_window + 1)
    count = ccount[hi] - ccount[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (csum[hi] - csum[lo]) / count
    out[count == 0] = np.nan
    return out


CHUNK = 16384  # frames per block: keeps float64 temporaries to a few tens of MB


def features(tracking, pitch, fps, cfg):
    length, width = pitch
    scale = np.array([length, width], dtype=np.float64)
    ball = tracking.ball_xy.astype(np.float64) * scale            # metres, (N, 2): small
    ball_ok = np.isfinite(ball).all(axis=1)
    n = len(ball)
    home_slots = tracking.team_of_slot == HOME
    away_slots = tracking.team_of_slot == AWAY

    out = {"ball_ok": ball_ok, "d_home": np.empty(n), "d_away": np.empty(n),
           "player_speed_median": np.empty(n), "n_players": np.empty(n, dtype=np.int64)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows -> NaN, as intended
        for lo in range(0, n, CHUNK):
            hi = min(n, lo + CHUNK)
            players = tracking.player_xy[lo:hi].astype(np.float64) * scale     # (c, S, 2)
            dist = np.linalg.norm(players - ball[lo:hi, None, :], axis=2)      # NaN if missing
            out["d_home"][lo:hi] = np.nanmin(dist[:, home_slots], axis=1)
            out["d_away"][lo:hi] = np.nanmin(dist[:, away_slots], axis=1)
            out["player_speed_median"][lo:hi] = np.nanmedian(
                tracking.player_speed[lo:hi].astype(np.float64), axis=1)
            out["n_players"][lo:hi] = np.isfinite(tracking.player_xy[lo:hi]).all(axis=2).sum(axis=1)

    # Ball speed only across consecutive frames that both observe the ball.
    step = np.full(len(ball), np.nan)
    both = ball_ok[1:] & ball_ok[:-1]
    step[1:][both] = np.linalg.norm(ball[1:][both] - ball[:-1][both], axis=1) * fps
    out["ball_speed"] = _rolling_nanmean(step, cfg.speed_half_window)

    margin_x = cfg.out_of_bounds_margin_m / length
    margin_y = cfg.out_of_bounds_margin_m / width
    bx, by = tracking.ball_xy[:, 0], tracking.ball_xy[:, 1]
    with np.errstate(invalid="ignore"):
        oob = (bx < -margin_x) | (bx > 1 + margin_x) | (by < -margin_y) | (by > 1 + margin_y)
    out["out_of_bounds"] = oob & ball_ok
    return out


def tag_evidence(events, n_frames, cfg):
    """(team_bias, set_piece_mask). team_bias > 0 favours home, < 0 away."""
    team_bias = np.zeros(n_frames)
    set_piece = np.zeros(n_frames, dtype=bool)
    for ev in events:
        lo, hi = max(0, ev["start"]), min(n_frames, ev["end"])
        if hi <= lo:
            continue
        is_set_piece = ev["code"] == "set pieces"
        if is_set_piece:
            set_piece[lo:hi] = True
        if ev["side"] is None or ev["code"] == "goals":
            continue
        gain = cfg.tag_set_piece_gain if is_set_piece else cfg.tag_phase_gain
        team_bias[lo:hi] += gain if ev["side"] == HOME else -gain
    return team_bias, set_piece


def evidence(feat, team_bias, set_piece, cfg):
    """(p_dead, p_home, controls) per frame. controls = (c_home, c_away) for diagnostics."""
    speed = np.nan_to_num(feat["ball_speed"], nan=0.0)
    carried = sigmoid((cfg.carry_speed_mps - speed) / cfg.carry_speed_softness_mps)

    def near(d):
        return np.where(np.isfinite(d),
                        sigmoid((cfg.control_radius_m - np.nan_to_num(d, nan=1e3))
                                / cfg.control_softness_m), 0.0)

    near_h, near_a = near(feat["d_home"]), near(feat["d_away"])
    ok = feat["ball_ok"]
    c_home = np.where(ok, near_h * (1 - near_a) * carried, 0.0)   # contested -> low control
    c_away = np.where(ok, near_a * (1 - near_h) * carried, 0.0)

    p_home = sigmoid(cfg.control_gain * (c_home - c_away) + team_bias)

    psm = feat["player_speed_median"]
    slow = np.where(np.isfinite(psm),
                    sigmoid((cfg.slow_players_mps - np.nan_to_num(psm)) / cfg.slow_players_softness_mps),
                    0.0)
    still = np.where(ok & np.isfinite(feat["ball_speed"]),
                     (np.nan_to_num(feat["ball_speed"], nan=1e3) < cfg.still_ball_mps).astype(float),
                     0.0)
    logit = (cfg.dead_base_logit
             + cfg.dead_out_of_bounds * feat["out_of_bounds"]
             + cfg.dead_slow_players * slow
             + cfg.dead_still_ball * still
             + cfg.dead_set_piece_tag * set_piece
             + cfg.dead_few_players * (feat["n_players"] < cfg.min_players_for_play)
             - cfg.alive_control * np.maximum(c_home, c_away) * (1 - slow))
    p_dead = sigmoid(logit)
    return p_dead, p_home, (c_home, c_away)


def log_emissions(p_dead, p_home, cfg, eps=1e-6):
    """(N, 3) tempered log-likelihoods for states H, A, D."""
    alive = 1.0 - p_dead
    probs = np.stack([alive * p_home, alive * (1 - p_home), p_dead], axis=1)
    return cfg.evidence_temperature * np.log(np.clip(probs, eps, 1.0))
