"""Evaluate frozen possession variants on external IDSSE labels."""

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from possession import decode, evidence, upstream
from possession.config import DEFAULT
from evaluation import possession_metrics as official
from starter.baseline import smooth
from train_dfl_live_ball import BALL_COLUMNS, DEV, TEST, TRAIN, features
from train_live_ball_pilot import add_ball_step, feature_rows, fit, predict

VARIANTS = ("baseline", "architecture_a", "learned_live", "learned_control", "combined")
CONTROL_FEATURES = ["distance_advantage", "sustained_distance_advantage"]
MASK_SECONDS = 4
PERIOD_SECONDS = 10


def segments(rows):
    return (rows["half"].ne(rows["half"].shift()) | rows["frame"].diff().ne(1)).cumsum()


def mask_tracking(rows):
    masked = rows.copy()
    offset = masked["frame"] - masked.groupby("half")["frame"].transform("min")
    hidden = offset % (PERIOD_SECONDS * upstream.FPS) < MASK_SECONDS * upstream.FPS
    masked.loc[hidden, BALL_COLUMNS] = np.nan
    return add_ball_step(masked)


def truth_labels(rows):
    team = rows["ball_team"].map({1: "H", 2: "A"}).fillna("X")
    return team.where(rows["ball_live"].eq(1), "D").to_numpy(dtype="<U1")


def control_features(rows):
    advantage = np.log1p(rows["nearest_away_m"]) - np.log1p(rows["nearest_home_m"])
    sustained = advantage.groupby(segments(rows)).transform(
        lambda signal: signal.rolling(upstream.FPS, center=True, min_periods=1).mean()
    )
    return np.column_stack([advantage, sustained])


def read_match(exports, match):
    rows = feature_rows(exports / f"{match}.csv").reset_index(drop=True)
    if rows["match_id"].iloc[0] != f"DFL-MAT-{match}":
        raise ValueError(f"Unexpected match ID in {match} export")
    return rows


def fit_control(exports):
    values, labels = [], []
    for match in TRAIN:
        rows = read_match(exports, match)
        truth = truth_labels(rows)
        selected = (rows["frame"].to_numpy() % 5 == 0) & np.isin(truth, ["H", "A"])
        values.append(control_features(rows)[selected])
        labels.append(truth[selected] == "H")
    x, y = np.concatenate(values), np.concatenate(labels)
    return fit(np.concatenate([x, -x]), np.concatenate([y, ~y]), CONTROL_FEATURES)


def predict_labels(rows, pitch, variant, live_model, control_model):
    fps = upstream.FPS
    learned_dead = None
    if live_model is not None:
        live_features = features(rows)
        if list(live_features.columns) != live_model["features"]:
            raise ValueError("Live model feature order differs from the experiment")
        learned_dead = 1 - predict(live_features.to_numpy(), live_model)
    learned_home = None
    if control_model:
        control = control_features(rows)
        learned_home = (predict(control, control_model) + 1 - predict(-control, control_model)) / 2

    parts = []
    transition = decode.transition_log_matrix(DEFAULT, fps)
    start = 0
    for _, block in rows.groupby(segments(rows), sort=False):
        stop = start + len(block)
        home = block["nearest_home_m"].fillna(np.inf).to_numpy()
        away = block["nearest_away_m"].fillna(np.inf).to_numpy()
        if variant == "baseline":
            nearest = np.where(np.isfinite(np.minimum(home, away)), (away < home).astype(int), -1)
            labels = smooth(nearest, 50)
        else:
            ball_ok = block[["ball_x_m", "ball_y_m"]].notna().all(axis=1).to_numpy()
            observed = {
                "d_home": block["nearest_home_m"].to_numpy(),
                "d_away": block["nearest_away_m"].to_numpy(),
                "ball_ok": ball_ok,
                "n_players": (block["home_players"] + block["away_players"]).to_numpy(),
                "ball_speed": evidence._rolling_nanmean(block["ball_step_m"].to_numpy() * fps,
                                                        DEFAULT.speed_half_window),
                "player_speed_median": block["player_speed_median_mps"].to_numpy(),
                "out_of_bounds": ((block["ball_x_m"].abs() > pitch[0] / 2 + DEFAULT.out_of_bounds_margin_m)
                                  | (block["ball_y_m"].abs() > pitch[1] / 2 + DEFAULT.out_of_bounds_margin_m)).to_numpy(),
            }
            p_dead, p_home, _ = evidence.evidence(observed, np.zeros(len(block)),
                                                 np.zeros(len(block), dtype=bool), DEFAULT)
            if learned_dead is not None:
                p_dead = learned_dead[start:stop]
            if learned_home is not None:
                p_home = learned_home[start:stop]
            codes = decode.viterbi(evidence.log_emissions(p_dead, p_home, DEFAULT), transition)
            labels = np.array(["H", "A", "D"])[codes]
        parts.append(labels)
        start = stop
    return np.concatenate(parts)


def score(rows, truth, predictions):
    gt, pred = official.label_codes(truth), official.label_codes(predictions)
    starts = np.flatnonzero(segments(rows).diff().ne(0).to_numpy())
    bounds = np.r_[starts, len(rows)]
    runs = [(left + lo, left + hi) for left, right in zip(bounds, bounds[1:])
            for lo, hi in official.valid_runs(gt[left:right])]
    spells = official.spell_scores(gt, pred, runs)
    timing = {}
    for name, outgoing in (("stoppage", False), ("restart", True)):
        errors = []
        missed = spurious = 0
        for run in runs:
            transitions = []
            for codes in (gt, pred):
                changes = official.transitions((codes == official.D).astype(int), [run])
                transitions.append([change for change in changes if (change[1] == 1) == outgoing])
            pairs, missing, extra = official.match_transitions(*transitions, tol=upstream.FPS)
            errors.extend((guess[0] - actual[0]) / upstream.FPS for actual, guess in pairs)
            missed += len(missing)
            spurious += len(extra)
        timing[name] = {"matched": len(errors), "missed": missed, "spurious": spurious,
                        "median_signed_seconds": float(np.median(errors)) if errors else None,
                        "median_absolute_seconds": float(np.median(np.abs(errors))) if errors else None}
    return {"spell_f1": spells["mean_f1"], "true_spells": spells["n_gt"],
            "predicted_spells": spells["n_pred"],
            "macro_f1": official.macro_f1(official.scored_confusion(official.confusion(gt, pred))),
            "timing_within_one_second": timing}


def run_experiment(variant, data, exports, model_path):
    live_model = json.loads(model_path.read_text()) if variant in ("learned_live", "combined") else None
    control_model = fit_control(exports) if variant in ("learned_control", "combined") else None
    results = {}
    for match in (DEV, *TEST):
        rows = read_match(exports, match)
        env = ET.parse(data / match / "metadata.xml").find(".//Environment").attrib
        pitch = float(env["PitchX"]), float(env["PitchY"])
        truth = truth_labels(rows)
        results[match] = {
            "unlabelled_live_frames": int((truth == "X").sum()),
            **{condition: score(rows, truth, predict_labels(observed, pitch, variant, live_model, control_model))
               for condition, observed in (("original", rows), ("masked", mask_tracking(rows)))},
        }
    return {"variant": variant, "mask_seconds": MASK_SECONDS, "period_seconds": PERIOD_SECONDS,
            "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--live-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_experiment(args.variant, args.data, args.exports, args.live_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
