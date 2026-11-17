"""Evaluate the frozen DFL live-ball model on one separate match."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from build_dfl_sample import FPS
from train_live_ball_pilot import FEATURES, add_ball_step, feature_rows, metrics, predict

MASK_SECONDS = 2
MASK_PERIOD_SECONDS = 5
RESTART_WINDOW_SECONDS = 2


def restart_metrics(rows, labels, predicted):
    # Split runs at state changes, missing frame IDs, or halftime.
    consecutive = rows["half"].eq(rows["half"].shift()) & rows["frame"].diff().eq(1)
    adjacent = consecutive.to_numpy()
    changes = np.flatnonzero(~adjacent[1:] | (labels[1:] != labels[:-1])) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes, len(labels)]
    restarts = already_live = missed = 0
    delays = []
    for start, end in zip(starts, ends):
        if not adjacent[start] or labels[start] != 1:
            continue
        restarts += 1
        if predicted[start - 1]:
            already_live += 1
            continue
        deadline = min(end, start + RESTART_WINDOW_SECONDS * FPS + 1)
        hits = np.flatnonzero(predicted[start:deadline])
        if hits.size:
            delays.append(int(hits[0]))
        else:
            missed += 1
    return {
        "labelled_restarts": restarts,
        "already_predicting_live": already_live,
        "detected_within_window": len(delays),
        "missed_within_window": missed,
        "median_detection_delay_seconds": float(np.median(delays) / FPS) if delays else None,
    }


def evaluate(rows, model, threshold):
    labels = rows["ball_live"].to_numpy()
    predictions = predict(rows[FEATURES].to_numpy(), model) >= threshold
    return dict(metrics(labels, predictions), restarts=restart_metrics(rows, labels, predictions))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True, help="Directory with model.json and report.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_bytes = (args.pilot / "model.json").read_bytes()
    model = json.loads(model_bytes)
    training = json.loads((args.pilot / "report.json").read_text())
    rows = feature_rows(args.sample)
    match_id = rows["match_id"].iloc[0]
    if match_id == training["match_id"]:
        parser.error("Choose a match different from the pilot training match")
    if model["features"] != FEATURES:
        parser.error("The model feature order differs from the exporter")
    threshold = training["config"]["threshold"]

    masked = rows.copy()
    offset = masked["frame"] - masked.groupby("half")["frame"].transform("min")
    missing = offset % (MASK_PERIOD_SECONDS * FPS) < MASK_SECONDS * FPS
    masked.loc[missing, ["ball_x_m", "ball_y_m", "nearest_home_m", "nearest_away_m"]] = np.nan
    masked = add_ball_step(masked)

    report = {
        "training_match": training["match_id"],
        "held_out_match": match_id,
        "rows": len(rows),
        "sample_sha256": hashlib.sha256(args.sample.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "threshold": threshold,
        "restart_window_seconds": RESTART_WINDOW_SECONDS,
        "mask": {"seconds": MASK_SECONDS, "period_seconds": MASK_PERIOD_SECONDS,
                 "frames": int(missing.sum())},
        "original": evaluate(rows, model, threshold),
        "masked": evaluate(masked, model, threshold),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
