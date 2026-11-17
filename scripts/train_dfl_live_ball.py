"""Train and evaluate a CPU live-ball classifier with a fixed DFL match split."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_dfl_sample import FPS
from evaluate_live_ball_pilot import MASK_PERIOD_SECONDS, MASK_SECONDS, restart_metrics
from train_live_ball_pilot import L2, LEARNING_RATE, STEPS, add_ball_step, feature_rows, fit, metrics, predict

TRAIN = ("J03WOH", "J03WOY", "J03WPY", "J03WQQ")
DEV = "J03WR9"
TEST = ("J03WN1", "J03WMX")
WINDOW = 3 * FPS
THRESHOLDS = (0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65)
BALL_COLUMNS = ["ball_x_m", "ball_y_m", "nearest_home_m", "nearest_away_m"]


def features(rows, mask_offset=None):
    if mask_offset is not None:
        rows = rows.copy()
        offset = rows["frame"] - rows.groupby("half")["frame"].transform("min")
        hidden = (offset + mask_offset) % (MASK_PERIOD_SECONDS * FPS) < MASK_SECONDS * FPS
        rows.loc[hidden, BALL_COLUMNS] = np.nan
        add_ball_step(rows)

    segment = (~rows["half"].eq(rows["half"].shift()) | rows["frame"].diff().ne(1)).cumsum()
    signal = pd.DataFrame({
        "log_ball_step": np.log1p(rows["ball_step_m"]),
        "log_nearest_player": np.log1p(rows[["nearest_home_m", "nearest_away_m"]].min(axis=1)),
        "players": rows["home_players"] + rows["away_players"],
        "ball_observed": rows[["ball_x_m", "ball_y_m"]].notna().all(axis=1).astype(float),
    })
    window = signal.groupby(segment, sort=False).rolling(WINDOW, center=True, min_periods=1)
    context = window.mean().droplevel(0).add_suffix("_window_mean")
    peak = window["log_ball_step"].max().droplevel(0).rename("log_ball_step_window_max")
    return pd.concat([signal, context, peak], axis=1)


def scores(rows, probabilities, threshold):
    labels = rows["ball_live"].to_numpy()
    predicted = probabilities >= threshold
    return dict(metrics(labels, predicted), restarts=restart_metrics(rows, labels, predicted))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    matches = {}
    hashes = {}
    for match in (*TRAIN, DEV, *TEST):
        path = args.exports / f"{match}-full.csv"
        rows = feature_rows(path)
        if rows["match_id"].iloc[0] != f"DFL-MAT-{match}":
            raise ValueError(f"Unexpected match ID in {path}")
        matches[match] = rows.reset_index(drop=True)
        with path.open("rb") as stream:
            hashes[match] = hashlib.file_digest(stream, "sha256").hexdigest()

    train_features, train_labels = [], []
    for index, match in enumerate(TRAIN):
        rows = matches[match]
        # Thin training to 5 Hz after computing the 25 Hz context.
        selected = (rows["frame"] % 5 == 0).to_numpy()
        for phase in (None, (index + 1) * FPS):
            train_features.append(features(rows, phase).iloc[selected])
            train_labels.append(rows["ball_live"].to_numpy()[selected])
    x = pd.concat(train_features)
    model = fit(x.to_numpy(), np.concatenate(train_labels), list(x.columns))

    dev_rows = matches[DEV]
    dev_predictions = [predict(features(dev_rows, phase).to_numpy(), model) for phase in (None, 0)]
    dev_labels = dev_rows["ball_live"].to_numpy()
    threshold_scores = {
        threshold: float(np.mean([metrics(dev_labels, p >= threshold)["balanced_accuracy"]
                                  for p in dev_predictions]))
        for threshold in THRESHOLDS
    }
    threshold = max(THRESHOLDS, key=lambda t: (threshold_scores[t], -abs(t - 0.5)))
    model["threshold"] = threshold
    serialized = json.dumps(model, indent=2) + "\n"
    reloaded = json.loads(serialized)

    results = {}
    for match in (DEV, *TEST):
        rows = matches[match]
        results[match] = {}
        for condition, phase in (("original", None), ("masked", 0)):
            values = features(rows, phase).to_numpy()
            probabilities = predict(values, reloaded)
            np.testing.assert_array_equal(probabilities, predict(values, model))
            results[match][condition] = scores(rows, probabilities, threshold)

    report = {
        "split": {"train": TRAIN, "development": DEV, "pilot_benchmark": TEST[0], "unseen_test": TEST[1]},
        "source_sha256": hashes,
        "rows": {match: len(rows) for match, rows in matches.items()},
        "training_examples_after_augmentation": len(x),
        "config": {"steps": STEPS, "learning_rate": LEARNING_RATE, "l2": L2,
                   "window_frames": WINDOW, "mask_seconds": MASK_SECONDS,
                   "mask_period_seconds": MASK_PERIOD_SECONDS},
        "numpy": np.__version__, "pandas": pd.__version__,
        "development_threshold_scores": threshold_scores,
        "threshold": threshold, "results": results,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "model.json").write_text(serialized)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Trained with {len(x):,} examples. Threshold {threshold}. Report: {args.output / 'report.json'}")


if __name__ == "__main__":
    main()
