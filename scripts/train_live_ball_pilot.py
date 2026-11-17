"""Fit and reload a CPU logistic model on the one-match DFL development sample."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["nearest_home_m", "nearest_away_m", "home_players", "away_players", "ball_step_m"]
TRAIN_HALF = "firstHalf"
CHECK_HALF = "secondHalf"
STEPS = 500
LEARNING_RATE = 0.1
L2 = 0.01


def feature_rows(path):
    rows = pd.read_csv(path).sort_values(["half", "frame"])
    if rows["match_id"].isna().any() or rows["match_id"].nunique() != 1:
        raise ValueError("The pilot expects exactly one DFL match")
    if set(rows["half"]) != {TRAIN_HALF, CHECK_HALF} or rows.duplicated(["half", "frame"]).any():
        raise ValueError("Expected unique frames in both halves")
    if not rows["ball_live"].isin([0, 1]).all():
        raise ValueError("Expected binary ball_live labels")
    for _, half in rows.groupby("half"):
        if set(half["ball_live"]) != {0, 1}:
            raise ValueError("Both target classes must occur in each half")
    return add_ball_step(rows)


def add_ball_step(rows):
    grouped = rows.groupby("half")
    dx = grouped["ball_x_m"].diff()
    dy = grouped["ball_y_m"].diff()
    rows["ball_step_m"] = np.hypot(dx, dy).where(grouped["frame"].diff() == 1)
    return rows


def standardize(values, model):
    valid = np.isfinite(values)
    filled = np.where(valid, values, model["mean"])
    return (filled - model["mean"]) / model["scale"]


def sigmoid(logits):
    return np.exp(-np.logaddexp(0, -logits))


def fit(values, labels, features=FEATURES):
    finite_values = np.where(np.isfinite(values), values, np.nan)
    if np.isnan(finite_values).all(axis=0).any():
        raise ValueError("Each feature needs a finite observation in the training data")
    mean = np.nanmean(finite_values, axis=0)
    scale = np.nanstd(finite_values, axis=0)
    model = {"features": features, "mean": mean.tolist(), "scale": np.where(scale > 0, scale, 1).tolist()}
    x = standardize(values, model)
    weights = np.zeros(x.shape[1])
    bias = 0.0
    for _ in range(STEPS):
        error = sigmoid(x @ weights + bias) - labels
        weights -= LEARNING_RATE * (x.T @ error / len(x) + L2 * weights)
        bias -= LEARNING_RATE * error.mean()
    return dict(model, weights=weights.tolist(), bias=float(bias))


def predict(values, model):
    return sigmoid(standardize(values, model) @ model["weights"] + model["bias"])


def metrics(labels, predicted):
    confusion = [[int(((labels == truth) & (predicted == guess)).sum())
                  for guess in (0, 1)] for truth in (0, 1)]
    recalls = [confusion[k][k] / sum(confusion[k]) for k in (0, 1)]
    return {"accuracy": float((labels == predicted).mean()), "balanced_accuracy": sum(recalls) / 2,
            "dead_recall": recalls[0], "live_recall": recalls[1], "confusion_dead_live": confusion}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = feature_rows(args.sample)
    train = rows[rows["half"] == TRAIN_HALF]
    check = rows[rows["half"] == CHECK_HALF]
    train_x, train_y = train[FEATURES].to_numpy(), train["ball_live"].to_numpy()
    check_x, check_y = check[FEATURES].to_numpy(), check["ball_live"].to_numpy()
    model = fit(train_x, train_y)
    probabilities = predict(check_x, model)

    args.output.mkdir(parents=True, exist_ok=False)
    model_path = args.output / "model.json"
    model_path.write_text(json.dumps(model, indent=2) + "\n")
    reloaded = json.loads(model_path.read_text())
    np.testing.assert_array_equal(probabilities, predict(check_x, reloaded))

    report = {
        "match_id": rows["match_id"].iloc[0],
        "sample_sha256": hashlib.sha256(args.sample.read_bytes()).hexdigest(),
        "split": {"train": TRAIN_HALF, "smoke_check": CHECK_HALF},
        "train_rows": len(train), "check_rows": len(check),
        "config": {"steps": STEPS, "learning_rate": LEARNING_RATE, "l2": L2, "threshold": 0.5},
        "numpy": np.__version__, "pandas": pd.__version__,
        "train": metrics(train_y, predict(train_x, model) >= 0.5),
        "smoke_check": metrics(check_y, probabilities >= 0.5),
        "train_majority_baseline": metrics(check_y, np.full(len(check_y), int(train_y.mean() > 0.5))),
        "reload_predictions_identical": True,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
