"""Freeze the external models or generate the rda2run Submission 1 CSV."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from dfl_spell_experiment import TRAIN, fit_control, predict_labels
from possession import evidence, loaders, pipeline, upstream
from possession.config import DEFAULT
from train_live_ball_pilot import add_ball_step

ROOT = Path(__file__).resolve().parents[1]


def freeze(args):
    external_inputs = [args.live_model, *(args.exports / f"{match}.csv" for match in TRAIN)]
    bundle = {
        "variant": "combined",
        "live_model": json.loads(args.live_model.read_text()),
        "team_model": fit_control(args.exports),
        "decoder": DEFAULT.as_dict(),
        "source_sha256": {path.name: loaders.file_sha256(path) for path in external_inputs},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(bundle, stream, indent=2, sort_keys=True)
        stream.write("\n")


def infer(args):
    revision = subprocess.check_output(
        ["git", "-C", str(upstream.UPSTREAM), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != (ROOT / "docs/upstream-commit.txt").read_text().strip():
        raise ValueError("Official checkout revision differs from the project pin")
    bundle = json.loads(args.model.read_text())
    if bundle["variant"] != "combined" or bundle["decoder"] != DEFAULT.as_dict():
        raise ValueError("Frozen variant or decoder configuration does not match Submission 1")

    paths = upstream.games.asset_paths(upstream.GAME, str(args.data.resolve()))
    metadata, raw = (Path(paths[key]) for key in ("atd_meta", "atd_raw"))
    inputs = {path.name: loaders.file_sha256(path) for path in (metadata, raw)}
    tracking = loaders.read_atd(metadata, raw, upstream.N_FRAMES)
    pitch, fps = upstream.PITCH, upstream.FPS
    derived = evidence.features(tracking, pitch, fps, DEFAULT)
    home_slots = tracking.team_of_slot == loaders.HOME
    home_players = np.isfinite(tracking.player_xy[:, home_slots]).all(axis=2).sum(axis=1)
    labels = np.full(upstream.N_FRAMES, "X", dtype="<U1")
    for half, first, last in upstream.half_blocks():
        block = slice(first, last + 1)
        ball = tracking.ball_xy[block].astype(np.float64) * pitch
        rows = pd.DataFrame({
            "half": half, "frame": np.arange(first, last + 1),
            "ball_x_m": ball[:, 0], "ball_y_m": ball[:, 1],
            "nearest_home_m": derived["d_home"][block], "nearest_away_m": derived["d_away"][block],
            "home_players": home_players[block],
            "away_players": derived["n_players"][block] - home_players[block],
            "player_speed_median_mps": derived["player_speed_median"][block],
        })
        labels[block] = predict_labels(add_ball_step(rows), pitch, "combined",
                                       bundle["live_model"], bundle["team_model"])

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "rda2run.csv"
    manifest_path = args.output / "manifest.json"
    if csv_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Submission files already exist in {args.output}")
    pipeline.write_csv(csv_path, labels)
    subprocess.run([sys.executable, str(upstream.UPSTREAM / "evaluation/validate_submission.py"),
                    str(csv_path)], check=True)
    manifest = {
        "team": "rda2run", "submission": 1, "variant": "combined",
        "upstream_revision": revision,
        "code_revision": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                                 text=True).strip(),
        "model_sha256": loaders.file_sha256(args.model), "input_sha256": inputs,
        "csv_sha256": loaders.file_sha256(csv_path), "frames": len(labels),
        "label_counts": {name: int((labels == name).sum()) for name in "HADX"},
        "numpy": np.__version__, "pandas": pd.__version__, "python": sys.version.split()[0],
        "validation": "passed",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(required=True)
    freezing = modes.add_parser("freeze")
    freezing.add_argument("--exports", type=Path, required=True)
    freezing.add_argument("--live-model", type=Path, required=True)
    freezing.add_argument("--output", type=Path, required=True)
    freezing.set_defaults(action=freeze)
    inference = modes.add_parser("infer")
    inference.add_argument("--model", type=Path, required=True)
    inference.add_argument("--data", type=Path, required=True)
    inference.add_argument("--output", type=Path, required=True)
    inference.set_defaults(action=infer)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
