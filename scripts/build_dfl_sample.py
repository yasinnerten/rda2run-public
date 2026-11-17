"""Export a small live/dead training sample from one IDSSE match."""

import argparse
import csv
import math
from statistics import median
from pathlib import Path
import xml.etree.ElementTree as ET

FPS = 25
HALVES = ("firstHalf", "secondHalf")


def tracking_frames(path, match_id, teams):
    """Stream the requested team tracks without retaining the XML tree."""
    frameset = None
    for event, node in ET.iterparse(path, events=("start", "end")):
        if node.tag == "FrameSet":
            if event == "start":
                if node.attrib["MatchId"] != match_id:
                    raise ValueError("Tracking and metadata match IDs differ")
                frameset = node
            else:
                frameset = None
                node.clear()
        elif event == "end" and node.tag == "Frame" and frameset is not None:
            section = frameset.attrib["GameSection"]
            team = frameset.attrib["TeamId"]
            if section in HALVES and team in teams:
                yield section, team, int(node.attrib["N"]), node.attrib
            frameset.remove(node)


def position(frame):
    x, y = (float(frame.get(axis, "nan")) for axis in ("X", "Y"))
    return (x, y) if math.isfinite(x) and math.isfinite(y) else None


def build(metadata, tracking, frames_per_half, possession=False):
    general = ET.parse(metadata).find("./MatchInformation/General")
    match_id = general.attrib["MatchId"]
    teams = {general.attrib["HomeTeamId"]: "home", general.attrib["GuestTeamId"]: "away"}
    speeds = {}
    first_frame = {}
    rows = {}
    for half, _, frame, fields in tracking_frames(tracking, match_id, {"BALL"}):
        start = first_frame.setdefault(half, frame)
        if frame < start:
            raise ValueError(f"Ball frames out of order in {half}")
        if frames_per_half is not None and frame >= start + frames_per_half:
            continue
        key = (half, frame)
        if key in rows:
            raise ValueError(f"Duplicate ball frame {key}")
        status = fields.get("BallStatus")
        if status not in {"0", "1"}:
            continue
        ball = position(fields)
        rows[key] = {
            "match_id": match_id, "half": half, "frame": frame,
            "ball_x_m": ball[0] if ball else None, "ball_y_m": ball[1] if ball else None,
            "home_players": 0, "away_players": 0,
            "nearest_home_m": None, "nearest_away_m": None, "ball_live": int(status),
        }

        if possession:
            rows[key]["ball_team"] = fields.get("BallPossession", "")

    if {half for half, _ in rows} != set(HALVES):
        raise ValueError("Expected labelled ball frames in both halves")
    for half, team, frame, fields in tracking_frames(tracking, match_id, teams):
        row = rows.get((half, frame))
        if row is None:
            continue
        player = position(fields)
        if player is None:
            continue
        side = teams[team]
        row[f"{side}_players"] += 1
        if possession:
            speed = float(fields.get("S", "nan")) / 3.6
            if math.isfinite(speed):
                speeds.setdefault((half, frame), []).append(speed)
        if row["ball_x_m"] is not None:
            distance = math.hypot(player[0] - row["ball_x_m"], player[1] - row["ball_y_m"])
            key = f"nearest_{side}_m"
            if row[key] is None or distance < row[key]:
                row[key] = distance
    if possession:
        for key, row in rows.items():
            row["player_speed_median_mps"] = median(speeds[key]) if key in speeds else None
    return [rows[key] for key in sorted(rows)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--tracking", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    window = parser.add_mutually_exclusive_group()
    window.add_argument("--seconds-per-half", type=int, default=300)
    window.add_argument("--full-match", action="store_true")
    parser.add_argument("--possession", action="store_true", help="Include DFL team labels and player speed")
    args = parser.parse_args()
    if args.seconds_per_half <= 0:
        parser.error("--seconds-per-half must be positive")
    frames_per_half = None if args.full_match else args.seconds_per_half * FPS
    rows = build(args.metadata, args.tracking, frames_per_half, args.possession)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
