"""Stream the raw ATD and Smart Tagging into arrays indexed by tactical video frame.

ATD raw line: "<frame_id>:<x,y,s>;<x,y,s>;...:<ball group>" with NaN for missing values.
video_frame = frame_id - 1 (DATA.md). Coordinates are normalised 0-1; speeds are m/s.
Smart Tagging times are seconds from the first tactical video frame: frame = round(s * 25).
"""
import difflib
import hashlib
import io
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HOME, AWAY = 0, 1


@dataclass
class Tracking:
    ball_xy: np.ndarray        # (N, 2) normalised, NaN where missing
    player_xy: np.ndarray      # (N, S, 2) normalised, NaN where missing
    player_speed: np.ndarray   # (N, S) m/s, NaN where missing
    team_of_slot: np.ndarray   # (S,) 0 home / 1 away
    team_names: tuple          # (home, away)
    audit: dict = field(default_factory=dict)


def read_meta(meta_path):
    """(team_of_slot, (home_name, away_name)). First team in the file is home (FIFA/EPTS)."""
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(teams)))
    home_id = teams[0].get("id")
    slots = [p.get("teamId") for p in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    team_of_slot = np.array([HOME if t == home_id else AWAY for t in slots], dtype=np.int8)
    names = tuple((t.findtext("Name") or "").strip() for t in teams)
    return team_of_slot, names


PARSER_VERSION = 2  # bump whenever parsing semantics change; invalidates caches


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_atd(meta_path, raw_path, n_frames, cache_dir=None, raw_sha256=None):
    """Parse the ATD pair, using a content-addressed cache when cache_dir is given.

    The fast path (pandas C parser) is used when every line has the documented regular shape;
    anything irregular falls back to the line-by-line parser. Both produce identical arrays.
    """
    team_of_slot, names = read_meta(meta_path)
    cache_path = None
    if cache_dir is not None:
        key = hashlib.sha256("|".join([
            str(PARSER_VERSION), str(n_frames), file_sha256(meta_path),
            raw_sha256 or file_sha256(raw_path)]).encode()).hexdigest()[:32]
        cache_path = Path(cache_dir) / "atd-{}.npz".format(key)
        if cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as z:
                players = z["players"]
                audit = json.loads(str(z["audit"]))
                audit["cache"] = "hit"
                return Tracking(z["ball"], players[:, :, :2], players[:, :, 2],
                                team_of_slot, names, audit)

    try:
        ball, players, audit = _read_atd_fast(raw_path, len(team_of_slot), n_frames)
    except _Fallback as why:
        ball, players, audit = _read_atd_lines(raw_path, len(team_of_slot), n_frames)
        audit["parser"] = "line-by-line ({})".format(why)
    audit["frames_expected"] = n_frames
    audit["n_slots"] = len(team_of_slot)
    audit["ball_present_frames"] = int(np.isfinite(ball).all(axis=1).sum())

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp")
        with open(tmp, "wb") as handle:
            np.savez(handle, ball=ball, players=players, audit=json.dumps(audit))
        os.replace(tmp, cache_path)
        audit = dict(audit, cache="written")
    return Tracking(ball, players[:, :, :2], players[:, :, 2], team_of_slot, names, audit)


_TO_COMMA = bytes.maketrans(b":;", b",,")


class _Fallback(Exception):
    pass


def _read_atd_fast(raw_path, n_slots, n_frames):
    """Vectorised parse: ':' and ';' become ',' so every line is one flat CSV row."""
    import pandas as pd

    with open(raw_path, "rb") as handle:
        first = handle.readline().rstrip(b"\r\n").split(b":")
    if len(first) != 3:
        raise _Fallback("first line is not frame:players:ball")
    groups = first[1].split(b";")
    if len(groups) != n_slots or any(g.count(b",") != 2 for g in groups):
        raise _Fallback("player groups are not x,y,s for every slot")
    ball_width = first[2].count(b",") + 1
    if ball_width < 2:
        raise _Fallback("ball group has fewer than 2 fields")
    n_cols = 1 + 3 * n_slots + ball_width

    with open(raw_path, "rb") as handle:
        data = handle.read().translate(_TO_COMMA)   # one pass, one copy
    if not data.endswith(b"\n"):
        data += b"\n"

    # Every line must have exactly n_cols - 1 separators; otherwise pandas would silently pad.
    buf = np.frombuffer(data, dtype=np.uint8)
    ends = np.flatnonzero(buf == 10)
    starts = np.concatenate(([0], ends[:-1] + 1))
    if np.any(ends == starts) or np.any(ends - starts < n_cols):
        raise _Fallback("blank or truncated line")
    commas = np.add.reduceat(buf == 44, starts, dtype=np.int64)
    del buf
    if np.any(commas != n_cols - 1):
        raise _Fallback("irregular field count on some lines")

    try:
        table = pd.read_csv(io.BytesIO(data), header=None, engine="c", dtype=np.float32,
                            na_filter=True).to_numpy()
    except (ValueError, pd.errors.ParserError) as error:
        raise _Fallback("non-numeric field: {}".format(str(error)[:80]))
    del data
    if table.shape[1] != n_cols:
        raise _Fallback("column count mismatch")

    frame_col = table[:, 0].astype(np.float64)
    if not np.all(np.isfinite(frame_col)) or np.any(frame_col != np.round(frame_col)):
        raise _Fallback("non-integer frame id")
    frames = frame_col.astype(np.int64) - 1
    ok = (frames >= 0) & (frames < n_frames)

    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    players[frames[ok]] = table[ok, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3)
    ball[frames[ok]] = table[ok, 1 + 3 * n_slots:3 + 3 * n_slots]
    n_ok = int(ok.sum())
    n_unique = len(np.unique(frames[ok]))
    audit = {"parser": "fast", "lines": len(table), "malformed_lines": 0,
             "out_of_range_frames": int(len(table) - n_ok),
             "duplicate_frames": n_ok - n_unique, "slot_count_mismatch": 0,
             "nonstandard_groups": 0, "ball_group_widths": {str(ball_width): len(table)},
             "frames_with_record": n_unique}
    return ball, players, audit


def _parse_group(text):
    """x,y,s with NaN padding; tolerates a bare 'NaN' or extra fields. ValueError if non-numeric."""
    fields = text.split(",")[:3]
    values = [float(v) for v in fields]
    return values + [float("nan")] * (3 - len(values))


def _read_atd_lines(raw_path, n_slots, n_frames):
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    audit = {"lines": 0, "malformed_lines": 0, "out_of_range_frames": 0,
             "duplicate_frames": 0, "slot_count_mismatch": 0, "nonstandard_groups": 0,
             "ball_group_widths": {}}
    seen = np.zeros(n_frames, dtype=bool)

    with open(raw_path) as handle:
        for line in handle:
            audit["lines"] += 1
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                audit["malformed_lines"] += 1
                continue
            try:
                frame = int(parts[0]) - 1
            except ValueError:
                audit["malformed_lines"] += 1
                continue
            if not 0 <= frame < n_frames:
                audit["out_of_range_frames"] += 1
                continue

            groups = parts[1].split(";") if parts[1] else []
            if len(groups) != n_slots:
                audit["slot_count_mismatch"] += 1
                groups = (groups + ["NaN,NaN,NaN"] * n_slots)[:n_slots]
            try:
                values = np.array(",".join(groups).split(","), dtype=np.float32)
                row = values.reshape(n_slots, 3)
            except ValueError:
                try:
                    row = np.array([_parse_group(g) for g in groups], dtype=np.float32)
                    audit["nonstandard_groups"] += 1
                except ValueError:
                    audit["malformed_lines"] += 1
                    continue
            if seen[frame]:
                audit["duplicate_frames"] += 1
            seen[frame] = True
            players[frame] = row

            ball_fields = parts[2].split(",")
            width = str(len(ball_fields))
            audit["ball_group_widths"][width] = audit["ball_group_widths"].get(width, 0) + 1
            if len(ball_fields) >= 2:
                try:
                    ball[frame] = (float(ball_fields[0]), float(ball_fields[1]))
                except ValueError:
                    pass

    audit["frames_with_record"] = int(seen.sum())
    return ball, players, audit


def _team_side(tag_value, team_names, min_gap=0.15):
    """Map a Smart Tagging Team string to HOME/AWAY by name similarity, or None."""
    if not tag_value:
        return None
    value = tag_value.strip().lower()
    if value in ("n/a", "na", ""):
        return None
    scores = [difflib.SequenceMatcher(None, value, name.lower()).ratio() for name in team_names]
    if abs(scores[0] - scores[1]) < min_gap:
        return None
    return HOME if scores[0] > scores[1] else AWAY


def read_tagging(xml_path, team_names, fps):
    """[{code, start, end (half-open), side (HOME/AWAY/None), type}] in video frames."""
    events = []
    for inst in ET.parse(xml_path).iter("instance"):
        tags = {lab.findtext("group"): lab.findtext("text") for lab in inst.findall("label")}
        events.append({
            "code": (inst.findtext("code") or "").strip().lower(),
            "start": int(round(float(inst.findtext("start")) * fps)),
            "end": int(round(float(inst.findtext("end")) * fps)),
            "side": _team_side(tags.get("Team"), team_names),
            "type": (tags.get("Type") or "").strip().lower(),
        })
    return events
