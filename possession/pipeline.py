"""End-to-end architecture A: assets -> labels. Pure function of inputs + config."""
import numpy as np

from . import decode, evidence, loaders, upstream
from .config import DEFAULT

LETTERS = np.array(["H", "A", "D"])


def labels_from_tracking(tracking, events, cfg=DEFAULT, blocks=None, n_frames=None,
                         pitch=None, fps=None):
    n_frames = n_frames or upstream.N_FRAMES
    blocks = blocks if blocks is not None else upstream.half_blocks()
    pitch = pitch or upstream.PITCH
    fps = fps or upstream.FPS

    feat = evidence.features(tracking, pitch, fps, cfg)
    team_bias, set_piece = evidence.tag_evidence(events, n_frames, cfg)
    p_dead, p_home, controls = evidence.evidence(feat, team_bias, set_piece, cfg)
    log_emit = evidence.log_emissions(p_dead, p_home, cfg)
    codes = decode.decode_halves(log_emit, blocks, cfg, fps, n_frames)

    labels = np.full(n_frames, "X", dtype="<U1")
    graded = codes >= 0
    labels[graded] = LETTERS[codes[graded]]
    diagnostics = {"p_dead": p_dead.astype(np.float32), "p_home": p_home.astype(np.float32),
                   "c_home": controls[0].astype(np.float32),
                   "c_away": controls[1].astype(np.float32)}
    return labels, diagnostics


def run(data_dir, cfg=DEFAULT, cache_dir=None, raw_sha256=None):
    """cache_dir: private folder for the parsed-ATD cache (content-addressed; safe to delete)."""
    paths = upstream.games.asset_paths(upstream.GAME, str(data_dir))
    tracking = loaders.read_atd(paths["atd_meta"], paths["atd_raw"], upstream.N_FRAMES,
                                cache_dir=cache_dir, raw_sha256=raw_sha256)
    events = loaders.read_tagging(paths["smart_tagging"], tracking.team_names, upstream.FPS)
    labels, diagnostics = labels_from_tracking(tracking, events, cfg)
    sides = [e["side"] for e in events]
    tag_audit = {"instances": len(events),
                 "home_tagged": sum(s == loaders.HOME for s in sides),
                 "away_tagged": sum(s == loaders.AWAY for s in sides),
                 "no_side": sum(s is None for s in sides)}
    return labels, diagnostics, {"atd": tracking.audit, "tagging": tag_audit,
                                 "team_names": list(tracking.team_names)}


def write_csv(path, labels):
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
