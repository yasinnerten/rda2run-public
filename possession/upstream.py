"""Access to the pinned official checkout (games.py constants, validator).

The official repository is cloned by scripts/bootstrap.sh into ./more-than-a-hack-2027 at the
revision in docs/upstream-commit.txt. We import its games.py instead of copying constants, so
frame counts and half boundaries have exactly one source.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path(os.environ.get("MTAH_UPSTREAM_DIR", ROOT / "more-than-a-hack-2027"))

if not (UPSTREAM / "games.py").is_file():
    raise ImportError(
        "official checkout not found at {}; run scripts/bootstrap.sh first".format(UPSTREAM))
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

import games  # noqa: E402

GAME = games.GAME
N_FRAMES = games.source_frames("tactical", GAME)
FPS = games.GAMES[GAME]["fps"]
PITCH = games.GAMES[GAME]["pitch"]  # metres (length, width)


def half_blocks():
    """[(half, first_video_frame, last_video_frame)] -- inclusive, graded frames only."""
    return games.half_video_blocks(GAME)
