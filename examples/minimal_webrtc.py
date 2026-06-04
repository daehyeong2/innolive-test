from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from privacy_blur import protect_video_track


def handle_video_track(source_track):
    return protect_video_track(source_track)
