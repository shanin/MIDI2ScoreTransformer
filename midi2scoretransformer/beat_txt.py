from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BeatTxtConfig:
    """
    ASAP-style beat annotation text files like:

      <t1>\\t<t2>\\t<kind>[,<ts_string>,<beats_per_bar>]

    Examples:
      1.513086  1.513086  db,2/4,6
      1.897701  1.897701  b
      2.224223  2.224223  db

    We interpret:
      - kind: 'b' or 'db' (and any other tag is preserved in beat_types)
      - time: use the *first* float column (t1) as seconds
      - optional time signature segment: when present on a db line, record it as
        perf_time_signatures[time] = [ts_string, beats_per_bar]
    """

    suffix: str = "_annotations.txt"


def infer_annotations_txt_path(midi_path: str, *, config: BeatTxtConfig = BeatTxtConfig()) -> str:
    midi_path = os.path.abspath(midi_path)
    stem, _ext = os.path.splitext(midi_path)
    return stem + config.suffix


def load_annotations_txt(path: str) -> dict[str, Any]:
    beats: list[float] = []
    downbeats: list[float] = []
    beat_types: dict[str, str] = {}
    perf_time_signatures: dict[str, list[Any]] = {}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue

            try:
                t = float(parts[0])
            except Exception:
                continue

            tag = parts[2]
            items = [x.strip() for x in tag.split(",") if x.strip()]
            if not items:
                continue

            kind = items[0]
            beats.append(t)
            beat_types[str(t)] = kind

            if kind == "db":
                downbeats.append(t)
                if len(items) >= 3:
                    ts_string = items[1]
                    try:
                        beats_per_bar = int(items[2])
                    except Exception:
                        beats_per_bar = None
                    if beats_per_bar is not None:
                        perf_time_signatures[str(t)] = [ts_string, beats_per_bar]

    # Ensure monotonic ordering just in case
    beats = sorted(set(beats))
    downbeats = sorted(set(downbeats))

    return {
        "beats_s": beats,
        "downbeats_s": downbeats,
        "beat_types": beat_types,
        "perf_time_signatures": perf_time_signatures,
    }

