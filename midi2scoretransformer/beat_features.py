from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BeatFeatureConfig:
    phase_bins: int = 48
    max_beats_per_bar: int = 12
    time_tolerance_s: float = 1e-3

    @property
    def beat_in_bar_vocab_size(self) -> int:
        # 0..max_beats_per_bar-1 plus unknown at last index
        return self.max_beats_per_bar + 1

    @property
    def beat_phase_vocab_size(self) -> int:
        # 0..phase_bins-1 plus unknown at last index
        return self.phase_bins + 1


def _as_float_array(x: Any) -> np.ndarray:
    if x is None:
        return np.array([], dtype=np.float64)
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    return np.array(list(x), dtype=np.float64)


def _sorted_time_signature_segments(
    perf_time_signatures: dict[str, list[Any]] | dict[float, list[Any]] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ASAP: perf_time_signatures maps time-> [time_signature_string, number_of_beats]
    Keys are typically JSON strings.
    """
    if not perf_time_signatures:
        return np.array([], dtype=np.float64), np.array([], dtype=np.int64)
    items: list[tuple[float, int]] = []
    for k, v in perf_time_signatures.items():
        try:
            t = float(k)
        except Exception:
            t = float(str(k))
        # v: [ts_string, number_of_beats]
        beats_per_bar = int(v[1])
        items.append((t, beats_per_bar))
    items.sort(key=lambda kv: kv[0])
    times = np.array([t for t, _ in items], dtype=np.float64)
    beats = np.array([b for _, b in items], dtype=np.int64)
    return times, beats


def _beats_within_measures_from_downbeats(
    downbeats_s: np.ndarray,
    beats_per_bar: int,
) -> np.ndarray:
    """
    Build a beat grid from downbeats only, by linear interpolation within each bar.
    Returns beat times including downbeats; last measure is excluded (needs a next downbeat).
    """
    if len(downbeats_s) < 2 or beats_per_bar <= 0:
        return np.array([], dtype=np.float64)
    out: list[float] = []
    for i in range(len(downbeats_s) - 1):
        t0 = float(downbeats_s[i])
        t1 = float(downbeats_s[i + 1])
        if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
            continue
        step = (t1 - t0) / beats_per_bar
        for j in range(beats_per_bar):
            out.append(t0 + j * step)
    return np.array(out, dtype=np.float64)


def compute_midi_beat_features_onehot(
    onsets_s: torch.Tensor,
    *,
    beats_s: list[float] | np.ndarray | None,
    downbeats_s: list[float] | np.ndarray | None,
    perf_time_signatures: dict[str, list[Any]] | dict[float, list[Any]] | None,
    beat_types: dict[str, str] | dict[float, str] | None,
    config: BeatFeatureConfig = BeatFeatureConfig(),
) -> dict[str, torch.Tensor]:
    """
    Compute per-note beat features for a performance MIDI note list.

    Parameters
    ----------
    onsets_s:
        Tensor of absolute note onsets in seconds, shape (T,).
    beats_s / downbeats_s:
        Beat and downbeat times in seconds (ASAP annotations).
        If beats_s is empty but downbeats_s exists, we will synthesize beats
        using a constant beats-per-bar estimate from perf_time_signatures.
    perf_time_signatures:
        ASAP dict mapping time-> [ts_string, number_of_beats].
    beat_types:
        ASAP dict mapping time-> 'db' | 'b' | 'bR'. We treat bR regions as unknown.
    config:
        Controls phase bucket count and unknown handling.

    Returns
    -------
    Dict with keys 'beat_in_bar' and 'beat_phase', each a float one-hot tensor
    shaped (T, vocab_size).
    """
    if onsets_s.ndim != 1:
        onsets_s = onsets_s.reshape(-1)

    T = int(onsets_s.shape[0])
    device = onsets_s.device

    beats = _as_float_array(beats_s)
    downbeats = _as_float_array(downbeats_s)
    ts_change_times, ts_beats_per_bar = _sorted_time_signature_segments(perf_time_signatures)

    if beats.size == 0 and downbeats.size > 0:
        # Best-effort beat synthesis from downbeats: use the first annotated beats-per-bar.
        beats_per_bar_default = int(ts_beats_per_bar[0]) if ts_beats_per_bar.size > 0 else 4
        beats = _beats_within_measures_from_downbeats(downbeats, beats_per_bar_default)

    unknown_beat_in_bar = config.beat_in_bar_vocab_size - 1
    unknown_phase = config.beat_phase_vocab_size - 1

    beat_in_bar_idx = torch.full((T,), unknown_beat_in_bar, device=device, dtype=torch.long)
    beat_phase_idx = torch.full((T,), unknown_phase, device=device, dtype=torch.long)

    if beats.size < 2:
        return {
            "beat_in_bar": F.one_hot(beat_in_bar_idx, num_classes=config.beat_in_bar_vocab_size).float(),
            "beat_phase": F.one_hot(beat_phase_idx, num_classes=config.beat_phase_vocab_size).float(),
        }

    beats_np = beats
    onsets_np = onsets_s.detach().cpu().numpy().astype(np.float64, copy=False)

    # Determine which beat indices correspond to bR ("rubato/unknown beat position") annotations.
    br_indices: set[int] = set()
    if beat_types:
        for k, v in beat_types.items():
            if str(v) != "bR":
                continue
            try:
                t = float(k)
            except Exception:
                t = float(str(k))
            i = int(np.searchsorted(beats_np, t, side="left"))
            # Accept either exact-ish match at i or i-1.
            cand = []
            if 0 <= i < len(beats_np):
                cand.append(i)
            if 0 <= i - 1 < len(beats_np):
                cand.append(i - 1)
            best = None
            best_err = None
            for j in cand:
                err = abs(beats_np[j] - t)
                if best is None or err < best_err:
                    best = j
                    best_err = err
            if best is not None and best_err is not None and best_err <= config.time_tolerance_s:
                br_indices.add(int(best))

    # For each note, locate the beat interval [beats[i], beats[i+1]).
    i = np.searchsorted(beats_np, onsets_np, side="right") - 1
    valid = (i >= 0) & (i < (len(beats_np) - 1))

    if valid.any():
        i_valid = i[valid]
        t_valid = onsets_np[valid]
        b0 = beats_np[i_valid]
        b1 = beats_np[i_valid + 1]
        denom = (b1 - b0)
        # Guard against duplicate beat times.
        good_interval = denom > 1e-6

        # Phase buckets
        phase = np.zeros_like(denom, dtype=np.float64)
        phase[good_interval] = (t_valid[good_interval] - b0[good_interval]) / denom[good_interval]
        phase = np.clip(phase, 0.0, 0.999999)
        phase_bucket = np.floor(phase * config.phase_bins).astype(np.int64)

        # beats-per-bar at each note time (handles TS changes)
        if ts_change_times.size > 0:
            ts_idx = np.searchsorted(ts_change_times, t_valid, side="right") - 1
            ts_idx = np.clip(ts_idx, 0, len(ts_beats_per_bar) - 1)
            beats_per_bar = ts_beats_per_bar[ts_idx].astype(np.int64)
        else:
            beats_per_bar = np.full_like(phase_bucket, 4, dtype=np.int64)

        # Downbeat anchor: map each note to the most recent downbeat (bar start).
        if downbeats.size > 0:
            db_i = np.searchsorted(downbeats, t_valid, side="right") - 1
            has_db = db_i >= 0
            db_time = np.where(has_db, downbeats[np.clip(db_i, 0, len(downbeats) - 1)], np.nan)
            # Convert that downbeat time to a beat index anchor.
            db_beat_idx = np.searchsorted(beats_np, db_time, side="right") - 1
            db_beat_idx = np.clip(db_beat_idx, 0, len(beats_np) - 1)
        else:
            has_db = np.zeros_like(phase_bucket, dtype=bool)
            db_beat_idx = np.zeros_like(phase_bucket, dtype=np.int64)

        beat_in_bar = np.full_like(phase_bucket, unknown_beat_in_bar, dtype=np.int64)
        ok_bar = has_db & (beats_per_bar > 0) & (beats_per_bar <= config.max_beats_per_bar)
        beat_in_bar[ok_bar] = (i_valid[ok_bar] - db_beat_idx[ok_bar]) % beats_per_bar[ok_bar]

        # Apply bR masking: if the enclosing beat is bR, set unknown.
        is_br = np.array([int(ii) in br_indices for ii in i_valid], dtype=bool)

        beat_in_bar_t = torch.from_numpy(beat_in_bar).to(device=device, dtype=torch.long)
        phase_t = torch.from_numpy(phase_bucket).to(device=device, dtype=torch.long)
        phase_t[torch.from_numpy(~good_interval | is_br).to(device=device)] = unknown_phase
        beat_in_bar_t[torch.from_numpy(is_br).to(device=device)] = unknown_beat_in_bar

        beat_in_bar_idx[torch.from_numpy(valid).to(device=device)] = beat_in_bar_t
        beat_phase_idx[torch.from_numpy(valid).to(device=device)] = phase_t

    return {
        "beat_in_bar": F.one_hot(beat_in_bar_idx, num_classes=config.beat_in_bar_vocab_size).float(),
        "beat_phase": F.one_hot(beat_phase_idx, num_classes=config.beat_phase_vocab_size).float(),
    }

