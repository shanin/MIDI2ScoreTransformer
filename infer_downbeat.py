#!/usr/bin/env python3
"""
Beat-informed inference script for MIDI2ScoreTransformer.

Uses manually tapped downbeats from a syncpoints JSON file to hard-constrain
the model's downbeat predictions, instead of letting the model guess measure
structure from the MIDI alone.  Assumes 4/4 throughout.

The syncpoints file is expected to contain JSON of the form:
    [[beat_index, time_seconds, ...], ...]
Each entry marks the START of a measure (downbeat).  Only the annotated
segment — [first_syncpoint, last_syncpoint) — is processed; notes outside
that window are discarded.

Usage:
    python infer_downbeat.py \\
        --midi        FHP08.mid \\
        --syncpoints  FHP08.syncpoints.json \\
        --checkpoint  MIDI2ScoreTF.ckpt \\
        --output      FHP08_score.musicxml
"""

import argparse
import json
import os
import sys
import warnings
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup: works from repo root or any working directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "midi2scoretransformer"))

from models.roformer import Roformer
from tokenizer import MultistreamTokenizer, PARAMS, one_hot_bucketing
from score_utils import postprocess_score


# ---------------------------------------------------------------------------
# Downbeat bucket constants  (4/4 = 4.0 quarter notes per measure)
# ---------------------------------------------------------------------------
# Computed once at import time so they're available everywhere.

def _bucket(value: float) -> int:
    """Return the one-hot bucket index for a downbeat token value."""
    return int(
        one_hot_bucketing(
            torch.tensor([value]), **PARAMS["downbeat"]
        ).argmax(-1).item()
    )

NOT_DOWNBEAT_BUCKET = _bucket(PARAMS["downbeat"]["min"])   # value -1/24  → 0
MEASURE_4_4_BUCKET  = _bucket(4.0)                         # value 4.0 qn → 97


# ---------------------------------------------------------------------------
# Syncpoint loading
# ---------------------------------------------------------------------------

def load_syncpoints(path: str) -> list[tuple[int, float]]:
    """
    Load syncpoints from JSON.

    Handles both 2-element entries [beat_idx, time] and entries with
    trailing fields [beat_idx, time, 0, 0] (as in the last entry of some files).

    Returns
    -------
    list of (beat_index, time_in_seconds)
    """
    raw = json.load(open(path))
    return [(int(entry[0]), float(entry[1])) for entry in raw]


# ---------------------------------------------------------------------------
# MIDI trimming
# ---------------------------------------------------------------------------

def trim_midi_notes(midi_path: str, t_start: float, t_end: float):
    """
    Load MIDI and return only notes with onset in [t_start, t_end).
    Sorting matches MultistreamTokenizer.midi_to_list().

    Returns
    -------
    list of pretty_midi.Note
    """
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(midi_path)
    all_notes = sorted(
        [n for ins in pm.instruments for n in ins.notes],
        key=lambda n: (n.start, n.pitch, n.end - n.start),
    )
    return [n for n in all_notes if t_start <= n.start < t_end]


# ---------------------------------------------------------------------------
# Downbeat constraint builder
# ---------------------------------------------------------------------------

def build_downbeat_constraints(notes, downbeat_times: list[float]) -> np.ndarray:
    """
    Build a boolean array marking which notes should be treated as measure starts.

    For each downbeat time we find the note with the nearest onset and tag it.
    If two downbeat times map to the same note (very rare, only if the note
    list is extremely sparse), only one marker is kept.

    The very first note is always False: the tokenizer encodes the first note's
    downbeat value as -1/24 regardless (no previous measure to measure the
    length of), so forcing it to 4.0 would be incorrect.

    Parameters
    ----------
    notes : list of pretty_midi.Note
        Trimmed, sorted note list.
    downbeat_times : list of float
        Performance times (seconds) of each measure start.

    Returns
    -------
    np.ndarray of bool, shape (len(notes),)
    """
    onsets = np.array([n.start for n in notes])
    constraints = np.zeros(len(notes), dtype=bool)

    for t in downbeat_times:
        idx = int(np.argmin(np.abs(onsets - t)))
        constraints[idx] = True

    # First note: always "no previous measure" in the tokenizer encoding
    if len(constraints) > 0:
        constraints[0] = False

    return constraints


# ---------------------------------------------------------------------------
# Constrained generate  (modified copy of BaseModel.generate)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _constrained_generate(
    self,
    x,
    y=None,
    max_length: int = 512,
    temperature: float = 1.0,
    top_k: int = 1,
    kv_cache: bool = False,
    downbeat_constraints=None,    # BoolTensor (B, T) or None
) -> dict:
    """
    Drop-in replacement for BaseModel.generate() with downbeat constraints.

    downbeat_constraints : torch.BoolTensor of shape (B, chunk_len), optional
        True  at position t → hard-force a 4/4 measure start (bucket 97).
        False at position t → hard-force NOT a measure start (bucket 0).
        None                → fall back to the original offset-decrease heuristic.

    Everything else is identical to the original implementation.
    """
    B, T, _ = x["pitch"].shape
    device = x["pitch"].device
    conf = self.dec_config

    # ---------- start token (all zeros) --------------------------------------
    y_start = {
        "offset":       torch.zeros((B, 1, conf.out_offset_vocab_size),      device=device),
        "downbeat":     torch.zeros((B, 1, conf.out_downbeat_vocab_size),     device=device),
        "duration":     torch.zeros((B, 1, conf.out_duration_vocab_size),     device=device),
        "pitch":        torch.zeros((B, 1, conf.out_pitch_vocab_size),        device=device),
        "accidental":   torch.zeros((B, 1, conf.out_accidental_vocab_size),   device=device),
        "keysignature": torch.zeros((B, 1, conf.out_keysignature_vocab_size), device=device),
        "velocity":     torch.zeros((B, 1, conf.out_velocity_vocab_size),     device=device),
        "grace":        torch.zeros((B, 1, conf.out_grace_vocab_size),        device=device),
        "trill":        torch.zeros((B, 1, conf.out_trill_vocab_size),        device=device),
        "staccato":     torch.zeros((B, 1, conf.out_staccato_vocab_size),     device=device),
        "voice":        torch.zeros((B, 1, conf.out_voice_vocab_size),        device=device),
        "stem":         torch.zeros((B, 1, conf.out_stem_vocab_size),         device=device),
        "hand":         torch.zeros((B, 1, conf.out_hand_vocab_size),         device=device),
        "pad":          torch.zeros((B, 1),                                   device=device).long(),
    }

    # ---------- encoder -------------------------------------------------------
    if "encoder" in self.hyperparameters["components"]:
        enc_out  = self.forward_enc(x, attention_mask=x["pad"])
        enc_mask = x["pad"]
    else:
        enc_out = enc_mask = None

    # ---------- seed decoder (optional context) -------------------------------
    if y is None:
        y = y_start
        past_kv = None
    else:
        y = {k: torch.cat([y_start[k], y[k]], dim=1) for k in y}
        past_kv = self.forward_dec(
            input_streams={k: torch.roll(v[:, :-1], -1, 1) for k, v in y.items()},
            encoder_hidden_states=enc_out,
            encoder_attention_mask=enc_mask,
            past_key_values=None,
            use_cache=True,
        )[1]

    # ---------- autoregressive loop -------------------------------------------
    for _ in range(max_length + 1 - y["pad"].shape[1]):

        # Position in the output being generated right now (0-indexed,
        # start token excluded).  Used for constraint lookup.
        out_pos = y["pad"].shape[1] - 1

        if kv_cache:
            y_pred, past_kv = self.forward_dec(
                input_streams={k: v[:, -1:] for k, v in y.items()},
                encoder_hidden_states=enc_out,
                encoder_attention_mask=enc_mask,
                past_key_values=past_kv,
                use_cache=True,
            )
        else:
            y_pred = self.forward_dec(
                input_streams={k: torch.roll(v, -1, 1) for k, v in y.items()},
                encoder_hidden_states=enc_out,
                encoder_attention_mask=enc_mask,
            )

        for k in y.keys():
            logits = y_pred[k][:, -1, :] / temperature

            # ---- downbeat: either constrained or original heuristic ----------
            if k == "downbeat":
                if (
                    downbeat_constraints is not None
                    and out_pos < downbeat_constraints.shape[1]
                ):
                    is_db = downbeat_constraints[:, out_pos].to(device)  # (B,) bool

                    forced_db    = logits.new_full(logits.shape, float("-inf"))
                    forced_db[:, MEASURE_4_4_BUCKET] = 0.0

                    forced_no_db = logits.new_full(logits.shape, float("-inf"))
                    forced_no_db[:, NOT_DOWNBEAT_BUCKET] = 0.0

                    logits = torch.where(is_db.unsqueeze(-1), forced_db, forced_no_db)
                else:
                    # Original heuristic: force a downbeat whenever the predicted
                    # offset decreases (measure wrap-around detected)
                    if y["offset"].shape[1] > 1:
                        offset_decreased = (
                            y_pred["offset"][:, -1].argmax(-1)
                            < y["offset"][:, -2].argmax(-1)
                        )
                        logits[offset_decreased, 0] = float("-inf")

            # ---- accidental validity (unchanged from original) ---------------
            if k == "accidental":
                impossible = {
                    0: [1, 4],   1: [0, 2, 5], 2: [1, 3],    3: [2, 4, 5],
                    4: [0, 3],   5: [1, 4],    6: [0, 2, 5], 7: [1, 3],
                    8: [0, 2, 4, 5], 9: [1, 3], 10: [2, 4, 5], 11: [0, 3],
                }
                never = [0, 4, 6]
                for i in range(logits.shape[0]):
                    pc = y["pitch"][i, -1].argmax().item() % 12
                    logits[i, impossible.get(pc, []) + never] = float("-inf")

            # ---- top-k + sample ----------------------------------------------
            if top_k is not None:
                v_k, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v_k[:, [-1]]] = float("-inf")

            if k == "pad":
                probs = torch.cat(
                    [1 - F.sigmoid(logits), F.sigmoid(logits)], dim=-1
                )
                next_tok = probs.argmax(-1, keepdim=True)
                y[k] = torch.cat([y[k], next_tok], dim=1)
            else:
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
                next_tok = F.one_hot(next_tok, num_classes=y_pred[k].shape[-1])
                y[k] = torch.cat([y[k], next_tok], dim=1)

        # Zero out all streams for padded positions
        pad_mask = y["pad"][:, -1] == 0
        for k in y:
            if k != "pad":
                y[k][pad_mask, -1] = 0

    # Strip the start token
    for k in y:
        y[k] = y[k][:, 1:]
    y["pad"] = y["pad"].unsqueeze(-1).float()
    return y


# ---------------------------------------------------------------------------
# Chunked inference wrapper
# ---------------------------------------------------------------------------

def infer_constrained(
    x: dict,
    model: Roformer,
    downbeat_constraints: np.ndarray,
    chunk: int = 512,
    overlap: int = 64,
    kv_cache: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Chunked autoregressive inference with hard downbeat constraints.

    Temporarily monkey-patches model.generate so that no model files need
    to be modified.

    Parameters
    ----------
    x : dict
        Bucketed MIDI token streams (no batch dim), from bucket_midi().
    model : Roformer
    downbeat_constraints : np.ndarray of bool, shape (n_notes,)
        True  → force measure start at this note.
        False → force non-measure-start at this note.
    chunk, overlap : int
        Sliding window parameters (same as original infer()).
    """
    if chunk <= overlap:
        raise ValueError(f"chunk ({chunk}) must be > overlap ({overlap}).")

    # Temporarily replace generate
    original_generate = model.generate
    model.generate = MethodType(_constrained_generate, model)

    try:
        x = {k: v.unsqueeze(0).to(model.device) for k, v in x.items()}
        n_notes = x["pitch"].shape[1]
        use_cuda_amp = model.device.type == "cuda"

        db_tensor = torch.from_numpy(downbeat_constraints.astype(bool))  # (n_notes,)

        y_full = None
        for i in range(0, max(n_notes - overlap, 1), chunk - overlap):
            if verbose:
                print(f"  Decoding notes {i}–{min(i+chunk, n_notes)} / {n_notes} ...", end="\r")

            x_chunk = {k: v[:, i : i + chunk] for k, v in x.items()}
            # Constraint slice for this chunk, with batch dimension
            db_chunk = db_tensor[i : i + chunk].unsqueeze(0)  # (1, slice_len)

            def _forward_chunk() -> dict:
                if i == 0 or overlap == 0:
                    return model.generate(
                        x=x_chunk,
                        top_k=1,
                        max_length=chunk,
                        kv_cache=kv_cache,
                        downbeat_constraints=db_chunk,
                    )
                y_context = {
                    k: v[:, -overlap:] if k != "pad" else v[:, -overlap:, 0]
                    for k, v in y_full.items()
                }
                y_hat_inner = model.generate(
                    x=x_chunk,
                    y=y_context,
                    top_k=1,
                    max_length=chunk,
                    kv_cache=kv_cache,
                    downbeat_constraints=db_chunk,
                )
                return {k: v[:, overlap:] for k, v in y_hat_inner.items()}

            with torch.no_grad():
                if use_cuda_amp:
                    with torch.autocast(device_type="cuda", enabled=True):
                        y_hat = _forward_chunk()
                else:
                    y_hat = _forward_chunk()

            y_full = (
                y_hat
                if y_full is None
                else {k: torch.cat([y_full[k], y_hat[k]], dim=1) for k in y_full}
            )

        if verbose:
            print()

    finally:
        model.generate = original_generate  # always restore

    return {k: v[0].cpu() for k, v in y_full.items()}


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Beat-informed MIDI→MusicXML inference using manually tapped downbeats."
    )
    parser.add_argument("--midi",        type=str, required=True,
                        help="Performance MIDI file.")
    parser.add_argument("--syncpoints",  type=str, required=True,
                        help="Syncpoints JSON file (manually tapped downbeats).")
    parser.add_argument("--checkpoint",  type=str, required=True,
                        help="Model checkpoint (.ckpt).")
    parser.add_argument("--output",      type=str, default=None,
                        help="Output MusicXML path (default: <midi_stem>_score.musicxml).")
    parser.add_argument("--chunk",       type=int, default=512,
                        help="Notes per inference window (default: 512).")
    parser.add_argument("--overlap",     type=int, default=64,
                        help="Context overlap between windows (default: 64).")
    parser.add_argument("--no_kv_cache", action="store_true",
                        help="Disable KV cache (slower, less memory).")
    parser.add_argument("--cpu",         action="store_true",
                        help="Force CPU inference.")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.splitext(os.path.abspath(args.midi))[0] + "_score.musicxml"

    device = torch.device("cpu") if args.cpu else get_device()
    print(f"Device      : {device}")

    # --- Syncpoints ------------------------------------------------------------
    syncpoints = load_syncpoints(args.syncpoints)
    t_first    = syncpoints[0][1]
    t_last     = syncpoints[-1][1]

    # All syncpoints are downbeats except the last, which is used purely as the
    # right boundary for trimming (we'd have no "next downbeat" to anchor its
    # measure length).  Notes at [t_first, t_last) are kept.
    downbeat_times = [t for _, t in syncpoints[:-1]]

    print(f"Syncpoints  : {len(syncpoints)} entries  →  {len(downbeat_times)} measure starts")
    print(f"Region      : [{t_first:.3f}s, {t_last:.3f}s)  ({t_last - t_first:.1f}s)")

    # --- Trim MIDI -------------------------------------------------------------
    notes = trim_midi_notes(args.midi, t_first, t_last)
    print(f"Notes       : {len(notes)}  (after trimming to annotated region)")
    if len(notes) == 0:
        raise RuntimeError("No notes in the annotated region — check your syncpoints file.")

    # --- Build constraints -----------------------------------------------------
    constraints = build_downbeat_constraints(notes, downbeat_times)
    n_db = int(constraints.sum())
    print(f"Constrained : {n_db} / {len(downbeat_times)} downbeats matched to notes")
    if n_db < len(downbeat_times):
        print(f"  (warning: {len(downbeat_times) - n_db} downbeats collapsed onto the same note "
              f"as another — this is harmless but unusual)")

    # --- Tokenize (reuse the raw tensors already in memory) -------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        midi_streams = {
            "onset":    torch.FloatTensor([n.start    for n in notes]),
            "duration": torch.FloatTensor([n.end - n.start for n in notes]),
            "pitch":    torch.LongTensor( [n.pitch    for n in notes]),
            "velocity": torch.LongTensor( [n.velocity for n in notes]),
        }
        x = MultistreamTokenizer.bucket_midi(midi_streams)

    # --- Load model ------------------------------------------------------------
    print(f"Loading     : {args.checkpoint}")
    model = Roformer.load_from_checkpoint(args.checkpoint, map_location=device)
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model       : {n_params / 1e6:.1f}M parameters")

    # --- Constrained inference -------------------------------------------------
    print(f"Inferring   : chunk={args.chunk}, overlap={args.overlap}, "
          f"kv_cache={not args.no_kv_cache}")
    y_hat = infer_constrained(
        x,
        model,
        constraints,
        chunk=args.chunk,
        overlap=args.overlap,
        kv_cache=not args.no_kv_cache,
        verbose=True,
    )

    # --- Decode to score -------------------------------------------------------
    print("Decoding    : tokens → MusicXML score ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Pass original note list so performance timings are embedded as
        # MetronomeMarks (useful for playback / alignment downstream)
        score = MultistreamTokenizer.detokenize_mxl(y_hat, midi_sequence=notes)
        score = postprocess_score(score, inPlace=True)

    # --- Write -----------------------------------------------------------------
    print(f"Output      : {args.output}")
    score.write("musicxml", fp=args.output)
    print("Done.")


if __name__ == "__main__":
    main()
