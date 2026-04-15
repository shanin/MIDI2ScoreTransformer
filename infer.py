#!/usr/bin/env python3
"""
Single-file inference script for MIDI2ScoreTransformer.
Converts a performance MIDI file to a MusicXML score.

Usage:
    python infer.py --midi path/to/performance.mid --checkpoint path/to/MIDI2ScoreTF.ckpt

    # Specify output path:
    python infer.py --midi perf.mid --checkpoint model.ckpt --output my_score.musicxml

    # Force CPU (e.g. no GPU available):
    python infer.py --midi perf.mid --checkpoint model.ckpt --cpu

    # Larger chunks for long pieces (requires more VRAM/RAM):
    python infer.py --midi perf.mid --checkpoint model.ckpt --chunk 1024 --overlap 128

Requirements:
    Install all dependencies from requirements.txt, plus manually clone and install muster:
        git clone https://github.com/TimFelixBeyer/amtevaluation.github.io
        pip install -e amtevaluation.github.io --break-system-packages

Notes:
    - This script deliberately does NOT import from utils.py to avoid its hard dependency on
      muster/score_transformer at import time. Those are only needed for evaluation.
    - The infer() function below is a clean copy of the one in utils.py.
    - midi_sequence is passed to detokenize_mxl so that performance timing (seconds) is
      embedded as MetronomeMarks in the output score, preserving the tempo curve.
"""

import argparse
import os
import sys
import warnings

# ---------------------------------------------------------------------------
# Path setup — works whether you run from repo root or from anywhere else
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "midi2scoretransformer"))

import torch
from models.roformer import Roformer
from tokenizer import MultistreamTokenizer
from score_utils import postprocess_score


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Inference (chunked autoregressive sliding window)
# ---------------------------------------------------------------------------

def infer(
    x: dict,
    model: Roformer,
    overlap: int = 64,
    chunk: int = 512,
    kv_cache: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Run chunked autoregressive inference over a long MIDI sequence.

    The input is split into overlapping windows of `chunk` notes. For all
    windows after the first, the last `overlap` predicted tokens are fed back
    as decoder context, giving the model continuity across chunk boundaries.

    Parameters
    ----------
    x : dict[str, torch.Tensor]
        Bucketed MIDI token streams from MultistreamTokenizer.tokenize_midi().
        Tensors have shape (n_notes, vocab_size) — no batch dimension yet.
    model : Roformer
        Loaded model in eval mode.
    overlap : int
        Number of previously generated tokens carried over as decoder context.
    chunk : int
        Number of input notes processed per forward pass. Must be > overlap.
    kv_cache : bool
        Whether to use the decoder KV cache (faster, recommended).
    verbose : bool
        Print per-chunk progress.

    Returns
    -------
    dict[str, torch.Tensor]
        Predicted score token streams, each shape (n_notes, vocab_size).
        "pad" has shape (n_notes, 1).
    """
    if chunk <= overlap:
        raise ValueError(f"chunk ({chunk}) must be greater than overlap ({overlap}).")

    # Add batch dimension
    x = {k: v.unsqueeze(0).to(model.device) for k, v in x.items()}
    n_notes = x["pitch"].shape[1]
    device_type = model.device.type  # "cuda", "mps", or "cpu"

    # autocast: beneficial on CUDA (fp16), harmless elsewhere
    use_autocast = device_type == "cuda"

    y_full = None
    steps = range(0, max(n_notes - overlap, 1), chunk - overlap)

    for i in steps:
        if verbose:
            end_note = min(i + chunk, n_notes)
            print(f"  Decoding notes {i}–{end_note} / {n_notes} ...", end="\r")

        x_chunk = {k: v[:, i : i + chunk] for k, v in x.items()}

        if i == 0 or overlap == 0:
            # First chunk: no prior context
            with torch.no_grad():
                ctx = torch.autocast(device_type=device_type, enabled=use_autocast)
                with ctx:
                    y_hat = model.generate(
                        x=x_chunk,
                        top_k=1,
                        max_length=chunk,
                        kv_cache=kv_cache,
                    )
        else:
            # Subsequent chunks: seed decoder with last `overlap` generated tokens
            y_context = {
                k: v[:, -overlap:] if k != "pad" else v[:, -overlap:, 0]
                for k, v in y_full.items()
            }
            with torch.no_grad():
                ctx = torch.autocast(device_type=device_type, enabled=use_autocast)
                with ctx:
                    y_hat = model.generate(
                        x=x_chunk,
                        y=y_context,
                        top_k=1,
                        max_length=chunk,
                        kv_cache=kv_cache,
                    )
            # Drop the overlap tokens that were already in y_full
            y_hat = {k: v[:, overlap:] for k, v in y_hat.items()}

        if y_full is None:
            y_full = y_hat
        else:
            y_full = {k: torch.cat([y_full[k], y_hat[k]], dim=1) for k in y_full}

    if verbose:
        print()  # newline after \r progress line

    # Remove batch dimension, move to CPU
    return {k: v[0].cpu() for k, v in y_full.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert a performance MIDI to MusicXML using MIDI2ScoreTransformer."
    )
    parser.add_argument(
        "--midi",
        type=str,
        required=True,
        help="Path to the input performance MIDI file (.mid / .midi).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint file (MIDI2ScoreTF.ckpt).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output path for the MusicXML file. "
            "Defaults to <input_basename>_score.musicxml in the same directory as the MIDI."
        ),
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=512,
        help="Number of notes per inference chunk (default: 512). Increase for long pieces if VRAM allows.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Context overlap between consecutive chunks in notes (default: 64).",
    )
    parser.add_argument(
        "--no_kv_cache",
        action="store_true",
        help="Disable the decoder KV cache. Slower but uses less memory.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even if a GPU is available.",
    )
    args = parser.parse_args()

    # --- Resolve output path ---------------------------------------------------
    if args.output is None:
        base = os.path.splitext(os.path.abspath(args.midi))[0]
        args.output = base + "_score.musicxml"

    # --- Device ----------------------------------------------------------------
    device = torch.device("cpu") if args.cpu else get_device()
    print(f"Device : {device}")

    # --- Load model ------------------------------------------------------------
    print(f"Loading checkpoint: {args.checkpoint}")
    model = Roformer.load_from_checkpoint(args.checkpoint, map_location=device)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model  : {n_params / 1e6:.1f}M parameters")

    # --- Tokenize MIDI ---------------------------------------------------------
    print(f"Input  : {args.midi}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = MultistreamTokenizer.tokenize_midi(args.midi)
    n_notes = x["pitch"].shape[0]
    print(f"Notes  : {n_notes}")

    # --- Inference -------------------------------------------------------------
    print(f"Inferring (chunk={args.chunk}, overlap={args.overlap}, kv_cache={not args.no_kv_cache}) ...")
    y_hat = infer(
        x,
        model,
        chunk=args.chunk,
        overlap=args.overlap,
        kv_cache=not args.no_kv_cache,
        verbose=True,
    )

    # --- Decode to music21 Score -----------------------------------------------
    print("Decoding tokens to score ...")
    # Pass original MIDI notes so that performance timing (seconds) is embedded
    # as MetronomeMarks in the output — this preserves the tempo curve.
    midi_notes = MultistreamTokenizer.midi_to_list(args.midi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = MultistreamTokenizer.detokenize_mxl(y_hat, midi_sequence=midi_notes)
        score = postprocess_score(score, inPlace=True)

    # --- Write output ----------------------------------------------------------
    print(f"Output : {args.output}")
    score.write("musicxml", fp=args.output)
    print("Done.")


if __name__ == "__main__":
    main()
