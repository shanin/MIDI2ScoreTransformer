"""
Given a path to a model checkpoint and a dataset split ('test', 'train', 'validation', 'all'):
1) Run inference and SAVE predicted MusicXML files for each song.
2) Compute per-song metrics + aggregate metrics.
3) Save results to an Excel with 2 sheets:
   - aggregate: overall metrics
   - per_song: per-song metrics (with paths)
"""

import argparse
import os
import sys
import warnings
from typing import Dict, Any, Tuple, List

import torch
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

# Make repo imports work (same as your original script)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import ASAPDataset
from models.roformer import Roformer
from tokenizer import MultistreamTokenizer
from utils import infer, pad_batch, score_similarity_normalized
from score_utils import postprocess_score
from muster import muster
from beat_features import BeatFeatureConfig

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def asap_relpath(sample_midi_path: str) -> str:
    """
    Try to build a stable relative path under asap-dataset/...
    If not found, fallback to basename folder structure.
    """
    marker = "asap-dataset"
    if marker in sample_midi_path:
        # keep everything after ".../asap-dataset/"
        rel = sample_midi_path.split(marker + os.sep, 1)[-1]
        return rel
    return os.path.basename(sample_midi_path)


def build_pred_xml_path(out_dir: str, sample_midi_path: str, filename: str = "pred_score.musicxml") -> str:
    rel = asap_relpath(sample_midi_path)
    rel_dir = os.path.dirname(rel)
    pred_dir = os.path.join(out_dir, rel_dir)
    ensure_dir(pred_dir)
    return os.path.join(pred_dir, filename)


def parse_song_fields_from_rel(rel: str) -> Dict[str, Any]:
    """
    ASAP typical: Composer/Piece/...
    We'll expose a few helpful columns for case study.
    """
    parts = rel.split(os.sep)
    composer = parts[0] if len(parts) >= 1 else ""
    piece = parts[1] if len(parts) >= 2 else ""
    rest = os.sep.join(parts[2:]) if len(parts) >= 3 else ""
    return {"composer": composer, "piece": piece, "subpath": rest}


def eval_from_tokens_and_write_xml(
    y_hat_single: Dict[str, torch.Tensor],
    gt_mxl_path: str,
    pred_xml_path: str,
) -> Tuple[Dict[str, Dict[str, float] | None], str]:
    """
    Detokenize -> postprocess -> save predicted xml -> compute metrics.
    Returns:
      - sim dict: {"mxl <-> gt_mxl": {...}, "muster": {...}}
      - pred_xml_path
    """
    mxl = MultistreamTokenizer.detokenize_mxl(y_hat_single)
    mxl = postprocess_score(mxl, inPlace=True)

    # Write predicted xml
    # music21: .write("musicxml", fp=...)
    try:
        mxl.write("musicxml", fp=pred_xml_path)
    except Exception as e:
        # Still compute metrics, but record write failure by setting path to empty
        pred_xml_path = ""
        print(f"[WARN] Failed to write xml: {pred_xml_path} ({e})")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = {
            "mxl <-> gt_mxl": score_similarity_normalized(mxl, gt_mxl_path, full=False),
            "muster": muster(mxl, gt_mxl_path),
        }

    return sim, pred_xml_path


def flatten_metrics(sim: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten nested dict:
      sim["muster"]["PitchER"] -> "muster.PitchER"
      sim["mxl <-> gt_mxl"]["NoteDeletion"] -> "mxl.NoteDeletion"
    """
    out: Dict[str, Any] = {}
    if sim is None:
        return out

    # normalize names
    if "muster" in sim and isinstance(sim["muster"], dict):
        for k, v in sim["muster"].items():
            out[f"muster.{k}"] = v
    if "mxl <-> gt_mxl" in sim and isinstance(sim["mxl <-> gt_mxl"], dict):
        for k, v in sim["mxl <-> gt_mxl"].items():
            out[f"mxl.{k}"] = v
    return out


def aggregate_metrics(df_per_song: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate rules:
    - For TP/FP/FN/TN-like metrics: sum
    - For others: mean over non-null
    """
    metric_cols = [c for c in df_per_song.columns if c.startswith("muster.") or c.startswith("mxl.")]
    rows = []
    for col in metric_cols:
        series = pd.to_numeric(df_per_song[col], errors="coerce")
        metric_name = col.split(".", 1)[1] if "." in col else col
        group = col.split(".", 1)[0] if "." in col else "metric"

        if any(x in metric_name for x in ["TP", "FP", "FN", "TN"]):
            val = float(series.fillna(0).sum())
            agg_type = "sum"
        else:
            # mean over non-null
            val = float(series.dropna().mean()) if series.dropna().shape[0] > 0 else float("nan")
            agg_type = "mean"

        rows.append({"group": group, "metric": metric_name, "agg": agg_type, "value": val})

    df_agg = pd.DataFrame(rows).sort_values(["group", "metric"]).reset_index(drop=True)
    return df_agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True, help="train/test/validation/all")
    parser.add_argument("--model", type=str, required=True, help="path to lightning checkpoint")
    parser.add_argument("--data_dir", type=str, default="/mnt/ssd/hbli/datasets/PM2S_dataset/midi2scoretransformer/")
    parser.add_argument("--out_dir", type=str, required=True, help="directory to save predicted xmls")
    parser.add_argument("--excel_path", type=str, required=True, help="output excel path, e.g. results.xlsx")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--kv_cache", action="store_true", help="enable kv_cache in generate")
    parser.add_argument("--n_jobs", type=int, default=16, help="parallel jobs for eval")
    parser.add_argument("--fast_eval", action="store_true", help="keep for compatibility; not used here")

    # external beats (ASAP-style *_annotations.txt next to MIDI)
    parser.add_argument("--annotations_suffix", type=str, default="_annotations.txt")
    parser.add_argument("--beat_phase_bins", type=int, default=48, help="Must match the beat-augmented model config.")
    parser.add_argument("--max_beats_per_bar", type=int, default=12, help="Must match the beat-augmented model config.")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    ensure_dir(os.path.dirname(args.excel_path) or ".")

    print(f"Device: {device}")
    print("Loading dataset metadata")
    q = ASAPDataset(args.data_dir, args.split)

    # Build (midi_path, gt_xml_path)
    paths: List[Tuple[str, str]] = []
    for i in range(len(q.metadata)):
        sample = q.metadata.iloc[i]
        sample_path = sample["performance_MIDI_external"].replace("{ASAP}", f"{q.data_dir}asap-dataset")
        gt_path = os.path.dirname(sample_path) + "/xml_score.musicxml"
        paths.append((sample_path, gt_path))

    # Tokenize all inputs (like your original script)
    from lightning.pytorch import seed_everything
    seed_everything(42, workers=True)

    print("Tokenizing all songs")
    inputs = []
    lengths = []
    beat_feat_cfg = BeatFeatureConfig(phase_bins=args.beat_phase_bins, max_beats_per_bar=args.max_beats_per_bar)
    for midi_path, _ in tqdm(paths):
        x = MultistreamTokenizer.tokenize_midi_with_beats(
            midi_path,
            annotations_suffix=args.annotations_suffix,
            beat_feat_cfg=beat_feat_cfg,
        )
        inputs.append({k: v.unsqueeze(0).to(device) for k, v in x.items()})
        lengths.append(int(x["pitch"].shape[0]))

    # Sort by length for efficient padding
    sorted_data = sorted(zip(lengths, inputs, paths), key=lambda x: x[0])
    lengths, inputs, paths = zip(*sorted_data)

    print("Loading model")
    model = Roformer.load_from_checkpoint(args.model)
    model.to(device)
    model.eval()

    print("Running batched inference")
    y_full = None
    for i in tqdm(range(0, len(inputs), args.batch_size)):
        x = pad_batch(list(inputs[i : i + args.batch_size]))
        y_hat = infer(
            x,
            model,
            overlap=args.overlap,
            chunk=args.chunk,
            kv_cache=args.kv_cache,
            verbose=False,
        )
        if y_full is None:
            y_full = y_hat
        else:
            y_full = pad_batch([y_full, y_hat])

    # Evaluate per-song + write xml (parallel)
    print("Writing predicted XML + computing per-song metrics")

    def _one_song(i: int, midi_path: str, gt_path: str, length_i: int) -> Dict[str, Any]:
        # slice tokens back to true length for this song
        y_hat_i = {k: v[i, :length_i] for k, v in y_full.items()}

        pred_path = build_pred_xml_path(args.out_dir, midi_path, filename="pred_score.musicxml")
        sim, pred_written = eval_from_tokens_and_write_xml(y_hat_i, gt_path, pred_path)
        flat = flatten_metrics(sim)

        rel = asap_relpath(midi_path)
        fields = parse_song_fields_from_rel(rel)

        row = {
            "idx": i,
            "relpath": rel,
            "midi_path": midi_path,
            "gt_xml_path": gt_path,
            "pred_xml_path": pred_written,
            "n_notes": length_i,
            **fields,
            **flat,
        }
        return row

    rows = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(_one_song)(i, p[0], p[1], l)
        for i, (p, l) in enumerate(zip(paths, lengths))
    )

    df_per_song = pd.DataFrame(rows)

    # Aggregate
    df_agg = aggregate_metrics(df_per_song)
    # Add a small header block for counts
    header = pd.DataFrame(
        [
            {"group": "meta", "metric": "split", "agg": "", "value": args.split},
            {"group": "meta", "metric": "n_songs", "agg": "", "value": int(df_per_song.shape[0])},
            {"group": "meta", "metric": "model_ckpt", "agg": "", "value": args.model},
            {"group": "meta", "metric": "out_dir", "agg": "", "value": args.out_dir},
            {"group": "meta", "metric": "chunk", "agg": "", "value": args.chunk},
            {"group": "meta", "metric": "overlap", "agg": "", "value": args.overlap},
            {"group": "meta", "metric": "batch_size", "agg": "", "value": args.batch_size},
            {"group": "meta", "metric": "kv_cache", "agg": "", "value": bool(args.kv_cache)},
        ]
    )
    df_agg_full = pd.concat([header, df_agg], ignore_index=True)

    # Save Excel
    print(f"Saving Excel -> {args.excel_path}")
    with pd.ExcelWriter(args.excel_path, engine="openpyxl") as writer:
        df_agg_full.to_excel(writer, sheet_name="aggregate", index=False)
        df_per_song.to_excel(writer, sheet_name="per_song", index=False)

    print("Done.")
    print("Excel:", args.excel_path)
    print("Pred XML dir:", args.out_dir)


if __name__ == "__main__":
    main()
