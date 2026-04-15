#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader

from beat_features import BeatFeatureConfig, compute_midi_beat_features_onehot
from config import MyModelConfig
from dataset import ASAPDataset, sha256
from tokenizer import MultistreamTokenizer
from train import TrainRoformer
from utils import cat_dict, cut_pad


class BeatAugmentedASAPDataset(ASAPDataset):
    def __init__(
        self,
        *args,
        asap_annotations_json: str | None = None,
        beat_phase_bins: int = 48,
        max_beats_per_bar: int = 12,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if asap_annotations_json is None:
            asap_annotations_json = os.path.join(self.data_dir, "asap-dataset", "asap_annotations.json")
        with open(asap_annotations_json, "r") as f:
            self._asap_annots = json.load(f)
        self._beat_cfg = BeatFeatureConfig(phase_bins=beat_phase_bins, max_beats_per_bar=max_beats_per_bar)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # Copy of ASAPDataset.__getitem__ with beat feature injection before bucketing.
        if self.split == "train":
            idx = torch.multinomial(self.lengths, 1, replacement=True).item()
        sample = self.metadata.iloc[idx]
        sample_path = sample["performance_MIDI_external"].replace("{ASAP}", f"{self.data_dir}asap-dataset")
        sample_dir = os.path.dirname(sample_path)

        pkl_file = os.path.join(self.data_dir, "cache", f"{sha256(sample_path + self.id)}.pkl")

        if (not self.cache) or (not os.path.exists(pkl_file)):
            score_path = sample_dir + "/xml_score.musicxml"
            input_stream = MultistreamTokenizer.parse_midi(sample_path)
            output_stream = MultistreamTokenizer.parse_mxl(score_path)
            torch.save((input_stream, output_stream), pkl_file)

        input_stream, output_stream = torch.load(pkl_file, weights_only=False)

        if self.augmentations.get("transpose", False):
            max_semitones = int(self.augmentations["transpose"])
            shift = random.randint(-max_semitones, max_semitones)
            input_stream["pitch"], output_stream["pitch"], output_stream["accidental"], output_stream["keysignature"] = self._transpose(
                shift,
                midi_stream=input_stream["pitch"],
                mxl_stream=output_stream["pitch"],
                accidental_stream=output_stream["accidental"],
                keysignature_stream=output_stream["keysignature"],
            )

        if (v := self.augmentations.get("tempo_jitter", False)):
            alpha = random.uniform(*v)
            input_stream["onset"] = input_stream["onset"] * alpha
        if (v := self.augmentations.get("duration_jitter", False)):
            beta = random.uniform(*v)
            input_stream["duration"] = input_stream["duration"] * beta
        if (v := self.augmentations.get("onset_jitter", False)):
            jitter = 1 + torch.randn(input_stream["onset"].shape) * v
            inter_note_intervals = torch.diff(input_stream["onset"], prepend=torch.tensor([0]), dim=0)
            input_stream["onset"] = torch.cumsum(inter_note_intervals * jitter, dim=0)
        if (v := self.augmentations.get("velocity_jitter", False)):
            input_stream["velocity"] += torch.round(torch.randn(input_stream["velocity"].shape) * v).long()
            input_stream["velocity"] = torch.clamp(input_stream["velocity"], 1, 127)

        if self.return_continous:
            return input_stream, output_stream

        # ---- Beat feature injection (continuous onset seconds) -----------------
        asap_key = sample["performance_MIDI_external"].replace("{ASAP}/", "")
        ann = self._asap_annots.get(asap_key, None)
        if ann is None:
            beat_feats = compute_midi_beat_features_onehot(
                input_stream["onset"],
                beats_s=None,
                downbeats_s=None,
                perf_time_signatures=None,
                beat_types=None,
                config=self._beat_cfg,
            )
        else:
            beat_feats = compute_midi_beat_features_onehot(
                input_stream["onset"],
                beats_s=ann.get("performance_beats", None),
                downbeats_s=ann.get("performance_downbeats", None),
                perf_time_signatures=ann.get("perf_time_signatures", None),
                beat_types=ann.get("performance_beats_type", None),
                config=self._beat_cfg,
            )

        # ---- Standard bucketing + add beat streams -----------------------------
        input_stream = MultistreamTokenizer.bucket_midi(input_stream)
        output_stream = MultistreamTokenizer.bucket_mxl(output_stream)
        input_stream.update(beat_feats)

        if self.seq_length is not None:
            seq_length = self.seq_length
        else:
            seq_length = max(len(input_stream["onset"]), len(output_stream["offset"])) + 256

        chunks_path = sample_path.replace(".mid", "_chunks.json")
        if os.path.exists(chunks_path):
            chunk_annots = json.load(open(chunks_path))
        else:
            # Fallback: treat the whole performance/score as a single chunk.
            n_midi = int(input_stream["pad"].shape[0])
            n_mxl = int(output_stream["pad"].shape[0])
            chunk_annots = {"midi": [list(range(n_midi))], "mxl": [list(range(n_mxl))]}

        if (v := self.augmentations.get("random_crop", False)):
            min_beats = 16
            if v is True:
                n_0 = random.randint(0, max(len(chunk_annots["midi"]) - min_beats, 0))
            elif isinstance(v, int):
                average = sum([len(x) for x in chunk_annots["midi"]]) / len(chunk_annots["midi"])
                n_0 = random.choice(range(0, max(len(chunk_annots["midi"]) - min_beats, 1), max(1, int(v / average))))
            else:
                raise ValueError("Invalid random_crop value")
        else:
            n_0 = 0

        def process_chunk(stream, chunk, padding, length):
            if padding == "per-beat":
                return {k: cut_pad(v[chunk], length, 0) for k, v in stream.items()}
            return {k: v[chunk] for k, v in stream.items()}

        new_input_stream = None
        for midi_chunk, mxl_chunk in zip(chunk_annots["midi"][n_0:], chunk_annots["mxl"][n_0:]):
            length = max(len(midi_chunk), len(mxl_chunk))
            if (
                new_input_stream is not None
                and len(new_input_stream["onset"]) + length > seq_length + self.augmentations.get("random_shift", 0)
            ):
                break
            in_chunk = process_chunk(input_stream, midi_chunk, self.padding, length)
            out_chunk = process_chunk(output_stream, mxl_chunk, self.padding, length)
            if new_input_stream is None:
                new_input_stream = in_chunk
                new_output_stream = out_chunk
            else:
                new_input_stream = cat_dict(new_input_stream, in_chunk)
                new_output_stream = cat_dict(new_output_stream, out_chunk)

        if (v := self.augmentations.get("random_shift", False)):
            shift = random.randint(0, v - 1)
            for k, vv in new_input_stream.items():
                new_input_stream[k] = vv[shift:]
            for k, vv in new_output_stream.items():
                new_output_stream[k] = vv[shift:]

        if self.padding is not None:
            for k, vv in new_input_stream.items():
                input_stream[k] = cut_pad(vv, seq_length, 0)
            for k, vv in new_output_stream.items():
                output_stream[k] = cut_pad(vv, seq_length, 0)

        if self.return_paths:
            return input_stream, output_stream, sample_path, sample_dir + "/xml_score.musicxml"
        return input_stream, output_stream


def build_model_with_beats(args) -> TrainRoformer:
    in_beat_in_bar_vocab_size = args.max_beats_per_bar + 1
    in_beat_phase_vocab_size = args.beat_phase_bins + 1

    enc_conf = MyModelConfig(
        is_decoder=False,
        add_cross_attention=False,
        is_autoregressive=False,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
        embedding_size=args.hidden_size,
        bias=True,
        in_beat_in_bar_vocab_size=in_beat_in_bar_vocab_size,
        in_beat_phase_vocab_size=in_beat_phase_vocab_size,
    )
    dec_conf = MyModelConfig(
        is_decoder=True,
        add_cross_attention=True,
        is_autoregressive=True,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
        embedding_size=args.hidden_size,
        bias=True,
        in_beat_in_bar_vocab_size=in_beat_in_bar_vocab_size,
        in_beat_phase_vocab_size=in_beat_phase_vocab_size,
    )

    hyper = {"components": ["encoder", "decoder"]}
    return TrainRoformer(
        enc_configuration=enc_conf,
        dec_configuration=dec_conf,
        hyperparameters=hyper,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pad_loss_weight=args.pad_loss_weight,
        teacher_keep_prob=args.teacher_keep_prob,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--asap_annotations_json", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="pm2s_roformer_beats")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke_test_only",
        action="store_true",
        help="Only run a single batch key/shape/model forward check (no Lightning required).",
    )
    # logging
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging via WandbLogger.")
    parser.add_argument("--wandb_project", type=str, default="MIDI2ScoreTransformer")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default="", help="Comma-separated list of tags.")
    parser.add_argument("--wandb_mode", type=str, default=None, help="W&B mode: 'online', 'offline', or 'disabled'.")
    parser.add_argument("--wandb_log_model", type=str, default=None, help="WandbLogger log_model setting (e.g. 'all').")

    # beat feature config
    parser.add_argument("--beat_phase_bins", type=int, default=48)
    parser.add_argument("--max_beats_per_bar", type=int, default=12)

    # dataloader
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seq_length", type=int, default=512)

    # training schedule
    parser.add_argument("--max_steps", type=int, default=40000)
    parser.add_argument("--warmup_steps", type=int, default=4000)

    # optimizer
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # model arch
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--intermediate_size", type=int, default=3072)
    parser.add_argument("--dropout", type=float, default=0.1)

    # loss tricks
    parser.add_argument("--pad_loss_weight", type=float, default=0.1)
    parser.add_argument("--teacher_keep_prob", type=float, default=0.25)

    # runtime/debug
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)

    # output
    parser.add_argument("--out_dir", type=str, default="./runs")

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    train_set = BeatAugmentedASAPDataset(
        data_dir=args.data_dir,
        split="train",
        seq_length=args.seq_length,
        cache=True,
        padding="per-beat",
        augmentations={},
        return_continous=False,
        asap_annotations_json=args.asap_annotations_json,
        beat_phase_bins=args.beat_phase_bins,
        max_beats_per_bar=args.max_beats_per_bar,
    )
    val_set = BeatAugmentedASAPDataset(
        data_dir=args.data_dir,
        split="validation",
        seq_length=args.seq_length,
        cache=True,
        padding="per-beat",
        augmentations={},
        return_continous=False,
        asap_annotations_json=args.asap_annotations_json,
        beat_phase_bins=args.beat_phase_bins,
        max_beats_per_bar=args.max_beats_per_bar,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=max(0, args.num_workers // 2),
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    model = build_model_with_beats(args)

    # Quick runtime sanity check on keys + dims.
    xb, yb = next(iter(train_loader))
    assert "beat_in_bar" in xb and "beat_phase" in xb, f"missing beat keys: {xb.keys()}"
    assert xb["beat_in_bar"].shape[-1] == args.max_beats_per_bar + 1
    assert xb["beat_phase"].shape[-1] == args.beat_phase_bins + 1
    _ = model.forward_enc(xb, attention_mask=xb["pad"])

    if args.smoke_test_only:
        return

    try:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
        from pytorch_lightning.loggers import CSVLogger
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "pytorch_lightning is required for training. "
            "Install it, or re-run with --smoke_test_only to only validate wiring."
        ) from e

    pl.seed_everything(args.seed, workers=True)

    loggers = [CSVLogger(save_dir=args.out_dir, name=args.run_name)]
    if args.use_wandb:
        try:
            from pytorch_lightning.loggers import WandbLogger
        except Exception as e:
            raise ImportError(
                "W&B logging requested but WandbLogger could not be imported. "
                "Install: `pip install wandb` (and ensure Lightning supports it)."
            ) from e
        if args.wandb_mode is not None:
            os.environ["WANDB_MODE"] = args.wandb_mode
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            save_dir=args.out_dir,
            log_model=args.wandb_log_model,
            tags=[t.strip() for t in args.wandb_tags.split(",") if t.strip()] if args.wandb_tags else None,
        )
        try:
            wandb_logger.log_hyperparams(vars(args))
        except Exception:
            pass
        loggers.append(wandb_logger)

    logger = loggers if len(loggers) > 1 else loggers[0]
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(args.out_dir, args.run_name, "checkpoints"),
        filename="{step}-{val/loss_total:.4f}",
        save_top_k=3,
        monitor="val/loss_total",
        mode="min",
        save_last=True,
        every_n_train_steps=1000,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = [args.gpu_id] if accelerator == "gpu" else 1

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=args.precision if accelerator == "gpu" else "32-true",
        max_steps=args.max_steps,
        gradient_clip_val=0.5,
        logger=logger,
        callbacks=[ckpt_cb, lr_cb],
        log_every_n_steps=20,
        fast_dev_run=args.fast_dev_run,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        check_val_every_n_epoch=1,
        enable_checkpointing=True,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()

