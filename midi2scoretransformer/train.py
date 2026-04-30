#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from dataset import ASAPDataset
from config import MyModelConfig, FEATURES
from models.roformer import Roformer

try:
    from transformers import get_cosine_schedule_with_warmup
except Exception:
    get_cosine_schedule_with_warmup = None


def _apply_history_dropout(
    y: Dict[str, torch.Tensor],
    keep_prob: float,
    pad_key: str = "pad",
) -> Dict[str, torch.Tensor]:
    """
    Randomly drop (mask to 0) a portion of the *decoder input history* tokens
    (teacher forcing). This approximates "model only sees 25% of output history".

    y: bucketed one-hot streams, plus y["pad"] (B,T) long/bool.
    keep_prob: probability to keep a timestep token (for each timestep).
    """
    if keep_prob >= 1.0:
        return y

    # pad: (B,T) where 1 indicates real token, 0 padding
    pad = y[pad_key]
    if pad.dtype != torch.bool:
        pad_mask = pad > 0
    else:
        pad_mask = pad

    B, T = pad_mask.shape
    device = pad_mask.device

    # Sample keep mask per timestep; always keep t=0 to avoid empty history edge cases
    keep = (torch.rand((B, T), device=device) < keep_prob)
    keep[:, 0] = True

    # Do NOT keep anything outside real tokens
    keep = keep & pad_mask

    out = {}
    for k, v in y.items():
        if k == pad_key:
            out[k] = v
            continue
        # v: (B,T,C)
        v2 = v.clone()
        # where keep is False, zero out the one-hot (no teacher signal)
        v2[~keep] = 0
        out[k] = v2
    out[pad_key] = y[pad_key]
    return out


class TrainRoformer(Roformer):
    """
    Add training_step / configure_optimizers on top of the model (Roformer is a LightningModule already).
    """

    def __init__(
        self,
        enc_configuration: MyModelConfig,
        dec_configuration: MyModelConfig,
        hyperparameters: dict,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        pad_loss_weight: float = 0.1,
        teacher_keep_prob: float = 0.25,
        warmup_steps: int = 4000,
        max_steps: int = 40000,
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,
        freeze_embeddings_enc: bool = False,
        freeze_embeddings_dec: bool = False,
        freeze_unembeddings_dec: bool = False,
    ):
        super().__init__(enc_configuration, dec_configuration, hyperparameters)
        self.lr = lr
        self.weight_decay = weight_decay
        self.pad_loss_weight = pad_loss_weight
        self.teacher_keep_prob = teacher_keep_prob
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
        self.freeze_embeddings_enc = freeze_embeddings_enc
        self.freeze_embeddings_dec = freeze_embeddings_dec
        self.freeze_unembeddings_dec = freeze_unembeddings_dec

        # Pre-build loss weights/ignore_index map from FEATURES
        self._feat_loss_weight = {k: v["loss_weight"] for k, v in FEATURES.items()}
        self._feat_ignore_index = {k: v["ignore_index"] for k, v in FEATURES.items()}

        self._bce = nn.BCEWithLogitsLoss(reduction="none")

        self.apply_freezing()

    def apply_freezing(self) -> None:
        """Set requires_grad according to freeze_* flags (call again after checkpoint load if needed)."""
        def _freeze_module(mod: nn.Module, flag: bool) -> None:
            for p in mod.parameters():
                p.requires_grad = not flag

        _freeze_module(self.encoder, self.freeze_encoder)
        _freeze_module(self.decoder, self.freeze_decoder)
        _freeze_module(self.embeddings_enc, self.freeze_embeddings_enc)
        _freeze_module(self.embeddings_dec, self.freeze_embeddings_dec)
        _freeze_module(self.unembeddings_dec, self.freeze_unembeddings_dec)
        # Post-encoder LayerNorm: tie to encoder freezing (if encoder is frozen, freeze this too).
        _freeze_module(self.norm, self.freeze_encoder)

    @staticmethod
    def _ce_onehot(
        logits: torch.Tensor,         # (B,T,C)
        target_onehot: torch.Tensor,  # (B,T,C)
        pad_mask: torch.Tensor,       # (B,T) bool, True for real tokens
        ignore_index: int,
    ) -> torch.Tensor:
        """
        Cross entropy for one-hot targets, masked by pad.
        """
        target_idx = target_onehot.argmax(dim=-1).long()  # (B,T)

        B, T, C = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * T, C),
            target_idx.reshape(B * T),
            ignore_index=ignore_index,
            reduction="none",
        ).reshape(B, T)

        loss = loss * pad_mask.float()
        denom = pad_mask.float().sum().clamp_min(1.0)
        return loss.sum() / denom

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch  # both are dicts of bucketed tensors; y has "pad" (B,T)
        # pad mask for loss
        pad_mask = (y["pad"] > 0)

        # teacher-forcing history dropout (approx paper trick)
        y_in = _apply_history_dropout(y, keep_prob=self.teacher_keep_prob)

        y_hat = self.forward(x, y_in)  # dict: streams + "pad" logits

        total = 0.0
        OUT_FEATURE_KEYS = [k for k in FEATURES.keys() if k != "onset"]
        
        # stream classification losses
        for k in OUT_FEATURE_KEYS:
            w = self._feat_loss_weight.get(k, 0.0)
            if w <= 0:
                continue
            loss_k = self._ce_onehot(
                logits=y_hat[k],
                target_onehot=y[k],
                pad_mask=pad_mask,
                ignore_index=self._feat_ignore_index[k],
            )
            self.log(f"train/loss_{k}", loss_k, prog_bar=False, on_step=True, on_epoch=True)
            total = total + w * loss_k

        # pad head BCE loss: y_hat["pad"] (B,T,1), y["pad"] (B,T)
        pad_logits = y_hat["pad"].squeeze(-1)
        pad_target = y["pad"].float()
        pad_loss = self._bce(pad_logits, pad_target)
        pad_loss = (pad_loss * pad_mask.float()).sum() / pad_mask.float().sum().clamp_min(1.0)
        total = total + self.pad_loss_weight * pad_loss
        self.log("train/loss_pad", pad_loss, prog_bar=False, on_step=True, on_epoch=True)

        self.log("train/loss_total", total, prog_bar=True, on_step=True, on_epoch=True)
        return total

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        pad_mask = (y["pad"] > 0)
        y_hat = self.forward(x, y)  # val 不做 history dropout

        total = 0.0
        OUT_FEATURE_KEYS = [k for k in FEATURES.keys() if k != "onset"]
        for k in OUT_FEATURE_KEYS:
            w = self._feat_loss_weight.get(k, 0.0)
            if w <= 0:
                continue
            loss_k = self._ce_onehot(
                logits=y_hat[k],
                target_onehot=y[k],
                pad_mask=pad_mask,
                ignore_index=self._feat_ignore_index[k],
            )
            total = total + w * loss_k

        pad_logits = y_hat["pad"].squeeze(-1)
        pad_target = y["pad"].float()
        pad_loss = self._bce(pad_logits, pad_target)
        pad_loss = (pad_loss * pad_mask.float()).sum() / pad_mask.float().sum().clamp_min(1.0)
        total = total + self.pad_loss_weight * pad_loss

        self.log("val/loss_total", total, prog_bar=True, on_step=False, on_epoch=True)
        return total

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable parameters: check freeze_* flags.")
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

        if get_cosine_schedule_with_warmup is None:
            return opt

        sch = get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.max_steps,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "interval": "step"},
        }


def build_model(args) -> TrainRoformer:
    # Encoder config
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
        embedding_size=args.hidden_size,  # embedding.py uses config.embedding_size
        bias=True,
    )
    # Decoder config
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
    parser.add_argument("--run_name", type=str, default="pm2s_roformer")
    parser.add_argument("--seed", type=int, default=42)
    # logging
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging via WandbLogger.")
    parser.add_argument("--wandb_project", type=str, default="MIDI2ScoreTransformer")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default="", help="Comma-separated list of tags.")
    parser.add_argument("--wandb_mode", type=str, default=None, help="W&B mode: 'online', 'offline', or 'disabled'.")
    parser.add_argument("--wandb_log_model", type=str, default=None, help="WandbLogger log_model setting (e.g. 'all').")

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
    parser.add_argument("--intermediate_size", type=int, default=1536)
    parser.add_argument("--dropout", type=float, default=0.1)

    # loss tricks
    parser.add_argument("--pad_loss_weight", type=float, default=0.1)
    parser.add_argument("--teacher_keep_prob", type=float, default=0.25)

    # runtime/debug
    parser.add_argument("--gpu_id", type=int, default=0, help="physical GPU id (use CUDA_VISIBLE_DEVICES or set this)")
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--fast_dev_run", action="store_true", help="Lightning fast dev run (1 train/val batch).")
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    # output
    parser.add_argument("--out_dir", type=str, default="./runs", help="Root output directory for logs and checkpoints",)
    
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)

    # datasets
    train_set = ASAPDataset(
        data_dir=args.data_dir,
        split="train",
        seq_length=args.seq_length,
        cache=True,
        padding="per-beat",
        augmentations={},   # 你后面要开 transpose 等，再加
        return_continous=False,
    )
    val_set = ASAPDataset(
        data_dir=args.data_dir,
        split="validation",
        seq_length=args.seq_length,
        cache=True,
        padding="per-beat",
        augmentations={},
        return_continous=False,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,  # train split 已经在 __getitem__ 里按 lengths multinomial 采样
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

    model = build_model(args)

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