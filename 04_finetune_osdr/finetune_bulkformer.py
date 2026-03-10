#!/usr/bin/env python3
"""
finetune_bulkformer.py -- Fine-tune a pre-trained BulkFormer on OSDR data.

Powered by PyTorch Lightning.  All hyperparameters live in
``configs/finetune.yaml`` and can be overridden from the CLI::

    python finetune_bulkformer.py
    python finetune_bulkformer.py training.lr=5e-6 training.freeze_layers=1

Multi-GPU is handled automatically by Lightning under SLURM or when
multiple GPUs are visible.
"""

import json
import sys
from pathlib import Path

import torch
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from omegaconf import OmegaConf

# Import shared model + data modules from Phase 3
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "03_pretrain_performer"))
from model.lit_bulkformer import BulkFormerLit
from model.data import BulkExprDataModule


def _build_logger(log_cfg):
    """Return WandbLogger if available, else TensorBoardLogger."""
    try:
        from lightning.pytorch.loggers import WandbLogger

        return WandbLogger(
            project=log_cfg.project,
            name=log_cfg.get("run_name"),
            log_model=False,
            tags=["finetune", "osdr"],
        )
    except Exception:
        from lightning.pytorch.loggers import TensorBoardLogger

        return TensorBoardLogger(
            save_dir="./lightning_logs", name=log_cfg.project
        )


def _load_pretrained_weights(model: BulkFormerLit, ckpt_path: str):
    """Load pre-trained state dict, skipping layers with shape mismatches."""
    pretrained = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    current = model.model.state_dict()

    skipped = [
        k for k in pretrained
        if k in current and pretrained[k].shape != current[k].shape
    ]
    if skipped:
        print(f"  WARNING: Skipping {len(skipped)} layers with shape mismatch:")
        for k in skipped:
            print(f"    {k}: pretrained={pretrained[k].shape} vs model={current[k].shape}")
        for k in skipped:
            del pretrained[k]
        model.model.load_state_dict(pretrained, strict=False)
    else:
        model.model.load_state_dict(pretrained)

    print(f"  Loaded pre-trained weights from {ckpt_path}")


def main():
    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    cfg_path = Path(__file__).parent / "configs" / "finetune.yaml"
    cfg = OmegaConf.load(cfg_path)
    cli = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli)

    L.seed_everything(cfg.training.seed, workers=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_dir = Path(cfg.data.data_dir).resolve()
    osdr_dir = Path(cfg.data.osdr_dir) if cfg.data.osdr_dir else data_dir / "osdr" / "processed"

    if not osdr_dir.exists():
        print(f"ERROR: OSDR data not found at {osdr_dir}")
        print("Run prepare_osdr_data.py first.")
        sys.exit(1)

    datamodule = BulkExprDataModule(
        expr_dir=str(osdr_dir),
        batch_size=cfg.training.batch_size,
        mask_ratio=cfg.training.mask_ratio,
        num_workers=cfg.data.num_workers,
    )
    datamodule.setup()

    # ------------------------------------------------------------------
    # Auxiliary tensors
    # ------------------------------------------------------------------
    esm2_path = data_dir / "embeddings" / "esm2_t6_8M_UR50D_gene_embeddings.pt"
    esm2_data = torch.load(esm2_path, map_location="cpu", weights_only=True)
    gene_emb = esm2_data["embeddings"].float()

    edge_path = data_dir / "graph" / "edge_index_top20.pt"
    edge_index = torch.load(edge_path, map_location="cpu", weights_only=True).long()

    # ------------------------------------------------------------------
    # Model config -- prefer values from the pre-training checkpoint
    # ------------------------------------------------------------------
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    pt_config_path = Path(cfg.pretrained.config)
    if pt_config_path.exists():
        with open(pt_config_path) as f:
            pt_config = json.load(f)
        for key in ("dim", "gb_repeat", "bins", "bin_head", "full_head", "p_repeat"):
            if key in pt_config:
                model_cfg[key] = pt_config[key]
        print(f"  Loaded model config from {pt_config_path}")

    model_cfg.update({
        "graph": edge_index,
        "gene_emb": gene_emb,
        "gene_length": datamodule.num_genes,
    })

    optim_cfg = OmegaConf.to_container(cfg.training, resolve=True)
    sched_cfg = OmegaConf.to_container(cfg.scheduler, resolve=True)

    model = BulkFormerLit(
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        scheduler_cfg=sched_cfg,
        compile_model=cfg.training.get("compile", False),
        freeze_layers=cfg.training.get("freeze_layers", 0),
    )

    # Load pre-trained weights
    _load_pretrained_weights(model, cfg.pretrained.checkpoint)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Parameters: {total:,}  (trainable: {trainable:,})")

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    ckpt_cfg = cfg.checkpoint
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_cfg.dirpath,
            filename="epoch{epoch:02d}-val_loss{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=ckpt_cfg.save_top_k,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=cfg.training.patience,
            mode="min",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
    ]

    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="auto",
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        precision=cfg.training.precision,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        gradient_clip_val=cfg.training.gradient_clip_val,
        callbacks=callbacks,
        logger=_build_logger(cfg.logging),
        log_every_n_steps=cfg.logging.get("log_every_n_steps", 10),
        deterministic=True,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer.fit(model, datamodule=datamodule)

    print(f"\nFine-tuning complete!")
    print(f"  Best val loss : {trainer.callback_metrics.get('val/loss', 'N/A')}")
    print(f"  Checkpoints   : {ckpt_cfg.dirpath}/")


if __name__ == "__main__":
    main()
