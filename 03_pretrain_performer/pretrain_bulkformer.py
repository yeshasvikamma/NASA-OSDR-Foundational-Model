#!/usr/bin/env python3
"""
pretrain_bulkformer.py -- Pre-train BulkFormer on ARCHS4 data using MLM.

Powered by PyTorch Lightning.  All hyperparameters live in
``configs/pretrain.yaml`` and can be overridden from the CLI::

    # Single GPU
    python pretrain_bulkformer.py

    # Override any config value
    python pretrain_bulkformer.py training.lr=5e-5 training.batch_size=8

    # Multi-GPU is handled automatically by Lightning when launched via
    # SLURM (srun) or when multiple GPUs are visible.

Adapted from: https://github.com/alwalt/BioFM
"""

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

from model.lit_bulkformer import BulkFormerLit
from model.data import BulkExprDataModule


def _build_logger(log_cfg):
    """Return WandbLogger if wandb is installed, else TensorBoardLogger."""
    try:
        from lightning.pytorch.loggers import WandbLogger

        return WandbLogger(
            project=log_cfg.project,
            name=log_cfg.get("run_name"),
            log_model=False,
        )
    except Exception:
        from lightning.pytorch.loggers import TensorBoardLogger

        return TensorBoardLogger(
            save_dir="./lightning_logs", name=log_cfg.project
        )


def main():
    # ------------------------------------------------------------------
    # Config: load YAML then merge CLI overrides
    # ------------------------------------------------------------------
    cfg_path = Path(__file__).parent / "configs" / "pretrain.yaml"
    cfg = OmegaConf.load(cfg_path)
    cli = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli)

    L.seed_everything(cfg.training.seed, workers=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_dir = Path(cfg.data.data_dir).resolve()
    expr_dir = data_dir / "archs4" / "processed"
    if not expr_dir.exists():
        expr_dir = data_dir / "archs4" / "processed_short_proteins"

    datamodule = BulkExprDataModule(
        expr_dir=str(expr_dir),
        batch_size=cfg.training.batch_size,
        mask_ratio=cfg.training.mask_ratio,
        num_workers=cfg.data.num_workers,
    )
    datamodule.setup()

    # ------------------------------------------------------------------
    # Load auxiliary tensors (ESM2 embeddings + gene graph)
    # ------------------------------------------------------------------
    esm2_path = data_dir / "embeddings" / "esm2_t6_8M_UR50D_gene_embeddings.pt"
    esm2_data = torch.load(esm2_path, map_location="cpu", weights_only=True)
    gene_emb = esm2_data["embeddings"].float()

    edge_path = data_dir / "graph" / "edge_index_top20.pt"
    edge_index = torch.load(edge_path, map_location="cpu", weights_only=True).long()

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_cfg = {
        **OmegaConf.to_container(cfg.model, resolve=True),
        "graph": edge_index,
        "gene_emb": gene_emb,
        "gene_length": datamodule.num_genes,
    }
    optim_cfg = OmegaConf.to_container(cfg.training, resolve=True)
    sched_cfg = OmegaConf.to_container(cfg.scheduler, resolve=True)

    model = BulkFormerLit(
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        scheduler_cfg=sched_cfg,
        compile_model=cfg.training.get("compile", False),
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Parameters: {total_params:,}")

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

    print(f"\nPre-training complete!")
    print(f"  Best val loss : {trainer.callback_metrics.get('val/loss', 'N/A')}")
    print(f"  Checkpoints   : {ckpt_cfg.dirpath}/")


if __name__ == "__main__":
    main()
