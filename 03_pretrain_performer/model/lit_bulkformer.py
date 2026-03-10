"""
LightningModule wrapper for BulkFormer.

Encapsulates all training logic: masked-MLM loss, AdamW + cosine-warmup
LR schedule, torchmetrics (Pearson R, R²), and optional torch.compile.
The underlying BulkFormer nn.Module remains pure PyTorch.
"""

import math

import torch
import torch.nn as nn
import lightning as L
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import PearsonCorrCoef, R2Score

from model.bulkformer import BulkFormer


class BulkFormerLit(L.LightningModule):
    """Lightning wrapper around BulkFormer for masked-MLM pre-training / fine-tuning."""

    def __init__(
        self,
        model_cfg: dict,
        optim_cfg: dict,
        scheduler_cfg: dict | None = None,
        compile_model: bool = False,
        freeze_layers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = BulkFormer(**model_cfg)

        if compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        if freeze_layers > 0:
            for i, layer in enumerate(self.model.gb_formers):
                if i < freeze_layers:
                    for p in layer.parameters():
                        p.requires_grad = False

        self.mse = nn.MSELoss()

        self.val_pearson = PearsonCorrCoef()
        self.val_r2 = R2Score()

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def _masked_mse(self, pred, x_true, mask_idx):
        """MSE loss computed only on masked gene positions."""
        losses = []
        for i in range(pred.size(0)):
            idx = mask_idx[i]
            losses.append(self.mse(pred[i, idx], x_true[i, idx]))
        return torch.stack(losses).mean()

    @staticmethod
    def _gather_masked(pred, x_true, mask_idx):
        """Collect all masked predictions / targets into flat tensors for metrics."""
        preds, trues = [], []
        for i in range(pred.size(0)):
            idx = mask_idx[i]
            preds.append(pred[i, idx])
            trues.append(x_true[i, idx])
        return torch.cat(preds), torch.cat(trues)

    # ------------------------------------------------------------------
    # Training / validation steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        x_masked, x_true, mask_idx = batch
        pred = self.model(x_masked)
        loss = self._masked_mse(pred, x_true, mask_idx)
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_masked, x_true, mask_idx = batch
        pred = self.model(x_masked)
        loss = self._masked_mse(pred, x_true, mask_idx)

        pred_flat, true_flat = self._gather_masked(pred, x_true, mask_idx)
        self.val_pearson.update(pred_flat, true_flat)
        self.val_r2.update(pred_flat, true_flat)

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        self.log("val/pearson_r", self.val_pearson.compute(), prog_bar=True)
        self.log("val/r2", self.val_r2.compute())
        self.val_pearson.reset()
        self.val_r2.reset()

    # ------------------------------------------------------------------
    # Optimizer + LR schedule
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optim_cfg = self.hparams.optim_cfg
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=optim_cfg["lr"],
            weight_decay=optim_cfg.get("weight_decay", 0.01),
        )

        sched_cfg = self.hparams.scheduler_cfg
        if sched_cfg is None:
            return optimizer

        warmup_steps = sched_cfg.get("warmup_steps", 0)
        total_steps = self.trainer.estimated_stepping_batches
        min_lr_ratio = sched_cfg.get("min_lr_ratio", 0.01)

        def _cosine_warmup(current_step: int) -> float:
            if current_step < warmup_steps:
                return current_step / max(warmup_steps, 1)
            progress = (current_step - warmup_steps) / max(
                total_steps - warmup_steps, 1
            )
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (
                1.0 + math.cos(math.pi * progress)
            )

        scheduler = LambdaLR(optimizer, lr_lambda=_cosine_warmup)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
