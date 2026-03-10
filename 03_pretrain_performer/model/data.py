"""
LightningDataModule and Dataset for BulkFormer masked-MLM training.

Shared by both pre-training (ARCHS4) and fine-tuning (OSDR) scripts --
just point `expr_dir` at the appropriate processed parquet directory.
"""

from pathlib import Path

import numpy as np
import torch
import pandas as pd
import lightning as L
from torch.utils.data import DataLoader, Dataset


class BulkMLMDataset(Dataset):
    """Apply random gene masking for masked-language-model style training.

    Each call to __getitem__ draws a fresh random mask so every epoch
    sees different masked positions for the same sample.
    """

    MASK_TOKEN = -10

    def __init__(self, X_np: np.ndarray, mask_ratio: float = 0.15):
        self.X = X_np.astype(np.float32)
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        num_genes = x.shape[0]
        num_mask = int(num_genes * self.mask_ratio)
        mask_idx = np.random.choice(num_genes, num_mask, replace=False)

        x_masked = x.copy()
        x_masked[mask_idx] = self.MASK_TOKEN

        return (
            torch.from_numpy(x_masked),
            torch.from_numpy(x),
            torch.from_numpy(mask_idx.astype(np.int64)),
        )


class BulkExprDataModule(L.LightningDataModule):
    """Loads pre-processed expression parquets and serves train / val DataLoaders.

    Works for both ARCHS4 (pre-training) and OSDR (fine-tuning) data --
    just set ``expr_dir`` to the directory that contains
    ``train_expr_logtpm_short.parquet`` and ``val_expr_logtpm_short.parquet``.
    """

    def __init__(
        self,
        expr_dir: str,
        batch_size: int = 4,
        mask_ratio: float = 0.15,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.expr_dir = Path(expr_dir)
        self.batch_size = batch_size
        self.mask_ratio = mask_ratio
        self.num_workers = num_workers

        self._train_ds: BulkMLMDataset | None = None
        self._val_ds: BulkMLMDataset | None = None

    @property
    def num_genes(self) -> int:
        """Number of genes (available after setup)."""
        assert self._train_ds is not None, "Call setup() first"
        return self._train_ds.X.shape[1]

    def setup(self, stage: str | None = None):
        if self._train_ds is not None:
            return

        train_df = pd.read_parquet(
            self.expr_dir / "train_expr_logtpm_short.parquet"
        ).T
        val_df = pd.read_parquet(
            self.expr_dir / "val_expr_logtpm_short.parquet"
        ).T

        self._train_ds = BulkMLMDataset(train_df.values, self.mask_ratio)
        self._val_ds = BulkMLMDataset(val_df.values, self.mask_ratio)

    def train_dataloader(self):
        use_workers = self.num_workers > 0
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=use_workers,
        )

    def val_dataloader(self):
        use_workers = self.num_workers > 0
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=use_workers,
        )
