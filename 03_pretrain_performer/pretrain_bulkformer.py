#!/usr/bin/env python3
"""
pretrain_bulkformer.py -- Pre-train BulkFormer on ARCHS4 data using MLM.

Masked Language Modeling (MLM) for gene expression:
  - Randomly mask 15% of genes per sample
  - Train model to reconstruct the masked expression values
  - Loss: MSE only on masked positions

Supports multi-GPU training via PyTorch DDP (DistributedDataParallel).

Usage:
    # Single GPU
    python pretrain_bulkformer.py --data-dir ../data

    # Multi-GPU (2 GPUs)
    torchrun --nproc_per_node=2 pretrain_bulkformer.py --data-dir ../data

Adapted from: https://github.com/alwalt/BioFM (4_bulkformer_run.py)
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from model.bulkformer import BulkFormer, model_params

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train BulkFormer on ARCHS4 data.")
    parser.add_argument("--data-dir", default="../data", help="Root data directory")
    parser.add_argument("--epochs", type=int, default=20, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--mask-ratio", type=float, default=0.15, help="Gene masking ratio")
    parser.add_argument("--checkpoint-dir", default="./bulkformer_checkpoints", help="Checkpoint output dir")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


class BulkMLMDataset(Dataset):
    """Dataset that applies random gene masking for MLM-style pre-training."""

    def __init__(self, X_np, mask_ratio=0.15, mask_token=-10):
        self.X = X_np.astype(np.float32)
        self.mask_ratio = mask_ratio
        self.mask_token = mask_token

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        g = x.shape[0]
        num_mask = int(g * self.mask_ratio)
        mask_idx = np.random.choice(g, num_mask, replace=False)
        x_masked = x.copy()
        x_masked[mask_idx] = self.mask_token
        return (
            torch.tensor(x_masked),
            torch.tensor(x),
            torch.tensor(mask_idx),
        )


def setup_ddp():
    """Initialize DDP if running with torchrun, else use single GPU."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        return rank, world_size, device, True
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, device, False


def cleanup_ddp(use_ddp):
    if use_ddp:
        dist.destroy_process_group()


def main():
    args = parse_args()
    script_start = time.time()

    rank, world_size, device, use_ddp = setup_ddp()
    is_main = rank == 0

    if is_main:
        print("\n" + "=" * 70)
        print(f"BulkFormer Pre-training {'(DDP, ' + str(world_size) + ' GPUs)' if use_ddp else '(single device)'}")
        print("=" * 70)
        print(f"  Device: {device}")
        print(f"  Data dir: {args.data_dir}")
        print(f"  Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
        print(f"  Mask ratio: {args.mask_ratio}, Patience: {args.patience}")

    data_dir = Path(args.data_dir).resolve()
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Load training data
    # ------------------------------------------------------------------
    if is_main:
        print("\n[DATA] Loading expression data ...")

    processed_dir = data_dir / "archs4" / "processed"
    if not processed_dir.exists():
        # Fallback to BioFM-style path
        processed_dir = data_dir / "archs4" / "processed_short_proteins"

    t0 = time.time()
    X_df = pd.read_parquet(processed_dir / "train_expr_logtpm_short.parquet").T
    X_np = X_df.values
    num_samples, num_genes = X_np.shape
    if is_main:
        print(f"  Train: {X_np.shape} ({time.time() - t0:.1f}s)")

    X_val_df = pd.read_parquet(processed_dir / "val_expr_logtpm_short.parquet").T
    X_val_np = X_val_df.values
    if is_main:
        print(f"  Val: {X_val_np.shape}")

    # ------------------------------------------------------------------
    # Load ESM2 gene identity embeddings
    # ------------------------------------------------------------------
    if is_main:
        print("\n[DATA] Loading ESM2 gene embeddings ...")
    esm2_path = data_dir / "embeddings" / "esm2_t6_8M_UR50D_gene_embeddings.pt"
    esm2_data = torch.load(esm2_path, map_location="cpu")
    esm2_raw = esm2_data["embeddings"].float().to(device)
    if is_main:
        print(f"  Shape: {esm2_raw.shape}")

    # ------------------------------------------------------------------
    # Load gene-gene graph
    # ------------------------------------------------------------------
    if is_main:
        print("\n[DATA] Loading gene graph edges ...")
    edge_path = data_dir / "graph" / "edge_index_top20.pt"
    edge_index = torch.load(edge_path, map_location="cpu").long().to(device)
    if is_main:
        print(f"  Shape: {edge_index.shape}")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    if is_main:
        print("\n[MODEL] Initializing BulkFormer ...")
    model_params["graph"] = edge_index
    model_params["gene_emb"] = esm2_raw
    model_params["gene_length"] = num_genes

    model = BulkFormer(**model_params).to(device)

    if args.resume:
        if is_main:
            print(f"  Resuming from checkpoint: {args.resume}")
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state)

    if use_ddp:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    total_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"  Parameters: {total_params:,}")
        print(f"\n  Model config:")
        for k, v in model_params.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: tensor{tuple(v.shape)}")
            else:
                print(f"    {k}: {v}")

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------
    train_dataset = BulkMLMDataset(X_np, mask_ratio=args.mask_ratio)
    val_dataset = BulkMLMDataset(X_val_np, mask_ratio=args.mask_ratio)

    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, shuffle=(train_sampler is None))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, shuffle=False)

    if is_main:
        print(f"\n[DATA] Train: {len(train_dataset)} samples, {len(train_loader)} batches")
        print(f"[DATA] Val:   {len(val_dataset)} samples, {len(val_loader)} batches")

    # ------------------------------------------------------------------
    # Optimizer & loss
    # ------------------------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=args.lr)
    mse_loss = nn.MSELoss()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    if is_main:
        print("\n[TRAIN] Starting training ...\n" + "=" * 70)

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        if use_ddp:
            train_sampler.set_epoch(epoch)

        running_loss = 0.0
        num_batches = 0

        for batch_idx, (x_masked, x_true, mask_idx) in enumerate(train_loader):
            x_masked = x_masked.to(device)
            x_true = x_true.to(device)

            pred = model(x_masked)

            loss_list = []
            for i in range(len(mask_idx)):
                idxs = mask_idx[i]
                loss_list.append(mse_loss(pred[i, idxs], x_true[i, idxs]))
            loss = torch.stack(loss_list).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            if is_main and (batch_idx + 1) % max(1, len(train_loader) // 4) == 0:
                avg = running_loss / num_batches
                print(f"  Epoch {epoch+1}/{args.epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.6f} | Avg: {avg:.6f}")

        epoch_avg_loss = running_loss / max(num_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for x_masked, x_true, mask_idx in val_loader:
                x_masked = x_masked.to(device)
                x_true = x_true.to(device)
                pred = model(x_masked)
                loss_list = []
                for i in range(len(mask_idx)):
                    idxs = mask_idx[i]
                    loss_list.append(mse_loss(pred[i, idxs], x_true[i, idxs]))
                val_loss += torch.stack(loss_list).mean().item()
                val_batches += 1

        if use_ddp:
            val_loss_t = torch.tensor(val_loss, device=device)
            val_batches_t = torch.tensor(val_batches, device=device)
            dist.all_reduce(val_loss_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_batches_t, op=dist.ReduceOp.SUM)
            val_avg_loss = (val_loss_t / val_batches_t).item()
        else:
            val_avg_loss = val_loss / max(val_batches, 1)

        train_losses.append(epoch_avg_loss)
        val_losses.append(val_avg_loss)
        epoch_time = time.time() - epoch_start

        model_to_save = model.module if use_ddp else model

        if is_main:
            print(f"\n  Epoch {epoch+1}/{args.epochs} | Train: {epoch_avg_loss:.6f} | Val: {val_avg_loss:.6f} | Time: {epoch_time:.1f}s")

            torch.save(model_to_save.state_dict(), ckpt_dir / f"epoch_{epoch}.pt")

            if val_avg_loss < best_val_loss:
                best_val_loss = val_avg_loss
                patience_counter = 0
                torch.save(model_to_save.state_dict(), ckpt_dir / "best_model.pt")
                print(f"    -> New best val loss! Saved best_model.pt")
            else:
                patience_counter += 1
                print(f"    -> No improvement ({patience_counter}/{args.patience})")
                if patience_counter >= args.patience:
                    print(f"    -> Early stopping triggered.\n")
                    break
            print()

    # ------------------------------------------------------------------
    # Save final artifacts
    # ------------------------------------------------------------------
    if is_main:
        print("[SAVE] Saving config + loss history ...")

        config = {
            "model_type": "bulkformer",
            "num_genes": num_genes,
            "dim": model_params.get("dim", 320),
            "gb_repeat": model_params.get("gb_repeat", 1),
            "bins": model_params.get("bins", 10),
            "bin_head": model_params.get("bin_head", 8),
            "full_head": model_params.get("full_head", 4),
            "p_repeat": model_params.get("p_repeat", 2),
            "final_epoch": epoch,
            "best_val_loss": best_val_loss,
            "early_stopped": patience_counter >= args.patience,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "mask_ratio": args.mask_ratio,
        }
        with open(ckpt_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        loss_df = pd.DataFrame(
            {"epoch": range(len(train_losses)), "train_loss": train_losses, "val_loss": val_losses}
        )
        loss_df.to_csv(ckpt_dir / "loss_history.csv", index=False)

        if HAS_MATPLOTLIB:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, marker="o", label="Train Loss", linewidth=2)
            plt.plot(val_losses, marker="s", label="Val Loss", linewidth=2)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("BulkFormer Pre-training Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(ckpt_dir / "loss_plot.png", dpi=150)
            plt.close()
            print(f"  Loss plot saved to {ckpt_dir / 'loss_plot.png'}")

        total_time = time.time() - script_start
        print(f"\n{'='*70}")
        print(f"Pre-training complete!")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"  Best val loss: {best_val_loss:.6f}")
        print(f"  Checkpoints: {ckpt_dir}/")
        print(f"{'='*70}\n")

    cleanup_ddp(use_ddp)


if __name__ == "__main__":
    main()
