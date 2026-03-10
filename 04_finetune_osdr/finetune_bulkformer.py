#!/usr/bin/env python3
"""
finetune_bulkformer.py -- Fine-tune a pre-trained BulkFormer on OSDR data.

Loads a pre-trained BulkFormer checkpoint (from Phase 3) and fine-tunes it
on OSDR reprocessed gene expression data (from Phase 1 -> Phase 4 prep).

Key differences from pre-training:
  - Lower learning rate (1e-5 vs 1e-4)
  - Fewer epochs (10 vs 20)
  - Loads pre-trained weights before training
  - Uses OSDR data instead of ARCHS4

Usage:
    # Single GPU
    python finetune_bulkformer.py \
        --pretrained ../03_pretrain_performer/bulkformer_checkpoints/best_model.pt \
        --data-dir ../data

    # Multi-GPU
    torchrun --nproc_per_node=2 finetune_bulkformer.py \
        --pretrained ../03_pretrain_performer/bulkformer_checkpoints/best_model.pt \
        --data-dir ../data
"""

import argparse
import json
import os
import sys
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

# Import model from Phase 3
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "03_pretrain_performer"))
from model.bulkformer import BulkFormer, model_params

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BulkFormer on OSDR data.")
    parser.add_argument("--pretrained", required=True, help="Path to pre-trained checkpoint (.pt)")
    parser.add_argument("--data-dir", default="../data", help="Root data directory")
    parser.add_argument("--osdr-data-dir", default=None, help="OSDR processed data dir (default: <data-dir>/osdr/processed)")
    parser.add_argument("--epochs", type=int, default=10, help="Max fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (lower than pre-training)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--mask-ratio", type=float, default=0.15, help="Gene masking ratio")
    parser.add_argument("--checkpoint-dir", default="./finetune_checkpoints", help="Output checkpoint dir")
    parser.add_argument("--freeze-layers", type=int, default=0, help="Number of GBFormer layers to freeze (0=none)")
    return parser.parse_args()


class BulkMLMDataset(Dataset):
    """Dataset with random gene masking for MLM-style training."""

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
    """Initialize DDP if running with torchrun."""
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
        print(f"BulkFormer Fine-tuning on OSDR Data")
        print(f"  {'DDP with ' + str(world_size) + ' GPUs' if use_ddp else 'Single device'}")
        print("=" * 70)
        print(f"  Pre-trained model: {args.pretrained}")
        print(f"  Device: {device}")
        print(f"  LR: {args.lr}, Epochs: {args.epochs}, Batch: {args.batch_size}")
        if args.freeze_layers > 0:
            print(f"  Freezing first {args.freeze_layers} GBFormer layer(s)")

    data_dir = Path(args.data_dir).resolve()
    osdr_dir = Path(args.osdr_data_dir) if args.osdr_data_dir else data_dir / "osdr" / "processed"
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Load OSDR fine-tuning data
    # ------------------------------------------------------------------
    if is_main:
        print("\n[DATA] Loading OSDR expression data ...")

    if not osdr_dir.exists():
        if is_main:
            print(f"ERROR: OSDR data not found at {osdr_dir}")
            print("Run prepare_osdr_data.py first.")
        sys.exit(1)

    t0 = time.time()
    X_df = pd.read_parquet(osdr_dir / "train_expr_logtpm_short.parquet").T
    X_np = X_df.values
    num_samples, num_genes = X_np.shape
    if is_main:
        print(f"  Train: {X_np.shape} ({time.time()-t0:.1f}s)")

    X_val_df = pd.read_parquet(osdr_dir / "val_expr_logtpm_short.parquet").T
    X_val_np = X_val_df.values
    if is_main:
        print(f"  Val:   {X_val_np.shape}")

    # ------------------------------------------------------------------
    # Load ESM2 embeddings + graph (same as pre-training)
    # ------------------------------------------------------------------
    if is_main:
        print("\n[DATA] Loading gene embeddings + graph ...")

    esm2_path = data_dir / "embeddings" / "esm2_t6_8M_UR50D_gene_embeddings.pt"
    esm2_data = torch.load(esm2_path, map_location="cpu")
    esm2_raw = esm2_data["embeddings"].float().to(device)

    edge_path = data_dir / "graph" / "edge_index_top20.pt"
    edge_index = torch.load(edge_path, map_location="cpu").long().to(device)

    if is_main:
        print(f"  ESM2: {esm2_raw.shape}, Graph: {edge_index.shape}")

    # ------------------------------------------------------------------
    # Build model and load pre-trained weights
    # ------------------------------------------------------------------
    if is_main:
        print("\n[MODEL] Building BulkFormer and loading pre-trained weights ...")

    model_params["graph"] = edge_index
    model_params["gene_emb"] = esm2_raw
    model_params["gene_length"] = num_genes

    # Load pre-training config if available
    pretrained_config_path = Path(args.pretrained).parent / "config.json"
    if pretrained_config_path.exists():
        with open(pretrained_config_path) as f:
            pt_config = json.load(f)
        for key in ["dim", "gb_repeat", "bins", "bin_head", "full_head", "p_repeat"]:
            if key in pt_config:
                model_params[key] = pt_config[key]
        if is_main:
            print(f"  Loaded config from {pretrained_config_path}")

    model = BulkFormer(**model_params).to(device)

    # Load pre-trained state dict
    pretrained_state = torch.load(args.pretrained, map_location=device)

    # Handle potential gene_length mismatch in ae_enc and head layers
    model_state = model.state_dict()
    skipped = []
    for key in pretrained_state:
        if key in model_state and pretrained_state[key].shape != model_state[key].shape:
            skipped.append(key)

    if skipped:
        if is_main:
            print(f"  WARNING: Skipping {len(skipped)} layers with shape mismatch:")
            for k in skipped:
                print(f"    {k}: pretrained={pretrained_state[k].shape} vs model={model_state[k].shape}")
        for k in skipped:
            del pretrained_state[k]
        model.load_state_dict(pretrained_state, strict=False)
    else:
        model.load_state_dict(pretrained_state)

    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total_params:,}")

    # Optionally freeze early GBFormer layers
    if args.freeze_layers > 0:
        for i, layer in enumerate(model.gb_formers):
            if i < args.freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if is_main:
            print(f"  Trainable parameters (after freezing): {trainable:,}")

    if use_ddp:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

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
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    mse_loss = nn.MSELoss()

    # ------------------------------------------------------------------
    # Fine-tuning loop
    # ------------------------------------------------------------------
    if is_main:
        print(f"\n[TRAIN] Starting fine-tuning ...\n{'='*70}")

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

        epoch_avg = running_loss / max(num_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for x_masked, x_true, mask_idx in val_loader:
                x_masked = x_masked.to(device)
                x_true = x_true.to(device)
                pred = model(x_masked)
                batch_losses = []
                for i in range(len(mask_idx)):
                    idxs = mask_idx[i]
                    batch_losses.append(mse_loss(pred[i, idxs], x_true[i, idxs]))
                val_loss += torch.stack(batch_losses).mean().item()
                val_batches += 1

        if use_ddp:
            vl_t = torch.tensor(val_loss, device=device)
            vb_t = torch.tensor(val_batches, device=device)
            dist.all_reduce(vl_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(vb_t, op=dist.ReduceOp.SUM)
            val_avg = (vl_t / vb_t).item()
        else:
            val_avg = val_loss / max(val_batches, 1)

        train_losses.append(epoch_avg)
        val_losses.append(val_avg)
        epoch_time = time.time() - epoch_start
        model_to_save = model.module if use_ddp else model

        if is_main:
            print(f"\n  Epoch {epoch+1}/{args.epochs} | Train: {epoch_avg:.6f} | Val: {val_avg:.6f} | Time: {epoch_time:.1f}s")
            torch.save(model_to_save.state_dict(), ckpt_dir / f"epoch_{epoch}.pt")

            if val_avg < best_val_loss:
                best_val_loss = val_avg
                patience_counter = 0
                torch.save(model_to_save.state_dict(), ckpt_dir / "best_model.pt")
                print(f"    -> New best val loss! Saved best_model.pt")
            else:
                patience_counter += 1
                print(f"    -> No improvement ({patience_counter}/{args.patience})")
                if patience_counter >= args.patience:
                    print(f"    -> Early stopping.\n")
                    break
            print()

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    if is_main:
        print("[SAVE] Saving fine-tune config + loss history ...")

        config = {
            "model_type": "bulkformer_finetuned",
            "pretrained_from": str(args.pretrained),
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
            "freeze_layers": args.freeze_layers,
            "osdr_train_samples": len(train_dataset),
            "osdr_val_samples": len(val_dataset),
        }
        with open(ckpt_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        loss_df = pd.DataFrame({
            "epoch": range(len(train_losses)),
            "train_loss": train_losses,
            "val_loss": val_losses,
        })
        loss_df.to_csv(ckpt_dir / "loss_history.csv", index=False)

        if HAS_MATPLOTLIB:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, marker="o", label="Train Loss", linewidth=2)
            plt.plot(val_losses, marker="s", label="Val Loss", linewidth=2)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("BulkFormer Fine-tuning on OSDR")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(ckpt_dir / "loss_plot.png", dpi=150)
            plt.close()

        total_time = time.time() - script_start
        print(f"\n{'='*70}")
        print(f"Fine-tuning complete!")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"  Best val loss: {best_val_loss:.6f}")
        print(f"  Checkpoints: {ckpt_dir}/")
        print(f"{'='*70}\n")

    cleanup_ddp(use_ddp)


if __name__ == "__main__":
    main()
