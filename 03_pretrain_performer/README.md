# Phase 3: Pre-train BulkFormer on ARCHS4

Pre-train the BulkFormer (Performer-based) model on ARCHS4 human + mouse RNA-seq data using masked language modeling (MLM).

## What This Phase Does

1. Loads preprocessed ARCHS4 expression data from Phase 2 (parquet files)
2. Loads ESM2 gene identity embeddings and KNN gene-gene graph
3. Trains BulkFormer using MLM:
   - Randomly masks 15% of genes per sample
   - Model learns to reconstruct masked expression values
   - Loss: MSE only on masked positions
4. Saves checkpoints per epoch + best model (by validation loss)
5. Early stopping with configurable patience

## Prerequisites

- Phase 2 must be completed (data in `../data/archs4/processed/`)
- `biofm` conda environment installed
- CUDA GPUs available (2+ recommended for DDP)

## How to Run

### Submit as SLURM job (recommended)

First, edit `pretrain.slurm` to set your GPU partition and resource limits:

```bash
#SBATCH --partition=gpu        # Your GPU partition
#SBATCH --gres=gpu:2           # Number/type of GPUs (e.g. gpu:a100:2)
#SBATCH --mail-user=you@email  # Your email
```

Then submit:

```bash
sbatch pretrain.slurm
```

### Run interactively

```bash
conda activate biofm

# Single GPU
python pretrain_bulkformer.py --data-dir ../data

# Multi-GPU (2 GPUs)
torchrun --nproc_per_node=2 pretrain_bulkformer.py --data-dir ../data
```

### Resume from checkpoint

```bash
python pretrain_bulkformer.py --data-dir ../data --resume ./bulkformer_checkpoints/epoch_5.pt
```

## Command-Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `../data` | Root data directory |
| `--epochs` | `20` | Maximum training epochs |
| `--batch-size` | `4` | Per-GPU batch size |
| `--lr` | `1e-4` | Learning rate (AdamW) |
| `--patience` | `3` | Early stopping patience |
| `--mask-ratio` | `0.15` | Fraction of genes to mask |
| `--checkpoint-dir` | `./bulkformer_checkpoints` | Where to save checkpoints |
| `--resume` | `None` | Checkpoint path to resume from |

## Output

After training, `bulkformer_checkpoints/` will contain:

```
bulkformer_checkpoints/
├── epoch_0.pt          # Checkpoint for each epoch
├── epoch_1.pt
├── ...
├── best_model.pt       # Best model (lowest validation loss)
├── config.json         # Model configuration + training metadata
├── loss_history.csv    # Per-epoch train/val loss
└── loss_plot.png       # Training curves visualization
```

## Model Architecture

```
Input: [B, G] expression vector
  |
  +-- PositionalExprEmbedding (REE) -- sinusoidal encoding of expression values
  +-- ESM2 projection            -- protein language model gene identities
  +-- AutoEncoder sample embedding -- global transcriptome context
  |
  v
x_proj MLP
  |
  v
N x GBFormer blocks:
  ├── LayerNorm
  ├── GCNConv (gene-gene graph)
  ├── Learned binning (sort by expression regime)
  ├── Local Performer attention (per-bin)
  └── Global Performer attention (all genes)
  |
  v
LayerNorm -> Prediction Head -> [B, G] reconstructed expression
```

Default hyperparameters (from BioFM):
- `dim=320` (embedding dimension)
- `gb_repeat=1` (number of GBFormer blocks)
- `p_repeat=2` (global Performer layers per block)
- `bins=10` (expression bins)
- `bin_head=8`, `full_head=4` (attention heads)
