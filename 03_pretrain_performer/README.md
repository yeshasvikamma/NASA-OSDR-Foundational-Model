# Phase 3: Pre-train BulkFormer on ARCHS4

Pre-train the BulkFormer (Performer-based) model on ARCHS4 human + mouse RNA-seq data using masked language modeling (MLM), powered by PyTorch Lightning.

## What This Phase Does

1. Loads preprocessed ARCHS4 expression data from Phase 2 (parquet files)
2. Loads ESM2 gene identity embeddings and KNN gene-gene graph
3. Trains BulkFormer using MLM:
   - Randomly masks 15% of genes per sample
   - Model learns to reconstruct masked expression values
   - Loss: MSE only on masked positions
   - Metrics: Pearson R and R-squared on masked positions (via torchmetrics)
4. Saves top-k checkpoints by validation loss + last checkpoint
5. Early stopping with configurable patience
6. Logs to Weights & Biases (falls back to TensorBoard)

## Training Infrastructure

| Feature | Implementation |
|---------|---------------|
| Distributed training | Lightning DDP (auto-detected under SLURM) |
| Mixed precision | FP16 via `precision: "16-mixed"` (~2x throughput) |
| Gradient accumulation | 8x (effective batch = `batch_size * 8`) |
| LR schedule | Cosine decay with linear warmup |
| Gradient clipping | Max norm 1.0 |
| Experiment tracking | Weights & Biases (or TensorBoard fallback) |
| torch.compile | Enabled by default (PyTorch 2.0+) |

## Prerequisites

- Phase 2 must be completed (data in `../data/archs4/processed/`)
- `biofm` conda environment installed
- CUDA GPUs available (2+ recommended for DDP)

## How to Run

### Submit as SLURM job (recommended)

Edit `pretrain.slurm` to set your GPU partition and resource limits, then:

```bash
sbatch pretrain.slurm
```

### Run interactively

```bash
conda activate biofm

# Single GPU (Lightning auto-detects)
python pretrain_bulkformer.py

# Override any config value from the CLI
python pretrain_bulkformer.py training.lr=5e-5 training.batch_size=8

# Disable torch.compile if needed
python pretrain_bulkformer.py training.compile=false
```

### Resume from Lightning checkpoint

```bash
# Lightning automatically saves last.ckpt -- the Trainer can resume from it
# by modifying the script or using the saved checkpoints
```

## Configuration

All hyperparameters live in `configs/pretrain.yaml`. Override any value from the CLI using dot notation:

```bash
python pretrain_bulkformer.py training.lr=5e-5 training.max_epochs=30 data.num_workers=8
```

Key settings:

| Config Key | Default | Description |
|------------|---------|-------------|
| `model.dim` | `320` | Embedding dimension |
| `model.gb_repeat` | `1` | Number of GBFormer encoder blocks |
| `model.bins` | `10` | Number of expression bins |
| `training.lr` | `1e-4` | Peak learning rate |
| `training.batch_size` | `4` | Per-GPU batch size |
| `training.max_epochs` | `20` | Maximum training epochs |
| `training.patience` | `3` | Early stopping patience |
| `training.mask_ratio` | `0.15` | Fraction of genes to mask |
| `training.precision` | `"16-mixed"` | Mixed precision mode |
| `training.accumulate_grad_batches` | `8` | Gradient accumulation steps |
| `training.compile` | `true` | Enable torch.compile |
| `scheduler.warmup_steps` | `500` | Linear warmup steps |
| `checkpoint.dirpath` | `./bulkformer_checkpoints` | Checkpoint directory |
| `logging.project` | `"nasa-osdr-bulkformer"` | W&B project name |

## Weights & Biases Setup

W&B provides real-time experiment tracking dashboards. It is optional -- if not configured, logging falls back to TensorBoard.

```bash
pip install wandb       # included in biofm.yml
wandb login             # one-time setup, paste your API key
```

Free for academics and individual researchers at [wandb.ai](https://wandb.ai).

## Output

After training, `bulkformer_checkpoints/` will contain:

```
bulkformer_checkpoints/
├── epoch00-val_loss0.XXXX.ckpt    # Top-k checkpoints by val loss
├── epoch01-val_loss0.XXXX.ckpt
├── ...
├── last.ckpt                       # Most recent checkpoint
└── lightning_logs/ or wandb/       # Experiment logs
```

## Code Organization

```
03_pretrain_performer/
├── model/
│   ├── bulkformer.py          # Pure PyTorch model (unchanged)
│   ├── lit_bulkformer.py      # LightningModule (training logic, metrics, optimizer)
│   └── data.py                # LightningDataModule + BulkMLMDataset
├── configs/
│   └── pretrain.yaml          # All hyperparameters
├── pretrain_bulkformer.py     # Lightning Trainer entry point
└── pretrain.slurm             # SLURM job script
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
