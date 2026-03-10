# Phase 4: Fine-tune BulkFormer on OSDR Data

Fine-tune the ARCHS4-pretrained BulkFormer model on OSDR reprocessed spaceflight RNA-seq data, powered by PyTorch Lightning.

## What This Phase Does

1. **`prepare_osdr_data.py`** -- Collects all `ens_gene_biot_abundance.tsv` files from Phase 1 Kallisto output, maps genes to the canonical set, normalizes, and saves as parquet
2. **`finetune_bulkformer.py`** -- Loads the best pre-trained checkpoint from Phase 3, then fine-tunes on the OSDR data with MLM using Lightning

## Prerequisites

- Phase 1 completed (Kallisto output exists under `../01_kallisto_reprocessing/`)
- Phase 3 completed (pre-trained checkpoint at `../03_pretrain_performer/bulkformer_checkpoints/best_model.pt`)
- Gene metadata in `../data/ensembl/` (from Phase 2 download)
- `biofm` conda environment installed

## How to Run

### Option A: Single SLURM job (recommended)

The SLURM script runs both data preparation and fine-tuning:

```bash
# Edit finetune.slurm for your HPC settings first
sbatch finetune.slurm
```

### Option B: Run steps separately

```bash
conda activate biofm

# Step 1: Convert Kallisto output to model format
python prepare_osdr_data.py \
    --kallisto-dir ../01_kallisto_reprocessing \
    --data-dir ../data

# Step 2: Fine-tune (Lightning auto-detects GPUs)
python finetune_bulkformer.py

# Override any config from the CLI
python finetune_bulkformer.py training.lr=5e-6 training.freeze_layers=1
```

## Configuration

All hyperparameters live in `configs/finetune.yaml`. Override any value from the CLI using dot notation:

```bash
python finetune_bulkformer.py training.lr=5e-6 training.freeze_layers=1 training.max_epochs=20
```

### prepare_osdr_data.py

| Flag | Default | Description |
|------|---------|-------------|
| `--kallisto-dir` | `../01_kallisto_reprocessing` | Kallisto output directory |
| `--data-dir` | `../data` | Root data directory |
| `--output-dir` | `<data-dir>/osdr/processed` | Where to save processed OSDR parquets |
| `--val-frac` | `0.2` | Validation fraction |

### finetune_bulkformer.py (YAML config)

| Config Key | Default | Description |
|------------|---------|-------------|
| `training.lr` | `1e-5` | Learning rate (10x lower than pre-training) |
| `training.max_epochs` | `10` | Max fine-tuning epochs |
| `training.patience` | `5` | Early stopping patience |
| `training.freeze_layers` | `0` | Number of GBFormer blocks to freeze |
| `training.batch_size` | `4` | Per-GPU batch size |
| `training.precision` | `"16-mixed"` | Mixed precision mode |
| `training.accumulate_grad_batches` | `8` | Gradient accumulation steps |
| `pretrained.checkpoint` | `../03_.../best_model.pt` | Path to pre-trained weights |
| `data.osdr_dir` | `null` (auto) | OSDR processed data directory |
| `checkpoint.dirpath` | `./finetune_checkpoints` | Checkpoint directory |
| `logging.project` | `"nasa-osdr-bulkformer"` | W&B project name |

## Fine-tuning Strategy

- **Learning rate**: 1e-5 (10x lower than pre-training's 1e-4) to avoid catastrophic forgetting
- **LR schedule**: Cosine decay with 200-step linear warmup
- **Epochs**: 10 max with early stopping (patience=5)
- **Mixed precision**: FP16 for ~2x throughput
- **Gradient accumulation**: 8x for effective batch size of 32
- **Layer freezing** (optional): Use `training.freeze_layers=N` to freeze the first N GBFormer blocks, training only the later layers and prediction head
- **Shape mismatch handling**: If the OSDR gene count differs from pre-training, layers with incompatible shapes are re-initialized automatically
- **Metrics**: Pearson R and R-squared on masked gene positions (via torchmetrics)

## Output

```
finetune_checkpoints/
├── epoch00-val_loss0.XXXX.ckpt    # Top-k checkpoints by val loss
├── epoch01-val_loss0.XXXX.ckpt
├── ...
├── last.ckpt                       # Most recent checkpoint
└── lightning_logs/ or wandb/       # Experiment logs
```

## Code Organization

```
04_finetune_osdr/
├── configs/
│   └── finetune.yaml          # All hyperparameters
├── prepare_osdr_data.py       # Kallisto -> model input converter
├── finetune_bulkformer.py     # Lightning Trainer entry point
└── finetune.slurm             # SLURM job script

Imports from Phase 3:
  03_pretrain_performer/model/lit_bulkformer.py   (LightningModule)
  03_pretrain_performer/model/data.py             (LightningDataModule)
  03_pretrain_performer/model/bulkformer.py       (BulkFormer nn.Module)
```

## Data Flow

```
Phase 1 output                    Phase 2 gene metadata
(ens_gene_biot_abundance.tsv)     (exon lengths, gene order)
         \                              /
          \                            /
           v                          v
       prepare_osdr_data.py
               |
               v
     data/osdr/processed/
     (train + val parquets)
               |
               v
     finetune_bulkformer.py  <--- Phase 3 checkpoint
               |                  (best_model.pt)
               v
     finetune_checkpoints/
     (fine-tuned model)
```
