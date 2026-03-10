# Phase 4: Fine-tune BulkFormer on OSDR Data

Fine-tune the ARCHS4-pretrained BulkFormer model on OSDR reprocessed spaceflight RNA-seq data.

## What This Phase Does

1. **`prepare_osdr_data.py`** -- Collects all `ens_gene_biot_abundance.tsv` files from Phase 1 Kallisto output, maps genes to the canonical set, normalizes, and saves as parquet
2. **`finetune_bulkformer.py`** -- Loads the best pre-trained checkpoint from Phase 3, then fine-tunes on the OSDR data with MLM

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

# Step 2: Fine-tune (single GPU)
python finetune_bulkformer.py \
    --pretrained ../03_pretrain_performer/bulkformer_checkpoints/best_model.pt \
    --data-dir ../data

# Step 2 alternative: Fine-tune (multi-GPU)
torchrun --nproc_per_node=2 finetune_bulkformer.py \
    --pretrained ../03_pretrain_performer/bulkformer_checkpoints/best_model.pt \
    --data-dir ../data
```

## Command-Line Options

### prepare_osdr_data.py

| Flag | Default | Description |
|------|---------|-------------|
| `--kallisto-dir` | `../01_kallisto_reprocessing` | Kallisto output directory |
| `--data-dir` | `../data` | Root data directory |
| `--output-dir` | `<data-dir>/osdr/processed` | Where to save processed OSDR parquets |
| `--val-frac` | `0.2` | Validation fraction |

### finetune_bulkformer.py

| Flag | Default | Description |
|------|---------|-------------|
| `--pretrained` | (required) | Path to pre-trained `.pt` checkpoint |
| `--data-dir` | `../data` | Root data directory |
| `--osdr-data-dir` | `<data-dir>/osdr/processed` | OSDR processed parquet dir |
| `--epochs` | `10` | Max fine-tuning epochs |
| `--batch-size` | `4` | Per-GPU batch size |
| `--lr` | `1e-5` | Learning rate (10x lower than pre-training) |
| `--patience` | `5` | Early stopping patience |
| `--freeze-layers` | `0` | Number of GBFormer layers to freeze |

## Fine-tuning Strategy

- **Learning rate**: 1e-5 (10x lower than pre-training's 1e-4) to avoid catastrophic forgetting
- **Epochs**: 10 max with early stopping (patience=5)
- **Layer freezing** (optional): Use `--freeze-layers N` to freeze the first N GBFormer blocks, training only the later layers and prediction head
- **Shape mismatch handling**: If the OSDR gene count differs from pre-training, layers with incompatible shapes are re-initialized automatically

## Output

```
finetune_checkpoints/
├── epoch_0.pt          # Per-epoch checkpoints
├── ...
├── best_model.pt       # Best model (lowest val loss)
├── config.json         # Fine-tuning config + metadata
├── loss_history.csv    # Per-epoch losses
└── loss_plot.png       # Training curves
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
