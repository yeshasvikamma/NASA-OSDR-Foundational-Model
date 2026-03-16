<p align="center">
  <h1 align="center">NASA OSDR Foundational Model</h1>
  <p align="center">
    A Performer-based foundation model for spaceflight transcriptomics,<br/>
    pre-trained on ARCHS4 bulk RNA-seq and fine-tuned on NASA Open Science Data Repository (OSDR) data.
  </p>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Phases](#pipeline-phases)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
- [Usage](#usage)
  - [Phase 1 -- Kallisto Reprocessing](#phase-1----kallisto-reprocessing)
  - [Phase 2 -- ARCHS4 Data Preparation](#phase-2----archs4-data-preparation)
  - [Phase 3 -- Pre-train BulkFormer](#phase-3----pre-train-bulkformer)
  - [Phase 4 -- Fine-tune on OSDR Data](#phase-4----fine-tune-on-osdr-data)
- [Model Architecture](#model-architecture)
- [Repository Structure](#repository-structure)
- [References](#references)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Overview

This project develops a **foundation model** for analyzing bulk RNA-seq gene expression from spaceflight experiments. The approach is two-stage:

1. **Pre-train** a Performer-based transformer (BulkFormer) on hundreds of thousands of publicly available human and mouse RNA-seq samples from [ARCHS4](https://archs4.org), learning general-purpose gene expression representations.

2. **Fine-tune** the pre-trained model on spaceflight-specific transcriptomics data reprocessed from the [NASA Open Science Data Repository (OSDR)](https://osdr.nasa.gov), adapting the model to capture the unique biological signatures of spaceflight.

The model uses **masked language modeling (MLM)** -- randomly masking 15% of gene expression values and training the model to reconstruct them -- enabling it to learn rich, contextualized representations of gene expression patterns across tissues, conditions, and organisms.

---

## Pipeline Phases

The project is organized into four sequential phases, each in its own directory with dedicated documentation:

| Phase | Directory | Description | Compute |
|:-----:|-----------|-------------|---------|
| 1 | [`01_kallisto_reprocessing/`](01_kallisto_reprocessing/) | Reprocess OSDR RNA-seq datasets through Kallisto to produce standardized gene-level counts | CPU (SLURM array) |
| 2 | [`02_data_preparation/`](02_data_preparation/) | Download ARCHS4 human + mouse expression data and preprocess into model-ready format | CPU, high-memory |
| 3 | [`03_pretrain_performer/`](03_pretrain_performer/) | Pre-train BulkFormer on ARCHS4 data using masked language modeling | Multi-GPU |
| 4 | [`04_finetune_osdr/`](04_finetune_osdr/) | Fine-tune the pre-trained model on OSDR reprocessed spaceflight data | GPU |

Each phase has its own `README.md` with detailed instructions, command-line options, and expected outputs.

---

## Getting Started

### Prerequisites

This project is configured for the **UC Berkeley Savio** HPC cluster ([hardware config](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/hardware-config/)). The following must be available:

| Requirement | Used In | Savio Notes |
|-------------|---------|-------------|
| **SLURM** scheduler | All phases | Pre-installed on Savio |
| **Conda** (`module load python`) | All phases | Available via Savio module system |
| **AWS CLI** | Phase 1 | Install in conda env; Savio nodes have internet access |
| **CUDA-capable GPUs** | Phases 3, 4 | `savio4_gpu`: A5000 (24 GB, 26 nodes) or L40 (46 GB, 3 nodes) |
| **wget** | Phase 2 | Pre-installed on Savio |
| **Kallisto indices** | Phase 1 | Reference index files (provided separately) |

### Installation

Clone this repository to your HPC working directory:

```bash
git clone https://github.com/yeshasvikamma/NASA-OSDR-Foundational-Model.git
cd NASA-OSDR-Foundational-Model
```

Create the required conda environments:

```bash
# Phase 1: Kallisto alignment + R for gene-level aggregation
conda env create -f environments/kallisto_v0.51.1.yml

# Phases 2-4: PyTorch, Performer, PyG, ESM2, and data processing
conda env create -f environments/biofm.yml
```

### Configuration

All HPC-specific settings are centralized in a single file. Edit it **once** before running any phase:

```bash
cp config.env config.env.bak   # optional backup
nano config.env                 # or your preferred editor
```

Replace every `CHANGE_ME` value:

| Setting | Example | Description |
|---------|---------|-------------|
| `HPC_ACCOUNT` | `"fc_bioinf"` | **Required** -- your Savio FCA/Condo account |
| `HPC_EMAIL` | `"you@berkeley.edu"` | Email for SLURM notifications |
| `HPC_PARTITION` | `"savio4_htc"` | CPU partition -- 212 nodes, 56 cores, per-core scheduling |
| `HPC_GPU_PARTITION` | `"savio4_gpu"` | GPU partition -- A5000 (26 nodes) or L40 (3 nodes) |
| `GPU_TYPE` | `"A5000"` | GPU for `--gres` (`A5000` 24 GB or `L40` 46 GB) |

All SLURM scripts source this file automatically. You must also update `--account=CHANGE_ME` in each `.slurm` file with your Savio account, or use a `sed` one-liner:

```bash
# Replace CHANGE_ME in all SLURM scripts at once
find . -name "*.slurm" -exec sed -i 's/--account=CHANGE_ME/--account=fc_bioinf/g' {} +
```

---

## Usage

### Phase 1 -- Kallisto Reprocessing

Reprocess OSDR mouse RNA-seq datasets (starting with 4 test datasets: OSD-47, OSD-104, OSD-202, OSD-244) through Kallisto to produce gene-level expression counts.

```bash
cd 01_kallisto_reprocessing/

# Set up per-dataset directories and SLURM scripts
python setup_datasets.py test_mouse_datasets.csv \
    --template-dir ./template_scripts \
    --make-dirs-script ./make_dirs.sh

# Submit all Kallisto jobs
bash run_kallisto.sh
```

See [`01_kallisto_reprocessing/README.md`](01_kallisto_reprocessing/README.md) for full details.

### Phase 2 -- ARCHS4 Data Preparation

Download ARCHS4 v2.3 HDF5 files (~40 GB total) and preprocess into normalized parquet matrices.

```bash
cd 02_data_preparation/

# Download ARCHS4 HDF5 files + gene metadata
sbatch download_archs4.slurm

# Preprocess: filter genes, normalize, split train/val/test
sbatch preprocess_archs4.slurm
```

See [`02_data_preparation/README.md`](02_data_preparation/README.md) for full details.

### Phase 3 -- Pre-train BulkFormer

Pre-train the BulkFormer model on ARCHS4 data using PyTorch Lightning with multi-GPU DDP, mixed precision, cosine warmup LR scheduling, W&B experiment tracking, and torchmetrics.

```bash
cd 03_pretrain_performer/

# Submit multi-GPU training job
sbatch pretrain.slurm

# Or run interactively (Lightning auto-detects GPUs)
python pretrain_bulkformer.py

# Override any config value from the CLI
python pretrain_bulkformer.py training.lr=5e-5 training.batch_size=8
```

All hyperparameters are in `configs/pretrain.yaml`. See [`03_pretrain_performer/README.md`](03_pretrain_performer/README.md) for full details.

### Phase 4 -- Fine-tune on OSDR Data

Convert Kallisto output to model format and fine-tune the pre-trained model on OSDR spaceflight data.

```bash
cd 04_finetune_osdr/

# Runs data preparation + fine-tuning in a single job
sbatch finetune.slurm

# Or run fine-tuning interactively with config overrides
python finetune_bulkformer.py training.freeze_layers=1
```

All hyperparameters are in `configs/finetune.yaml`. See [`04_finetune_osdr/README.md`](04_finetune_osdr/README.md) for full details.

---

## Model Architecture

**BulkFormer** is a Performer-based transformer designed for bulk RNA-seq gene expression. It integrates multiple biological priors into a unified architecture:

```
Input: [B, G] continuous gene expression vector
 |
 |--- Rotary Expression Embedding (REE)    Sinusoidal encoding of expression values
 |--- ESM2 Gene Identity Projection        Protein language model embeddings (320-dim)
 |--- Sample-level AutoEncoder Embedding   Global transcriptome context
 |
 v
Fusion + MLP Projection
 |
 v
N x GBFormer Encoder Blocks:
 +-- LayerNorm
 +-- GCN Message Passing              Gene-gene coexpression graph (top-20 KNN)
 +-- Learned Expression Binning        Sort genes into expression-regime bins
 +-- Local Performer Attention         Per-bin attention (FAVOR+ linear complexity)
 +-- Global Performer Attention        Cross-bin attention over all genes
 |
 v
LayerNorm --> Prediction Head MLP --> [B, G] reconstructed expression
```

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| `dim` | 320 | Embedding dimension |
| `gb_repeat` | 1 | Number of GBFormer encoder blocks |
| `p_repeat` | 2 | Global Performer layers per block |
| `bins` | 10 | Number of expression bins |
| `bin_head` | 8 | Attention heads (local, per-bin) |
| `full_head` | 4 | Attention heads (global) |
| `mask_ratio` | 0.15 | Fraction of genes masked during training |

---

## Repository Structure

```
NASA-OSDR-Foundational-Model/
|
+-- README.md                              Project documentation (this file)
+-- LICENSE                                MIT License
+-- .gitignore                             Git ignore rules
+-- config.env                             HPC configuration (edit once)
|
+-- environments/                          Conda environment definitions
|   +-- kallisto_v0.51.1.yml               Phase 1: Kallisto + R
|   +-- biofm.yml                          Phases 2-4: PyTorch + Lightning stack
|
+-- 01_kallisto_reprocessing/              Phase 1: OSDR Kallisto pipeline
|   +-- README.md                          Phase documentation
|   +-- setup_datasets.py                  Dataset directory + SLURM script generator
|   +-- run_kallisto.sh                    Batch job submission wrapper
|   +-- make_dirs.sh                       Directory structure creator
|   +-- test_mouse_datasets.csv            Test dataset manifest (4 OSD datasets)
|   +-- kallisto_v0.51.1.yml               Conda environment spec
|   +-- template_scripts/                  SLURM job templates (SE/PE, +/-ERCC)
|   +-- kallisto_indices/                  Reference indices (provided separately)
|
+-- 02_data_preparation/                   Phase 2: ARCHS4 download + preprocessing
|   +-- README.md                          Phase documentation
|   +-- download_archs4.sh                 ARCHS4 + gene metadata downloader
|   +-- download_archs4.slurm              SLURM wrapper for download
|   +-- preprocess_archs4.py               HDF5 -> normalized parquet pipeline
|   +-- preprocess_archs4.slurm            SLURM wrapper for preprocessing
|
+-- 03_pretrain_performer/                 Phase 3: BulkFormer pre-training
|   +-- README.md                          Phase documentation
|   +-- model/bulkformer.py                BulkFormer model definition (pure PyTorch)
|   +-- model/lit_bulkformer.py            LightningModule wrapper (training logic)
|   +-- model/data.py                      LightningDataModule + BulkMLMDataset
|   +-- configs/pretrain.yaml              Pre-training hyperparameters
|   +-- pretrain_bulkformer.py             Lightning Trainer script
|   +-- pretrain.slurm                     Multi-GPU SLURM job script
|
+-- 04_finetune_osdr/                      Phase 4: Fine-tuning on OSDR data
|   +-- README.md                          Phase documentation
|   +-- prepare_osdr_data.py               Kallisto output -> model input converter
|   +-- configs/finetune.yaml              Fine-tuning hyperparameters
|   +-- finetune_bulkformer.py             Lightning Trainer script
|   +-- finetune.slurm                     GPU SLURM job script
|
+-- data/                                  Runtime data directory (not committed)
    +-- archs4/                            ARCHS4 HDF5 files + processed parquets
    +-- embeddings/                        ESM2 gene identity embeddings
    +-- ensembl/                           Ortholog mappings, exon lengths
    +-- graph/                             KNN gene-gene coexpression graph edges
```

---

## References

- **ARCHS4** -- Lachmann A, Torre D, Keenan AB, et al. *Massive mining of publicly available RNA-seq data from human and mouse.* Nature Communications, 2018. [https://archs4.org](https://archs4.org)

- **BioFM** -- Walt A. BulkFormer implementation. [https://github.com/alwalt/BioFM](https://github.com/alwalt/BioFM)

- **Performer** -- Choromanski KM, Likhosherstov V, Dohan D, et al. *Rethinking Attention with Performers.* ICLR, 2021. [arXiv:2009.14794](https://arxiv.org/abs/2009.14794)

- **ESM-2** -- Lin Z, Akin H, Rao R, et al. *Evolutionary-scale prediction of atomic-level protein structure with a language model.* Science, 2023.

- **NASA OSDR** -- Open Science Data Repository. [https://osdr.nasa.gov](https://osdr.nasa.gov)

- **Kallisto** -- Bray NL, Pimentel H, Melsted P, Pachter L. *Near-optimal probabilistic RNA-seq quantification.* Nature Biotechnology, 2016.

---

## Acknowledgments

This work is part of the NASA Open Science Data Repository (OSDR) Foundation Model initiative. The BulkFormer architecture is adapted from [BioFM](https://github.com/alwalt/BioFM) by Walt Shands. ARCHS4 data is provided by the [Ma'ayan Laboratory](https://labs.icahn.mssm.edu/maayanlab/) at the Icahn School of Medicine at Mount Sinai.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
