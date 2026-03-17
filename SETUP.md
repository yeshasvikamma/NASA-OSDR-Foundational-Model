# Setup Guide

Complete setup instructions for the NASA OSDR Foundation Model project on the UC Berkeley Savio HPC cluster.

**Local machine (Windows/Mac)**: used only for editing code and pushing to GitHub. No conda or special software needed locally.

**Savio HPC**: where all computation happens -- conda environments, SLURM jobs, GPU training.

---

## Table of Contents

- [Local Development Workflow](#local-development-workflow)
- [Savio HPC Setup](#savio-hpc-setup)
  - [Step 1: SSH into Savio](#step-1-ssh-into-savio)
  - [Step 2: Clone the Repository](#step-2-clone-the-repository)
  - [Step 3: Create Conda Environments](#step-3-create-conda-environments)
  - [Step 4: Configure HPC Settings](#step-4-configure-hpc-settings)
  - [Step 5: Verify the Setup](#step-5-verify-the-setup)
- [Data Dependencies](#data-dependencies)
- [Run Order](#run-order)
- [Savio Quick Reference](#savio-quick-reference)
- [Troubleshooting](#troubleshooting)

---

## Local Development Workflow

Your local machine is only for editing and version control. The workflow is:

```
[Local] Edit code in Cursor/VSCode
   |
   v
[Local] git add / commit / push
   |
   v
[Savio] git pull
   |
   v
[Savio] sbatch job.slurm
```

No conda, Python, or special dependencies are needed locally. If your editor needs a Python interpreter for linting/autocomplete, install Python 3.10+ from [python.org](https://www.python.org/downloads/) and `pip install` the pure-Python packages (omegaconf, rich, etc.), but this is optional.

---

## Savio HPC Setup

### Step 1: SSH into Savio

```bash
ssh <your_username>@hpc.brc.berkeley.edu
```

You will need:
- A Savio account ([request access](https://docs-research-it.berkeley.edu/services/high-performance-computing/accounts/))
- One-time password (OTP) setup via Google Authenticator or Authy ([instructions](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/logging-in/setting-up-one-time-passwords/))

### Step 2: Clone the Repository

Clone into your home directory or scratch space:

```bash
cd /global/home/users/$USER
# or: cd /global/scratch/users/$USER

git clone https://github.com/yeshasvikamma/NASA-OSDR-Foundational-Model.git
cd NASA-OSDR-Foundational-Model
```

### Step 3: Create Conda Environments

Savio provides conda through the module system. Load it first:

```bash
module load anaconda3
```

Then create both environments:

```bash
# Phase 1: Kallisto + R (for gene-level aggregation)
conda env create -f environments/kallisto_v0.51.1.yml

# Phases 2-4: PyTorch, Lightning, Performer, ESM2, etc.
conda env create -f environments/biofm.yml
```

This will take 10-20 minutes. If this is your first time using conda on Savio, also run:

```bash
conda init bash
source ~/.bashrc
```

If `biofm.yml` fails on dependency resolution, try:

```bash
conda env create -f environments/biofm.yml --solver=libmamba
```

### Step 4: Configure HPC Settings

The project uses a single config file that all SLURM scripts source. It is already pre-filled for this project:

```bash
cat config.env
```

Verify these values match your Savio account:

| Setting | Current Value | What It Controls |
|---------|--------------|------------------|
| `HPC_ACCOUNT` | `pc_disconasabio` | SLURM `--account` for billing |
| `HPC_EMAIL` | `rishal_misra@berkeley.edu` | SLURM email notifications |
| `HPC_PARTITION` | `savio4_htc` | CPU jobs (Kallisto, data prep) |
| `HPC_GPU_PARTITION` | `savio4_gpu` | GPU jobs (pretrain, finetune) |
| `GPU_TYPE` | `A5000` | GPU model for `--gres` |

If any of these need changing, edit with:

```bash
nano config.env
```

### Step 5: Verify the Setup

Run these checks to confirm everything is ready:

```bash
# Check Kallisto environment
conda activate kallisto_v0.51.1
kallisto version
# Expected: kallisto, version 0.51.1
R --version | head -1
# Expected: R version 4.4.1 ...
conda deactivate

# Check biofm environment
conda activate biofm
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import lightning; print(f'Lightning {lightning.__version__}')"
python -c "from performer_pytorch import Performer; print('Performer OK')"
conda deactivate

# Check SLURM access
sinfo -p savio4_htc | head -5
sinfo -p savio4_gpu | head -5

# Check account allocation
sacctmgr show associations user=$USER format=Account,Partition
```

> **Note**: `torch.cuda.is_available()` will return `False` on login nodes (no GPUs). This is expected. It will return `True` when running on `savio4_gpu` compute nodes.

---

## Data Dependencies

Each phase requires specific data. Here is what must exist before each phase runs:

### Phase 1 -- Kallisto Reprocessing

| Dependency | Path | How to Get It |
|-----------|------|---------------|
| Kallisto reference indices | `01_kallisto_reprocessing/kallisto_indices/` | Provided separately (not in repo). Copy the `.idx` and `.rda` files into this directory. |

### Phase 2 -- ARCHS4 Data Preparation

| Dependency | Path | How to Get It |
|-----------|------|---------------|
| None | -- | `download_archs4.slurm` fetches everything automatically. |

Phase 2 produces:
- `data/archs4/` -- raw HDF5 files
- `data/archs4/processed/` -- normalized parquet matrices
- `data/ensembl/` -- gene metadata
- `data/embeddings/` -- ESM2 gene embeddings (if `--skip-esm2` is removed)
- `data/graph/` -- gene-gene KNN graph (if `--skip-graph` is removed)

### Phase 3 -- Pre-training

| Dependency | Path | How to Get It |
|-----------|------|---------------|
| Processed ARCHS4 data | `data/archs4/processed/` | Phase 2 output |
| ESM2 embeddings | `data/embeddings/esm2_t6_8M_UR50D_gene_embeddings.pt` | Phase 2 (or provided separately) |
| Gene graph | `data/graph/edge_index_top20.pt` | Phase 2 (or provided separately) |

### Phase 4 -- Fine-tuning

| Dependency | Path | How to Get It |
|-----------|------|---------------|
| Kallisto gene counts | `01_kallisto_reprocessing/OSD-*/` | Phase 1 output |
| Pre-trained checkpoint | `03_pretrain_performer/bulkformer_checkpoints/best_model.pt` | Phase 3 output |
| Gene order file | `data/archs4/processed/gene_order.csv` | Phase 2 output |

---

## Run Order

Phases 1 and 2 are independent and can run in parallel. Phase 3 requires Phase 2. Phase 4 requires both Phases 1 and 3.

```
Phase 1 (Kallisto) ─────────────────────────────┐
                                                 ├──> Phase 4 (Fine-tune)
Phase 2 (Download) ──> Phase 2 (Preprocess) ──> Phase 3 (Pre-train) ─┘
```

### Phase 1: Kallisto Reprocessing

```bash
cd 01_kallisto_reprocessing/

# Generate per-dataset directories and SLURM scripts
conda activate biofm
python setup_datasets.py test_mouse_datasets.csv \
    --template-dir ./template_scripts \
    --make-dirs-script ./make_dirs.sh
conda deactivate

# Submit all Kallisto array jobs
bash run_kallisto.sh

# Monitor
squeue -u $USER
```

### Phase 2: ARCHS4 Data Preparation

```bash
cd 02_data_preparation/

# Step 1: Download ARCHS4 HDF5 files + gene metadata (~40 GB)
sbatch download_archs4.slurm

# Wait for download to complete, then:
# Step 2: Preprocess into normalized parquet matrices
sbatch preprocess_archs4.slurm
```

### Phase 3: Pre-train BulkFormer

```bash
cd 03_pretrain_performer/

# Submit multi-GPU training (2x A5000, ~48 hours)
sbatch pretrain.slurm

# Monitor training progress
squeue -u $USER
tail -f pretrain_*.out
```

### Phase 4: Fine-tune on OSDR

```bash
cd 04_finetune_osdr/

# Runs data preparation + fine-tuning in one job
sbatch finetune.slurm
```

---

## Savio Quick Reference

### Partitions

| Partition | Nodes | Hardware | Scheduling | Use For |
|-----------|-------|----------|------------|---------|
| `savio4_htc` | 212 | 56-core Xeon Gold 6330, 256-512 GB | Per-core | Kallisto, data prep, downloads |
| `savio4_gpu` | 26 | 8x RTX A5000 (24 GB), 32 cores | Per-core | Training (default) |
| `savio4_gpu` | 3 | 8x L40 (46 GB), 64 cores | Per-core | Training (more VRAM) |

Full hardware details: [Savio Hardware Config](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/hardware-config/)

### Common Commands

```bash
# Check job queue
squeue -u $USER

# Cancel a job
scancel <JOB_ID>

# Check partition availability
sinfo -p savio4_htc -o "%P %a %D %t"
sinfo -p savio4_gpu -o "%P %a %D %t"

# Check your allocation balance
check_usage.sh -a pc_disconasabio

# Interactive GPU session (for debugging)
srun --partition=savio4_gpu --account=pc_disconasabio \
     --gres=gpu:A5000:1 --cpus-per-task=4 --time=1:00:00 --pty bash

# View job output
cat pretrain_<JOB_ID>.out
```

### File Storage

| Location | Path | Quota | Purpose |
|----------|------|-------|---------|
| Home | `/global/home/users/$USER` | 10 GB | Config files, small scripts |
| Scratch | `/global/scratch/users/$USER` | 12 TB, 90-day purge | Large data, training output |

For this project, clone to **scratch** if your data will exceed 10 GB (it will -- ARCHS4 alone is ~40 GB).

---

## Troubleshooting

### `conda: command not found`

You need to load the anaconda3 module first:

```bash
module load anaconda3
conda --version
```

### `conda env create` hangs or fails

Try the libmamba solver (faster dependency resolution):

```bash
conda env create -f environments/biofm.yml --solver=libmamba
```

Or create in steps:

```bash
conda create -n biofm python=3.10 "numpy>=1.24,<2" "scipy>=1.10,<1.14" \
    pandas scikit-learn matplotlib h5py pyarrow "setuptools<71" "libstdcxx-ng>=12" -c conda-forge -y
conda activate biofm
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install lightning==2.2.5 "torchmetrics>=1.3,<1.4" wandb omegaconf rich
pip install performer-pytorch fair-esm archs4py
pip install "torch-geometric>=2.5"
pip install fastparquet safetensors einops
```

### SLURM job fails with `Invalid partition`

Verify the partition exists and your account has access:

```bash
sinfo -p savio4_htc
sacctmgr show associations user=$USER format=Account,Partition
```

### `CUDA not available` during training

This is expected on login nodes. On compute nodes, check:

```bash
srun --partition=savio4_gpu --account=pc_disconasabio \
     --gres=gpu:A5000:1 --cpus-per-task=4 --time=0:10:00 --pty bash

module load anaconda3
conda activate biofm
python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi
```

### `ModuleNotFoundError` for project imports

Make sure you are running scripts from their own directory:

```bash
cd 03_pretrain_performer/
python pretrain_bulkformer.py
```

The SLURM scripts handle this with `cd "${SCRIPT_DIR}"`.

### Job runs out of memory

On `savio4_htc`, memory is proportional to cores requested. Increase `--cpus-per-task` to get more RAM:
- 256 GB nodes: ~4.6 GB/core
- 512 GB nodes: ~9.1 GB/core

### Lightning or PyTorch import fails (`GLIBCXX`, `pkg_resources`, `torch._dynamo`)

This usually means pip upgraded torch or numpy past the versions conda installed.
The `biofm.yml` includes pip-section constraints (`torch>=2.3,<2.4`, `numpy>=1.24,<2`)
to prevent this, but if the env is already broken, recreate it:

```bash
conda deactivate
conda env remove -n biofm
module load anaconda3
conda env create -f environments/biofm.yml
conda activate biofm
python -c "import lightning; print(f'Lightning {lightning.__version__}')"
```

### AWS S3 download fails in Kallisto jobs

Savio compute nodes have internet access, but if downloads fail:

```bash
# Test connectivity from a compute node
srun --partition=savio4_htc --account=pc_disconasabio \
     --cpus-per-task=1 --time=0:10:00 --pty bash

aws s3 ls --no-sign-request --region us-west-2 s3://nasa-osdr/OSD-47/
```
