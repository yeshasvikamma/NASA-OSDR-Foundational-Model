#!/bin/bash
#SBATCH --job-name="finetune_bulkformer"
#SBATCH --output=finetune_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=64000
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL

# ==============================================================================
# finetune.slurm -- Fine-tune BulkFormer on OSDR data (Lightning + DDP)
# ==============================================================================
#
# This script runs TWO steps:
#   1. prepare_osdr_data.py  -- converts Kallisto output to model input format
#   2. finetune_bulkformer.py -- loads pre-trained model and fine-tunes on OSDR
#
# Lightning auto-detects SLURM and configures DDP automatically.
#
# BEFORE SUBMITTING: Update #SBATCH lines with your HPC settings.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../config.env" 2>/dev/null || true

echo "============================================"
echo "BulkFormer Fine-tuning on OSDR (Lightning)"
echo "============================================"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       ${HOSTNAME}"
echo "GPUs:       ${SLURM_GPUS_ON_NODE:-$NUM_GPUS}"
echo "Tasks:      ${SLURM_NTASKS}"
echo "Start:      $(date)"
echo ""

# Activate conda
if [ -n "$CONDA_INIT_SCRIPT" ] && [ -f "$CONDA_INIT_SCRIPT" ]; then
    source "$CONDA_INIT_SCRIPT"
fi
conda activate "${BIOFM_ENV:-biofm}"

DATA="${DATA_DIR:-${SCRIPT_DIR}/../data}"

# Step 1: Prepare OSDR data (single process, CPU only)
echo ""
echo "============================================"
echo "Step 1: Preparing OSDR data"
echo "============================================"
python "${SCRIPT_DIR}/prepare_osdr_data.py" \
    --kallisto-dir "${KALLISTO_DIR:-${SCRIPT_DIR}/../01_kallisto_reprocessing}" \
    --data-dir "${DATA}"

# Step 2: Fine-tune (Lightning DDP via srun)
echo ""
echo "============================================"
echo "Step 2: Fine-tuning"
echo "============================================"

cd "${SCRIPT_DIR}"

srun python finetune_bulkformer.py \
    data.data_dir="${DATA}"

echo ""
echo "End: $(date)"
