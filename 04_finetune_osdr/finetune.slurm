#!/bin/bash
#SBATCH --job-name="finetune_bulkformer"
#SBATCH --output=finetune_%j.out
#SBATCH --partition=savio3_gpu
#SBATCH --account=pc_disconasabio
#SBATCH --gres=gpu:A40:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL

# ==============================================================================
# finetune.slurm -- Fine-tune BulkFormer on OSDR data (Lightning + DDP)
# ==============================================================================
#
# Configured for UC Berkeley Savio cluster.
#
# This script runs TWO steps:
#   1. prepare_osdr_data.py  -- converts Kallisto output to model input format
#   2. finetune_bulkformer.py -- loads pre-trained model and fine-tunes on OSDR
#
# BEFORE SUBMITTING: Set --account to your Savio project account (e.g. fc_bioinf).
#
# Lightning auto-detects SLURM and configures DDP automatically.
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

module load python
if [ -n "$CONDA_INIT_SCRIPT" ] && [ -f "$CONDA_INIT_SCRIPT" ]; then
    source "$CONDA_INIT_SCRIPT"
fi
conda activate "${BIOFM_ENV:-biofm}"

DATA="${DATA_DIR:-${SCRIPT_DIR}/../data}"

echo ""
echo "============================================"
echo "Step 1: Preparing OSDR data"
echo "============================================"
python "${SCRIPT_DIR}/prepare_osdr_data.py" \
    --kallisto-dir "${KALLISTO_DIR:-${SCRIPT_DIR}/../01_kallisto_reprocessing}" \
    --data-dir "${DATA}"

echo ""
echo "============================================"
echo "Step 2: Fine-tuning"
echo "============================================"

cd "${SCRIPT_DIR}"

srun python finetune_bulkformer.py \
    data.data_dir="${DATA}"

echo ""
echo "End: $(date)"
