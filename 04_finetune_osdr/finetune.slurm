#!/bin/bash
#SBATCH --job-name="finetune_bulkformer"
#SBATCH --output=finetune_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=64000
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL

# ==============================================================================
# finetune.slurm -- Fine-tune BulkFormer on OSDR reprocessed data
# ==============================================================================
#
# This script runs TWO steps:
#   1. prepare_osdr_data.py  -- converts Kallisto output to model input format
#   2. finetune_bulkformer.py -- loads pre-trained model and fine-tunes on OSDR
#
# BEFORE SUBMITTING: Update #SBATCH lines with your HPC settings.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../config.env" 2>/dev/null || true

echo "============================================"
echo "BulkFormer Fine-tuning on OSDR Data"
echo "============================================"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       ${HOSTNAME}"
echo "GPUs:       ${SLURM_GPUS_ON_NODE:-$NUM_GPUS}"
echo "Start:      $(date)"
echo ""

# Activate conda
if [ -n "$CONDA_INIT_SCRIPT" ] && [ -f "$CONDA_INIT_SCRIPT" ]; then
    source "$CONDA_INIT_SCRIPT"
fi
conda activate "${BIOFM_ENV:-biofm}"

DATA="${DATA_DIR:-${SCRIPT_DIR}/../data}"
PRETRAINED="${PRETRAIN_DIR:-${SCRIPT_DIR}/../03_pretrain_performer}/bulkformer_checkpoints/best_model.pt"

# Step 1: Prepare OSDR data
echo ""
echo "============================================"
echo "Step 1: Preparing OSDR data"
echo "============================================"
python "${SCRIPT_DIR}/prepare_osdr_data.py" \
    --kallisto-dir "${KALLISTO_DIR:-${SCRIPT_DIR}/../01_kallisto_reprocessing}" \
    --data-dir "${DATA}"

# Step 2: Fine-tune
echo ""
echo "============================================"
echo "Step 2: Fine-tuning"
echo "============================================"

NGPUS=${SLURM_GPUS_ON_NODE:-${NUM_GPUS:-2}}

cd "${SCRIPT_DIR}"

torchrun --nproc_per_node=${NGPUS} finetune_bulkformer.py \
    --pretrained "${PRETRAINED}" \
    --data-dir "${DATA}" \
    --epochs 10 \
    --batch-size 4 \
    --lr 1e-5 \
    --patience 5 \
    --checkpoint-dir ./finetune_checkpoints

echo ""
echo "End: $(date)"
