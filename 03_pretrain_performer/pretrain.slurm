#!/bin/bash
#SBATCH --job-name="pretrain_bulkformer"
#SBATCH --output=pretrain_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=64000
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL

# ==============================================================================
# pretrain.slurm -- Pre-train BulkFormer on ARCHS4 data (Lightning + DDP)
# ==============================================================================
#
# BEFORE SUBMITTING: Update the #SBATCH lines above:
#   --partition   : your GPU partition name
#   --gres        : GPU type/count, e.g. gpu:a100:2, gpu:v100:2
#   --ntasks-per-node : must equal the number of GPUs
#   --mail-user   : your email
#   --mem         : memory (64 GB recommended)
#   --time        : wall time (depends on dataset size + GPU speed)
#
# Lightning auto-detects SLURM environment variables and sets up DDP
# automatically -- no need for torchrun.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../config.env" 2>/dev/null || true

echo "============================================"
echo "BulkFormer Pre-training (Lightning)"
echo "============================================"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       ${HOSTNAME}"
echo "GPUs:       ${SLURM_GPUS_ON_NODE:-$NUM_GPUS}"
echo "Tasks:      ${SLURM_NTASKS}"
echo "CPUs/task:  ${SLURM_CPUS_PER_TASK}"
echo "Memory:     ${SLURM_MEM_PER_NODE} MB"
echo "Start:      $(date)"
echo ""

# Activate conda environment
if [ -n "$CONDA_INIT_SCRIPT" ] && [ -f "$CONDA_INIT_SCRIPT" ]; then
    source "$CONDA_INIT_SCRIPT"
fi
conda activate "${BIOFM_ENV:-biofm}"

cd "${SCRIPT_DIR}"

# Lightning detects SLURM and configures DDP via the strategy="ddp" flag.
# srun launches one process per task (= per GPU).
srun python pretrain_bulkformer.py \
    data.data_dir="${DATA_DIR:-${SCRIPT_DIR}/../data}"

echo ""
echo "End: $(date)"
