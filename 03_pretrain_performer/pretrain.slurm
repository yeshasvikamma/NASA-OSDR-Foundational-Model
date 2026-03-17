#!/bin/bash
#SBATCH --job-name="pretrain_bulkformer"
#SBATCH --output=pretrain_%j.out
#SBATCH --partition=savio4_gpu
#SBATCH --account=pc_disconasabio
#SBATCH --gres=gpu:A5000:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL

# ==============================================================================
# pretrain.slurm -- Pre-train BulkFormer on ARCHS4 data (Lightning + DDP)
# ==============================================================================
#
# Configured for UC Berkeley Savio cluster.
#
# Savio GPU partitions use per-core scheduling -- memory is proportional to
# the number of cores requested (do NOT use --mem).
#
# A5000 nodes (26 nodes): 32 cores / 8 GPUs = 4 cores per GPU (4:1 ratio).
#   2 GPUs → --gres=gpu:A5000:2, --cpus-per-task=4, 8 cores total, ~128 GB RAM.
# L40 nodes (3 nodes):    64 cores / 8 GPUs = 8 cores per GPU (8:1 ratio).
#   2 GPUs → --gres=gpu:L40:2, --cpus-per-task=8, 16 cores total, ~198 GB RAM.
#   L40 offers 46 GB VRAM vs A5000's 24 GB for larger batch sizes.
#
# BEFORE SUBMITTING: Set --account to your Savio project account (e.g. fc_bioinf).
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
echo "Start:      $(date)"
echo ""

module load anaconda3
eval "$(conda shell.bash hook)"
if [ -n "$CONDA_INIT_SCRIPT" ] && [ -f "$CONDA_INIT_SCRIPT" ]; then
    source "$CONDA_INIT_SCRIPT"
fi
conda activate "${BIOFM_ENV:-biofm}"

cd "${SCRIPT_DIR}"

srun python pretrain_bulkformer.py \
    data.data_dir="${DATA_DIR:-${SCRIPT_DIR}/../data}"

echo ""
echo "End: $(date)"
