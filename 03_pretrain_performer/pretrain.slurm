#!/bin/bash
#SBATCH --job-name="pretrain_bulkformer"
#SBATCH --output=pretrain_%j.out
#SBATCH --partition=savio3_gpu
#SBATCH --account=pc_disconasabio
#SBATCH --gres=gpu:A40:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL

# ==============================================================================
# pretrain.slurm -- Pre-train BulkFormer on ARCHS4 data (Lightning + DDP)
# ==============================================================================
#
# Configured for UC Berkeley Savio cluster.
#
# Savio GPU partitions use per-core scheduling -- memory is proportional to
# the number of cores requested (do NOT use --mem). On savio3_gpu with A40
# nodes: 16 cores / 2 GPUs = 8 cores per GPU, so 2 GPUs get 16 cores and
# ~128 GB RAM total.
#
# BEFORE SUBMITTING: Set --account to your Savio project account (e.g. fc_bioinf).
# To use savio4_gpu instead, change --partition and --gres=gpu:A5000:2 with
# --cpus-per-task=4 (4:1 CPU:GPU ratio on savio4_gpu A5000 nodes).
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

module load python
if [ -n "$CONDA_INIT_SCRIPT" ] && [ -f "$CONDA_INIT_SCRIPT" ]; then
    source "$CONDA_INIT_SCRIPT"
fi
conda activate "${BIOFM_ENV:-biofm}"

cd "${SCRIPT_DIR}"

srun python pretrain_bulkformer.py \
    data.data_dir="${DATA_DIR:-${SCRIPT_DIR}/../data}"

echo ""
echo "End: $(date)"
