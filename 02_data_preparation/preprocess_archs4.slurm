#!/bin/bash
#SBATCH --job-name="preprocess_archs4"
#SBATCH --output=preprocess_archs4_%j.out
#SBATCH --partition=savio4_htc
#SBATCH --account=pc_disconasabio
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --mail-type=END,FAIL

# ==============================================================================
# preprocess_archs4.slurm -- SLURM wrapper for ARCHS4 preprocessing
# ==============================================================================
# Configured for UC Berkeley Savio cluster.
#
# Requires ~40-60 GB RAM to hold full ARCHS4 expression matrices in memory.
# On savio4_htc, memory is ~4.6-9.1 GB/core (256 or 512 GB nodes), so
# 16 cores provides at least ~74 GB (up to ~146 GB on 512 GB nodes).
#
# BEFORE SUBMITTING: Set --account to your Savio project account (e.g. fc_bioinf).
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../config.env" 2>/dev/null || true

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${HOSTNAME}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Start: $(date)"
echo ""

module load anaconda3
eval "$(conda shell.bash hook)"
if [ -n "$CONDA_INIT_SCRIPT" ] && [ -f "$CONDA_INIT_SCRIPT" ]; then
    source "$CONDA_INIT_SCRIPT"
fi
conda activate "${BIOFM_ENV:-biofm}"

python "${SCRIPT_DIR}/preprocess_archs4.py" \
    --data-dir "${DATA_DIR:-${SCRIPT_DIR}/../data}" \
    --skip-esm2 \
    --skip-graph

echo ""
echo "End: $(date)"
