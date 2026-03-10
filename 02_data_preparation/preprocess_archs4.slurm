#!/bin/bash
#SBATCH --job-name="preprocess_archs4"
#SBATCH --output=preprocess_archs4_%j.out
#SBATCH --partition=normal
#SBATCH --mem=64000
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mail-type=END,FAIL

# ==============================================================================
# preprocess_archs4.slurm -- SLURM wrapper for ARCHS4 preprocessing
# ==============================================================================
# Requires ~40-60 GB RAM to hold full ARCHS4 expression matrices in memory.
#
# BEFORE SUBMITTING: Update the #SBATCH lines above with your HPC settings.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../config.env" 2>/dev/null || true

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${HOSTNAME}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE} MB"
echo "Start: $(date)"
echo ""

# Activate the BioFM conda environment
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
