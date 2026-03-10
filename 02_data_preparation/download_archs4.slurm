#!/bin/bash
#SBATCH --job-name="download_archs4"
#SBATCH --output=download_archs4_%j.out
#SBATCH --partition=normal
#SBATCH --mem=8000
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL

# ==============================================================================
# download_archs4.slurm -- SLURM wrapper for ARCHS4 data download
# ==============================================================================
# BEFORE SUBMITTING: Update the #SBATCH lines above with your HPC settings,
# or use a partition that has internet access.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../config.env" 2>/dev/null || true

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${HOSTNAME}"
echo "Start: $(date)"
echo ""

bash "${SCRIPT_DIR}/download_archs4.sh"

echo ""
echo "End: $(date)"
