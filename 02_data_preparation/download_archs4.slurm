#!/bin/bash
#SBATCH --job-name="download_archs4"
#SBATCH --output=download_archs4_%j.out
#SBATCH --partition=savio3_htc
#SBATCH --account=CHANGE_ME
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL

# ==============================================================================
# download_archs4.slurm -- SLURM wrapper for ARCHS4 data download
# ==============================================================================
# Configured for UC Berkeley Savio cluster.
#
# NOTE: Savio compute nodes have internet access, so downloads work from
# savio3_htc. 1 core is sufficient for a download job.
#
# BEFORE SUBMITTING: Set --account to your Savio project account (e.g. fc_bioinf).
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
