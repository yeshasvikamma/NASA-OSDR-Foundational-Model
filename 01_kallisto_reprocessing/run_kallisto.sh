#!/bin/bash
# ==============================================================================
# run_kallisto.sh -- Submit all Kallisto SLURM jobs from OSD_datasets.txt
# ==============================================================================
#
# Submits datasets sequentially using SLURM --dependency so that the next
# dataset only starts after the previous one finishes. This prevents
# concurrent downloads from overwhelming disk/scratch quotas.
#
# Run this AFTER setup_datasets.py has created the per-dataset directories.
# Usage: bash run_kallisto.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.env" 2>/dev/null || true

DATASETS_FILE="OSD_datasets.txt"

if [ ! -f "$DATASETS_FILE" ]; then
    echo "ERROR: $DATASETS_FILE not found."
    echo "Run setup_datasets.py first to generate it."
    exit 1
fi

SCRATCH_BASE="${KALLISTO_SCRATCH:-/global/scratch/users/${USER}/kallisto_tmp}"
mkdir -p "${SCRATCH_BASE}"

echo "============================================"
echo "Submitting Kallisto SLURM jobs"
echo "============================================"
echo "Scratch base: ${SCRATCH_BASE}"
echo ""

submitted=0
failed=0
prev_jobid=""

for osd in $(cat "$DATASETS_FILE"); do
    echo "--- ${osd} ---"

    slurm_dir="${osd}/proc_scripts/02-kallisto_counts"
    slurm_file=$(ls "${slurm_dir}"/02-kallisto_counts*.slurm 2>/dev/null | head -1)

    if [ -z "$slurm_file" ]; then
        echo "  WARNING: No SLURM script found in ${slurm_dir}. Skipping."
        failed=$((failed + 1))
        continue
    fi

    cd "${slurm_dir}"

    if [ -n "$prev_jobid" ]; then
        echo "  Submitting: $(basename "$slurm_file") (after job ${prev_jobid})"
        result=$(sbatch --dependency=afterany:${prev_jobid} "$(basename "$slurm_file")" 2>&1)
    else
        echo "  Submitting: $(basename "$slurm_file")"
        result=$(sbatch "$(basename "$slurm_file")" 2>&1)
    fi

    cd ../../../

    jobid=$(echo "$result" | grep -oP '\d+$')
    if [ -n "$jobid" ]; then
        echo "  $result"
        prev_jobid="$jobid"
        submitted=$((submitted + 1))
    else
        echo "  ERROR: sbatch failed: $result"
        failed=$((failed + 1))
    fi

    echo ""
done

echo "============================================"
echo "Done. Submitted: ${submitted}, Failed: ${failed}"
echo "Datasets run sequentially via SLURM dependencies."
echo "Monitor with: squeue -u \$USER"
echo "============================================"
