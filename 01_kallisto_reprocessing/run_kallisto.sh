#!/bin/bash
# ==============================================================================
# run_kallisto.sh -- Submit all Kallisto SLURM jobs from OSD_datasets.txt
# ==============================================================================
#
# Run this AFTER setup_datasets.py has created the per-dataset directories.
# Usage: bash run_kallisto.sh
# ==============================================================================

set -euo pipefail

DATASETS_FILE="OSD_datasets.txt"

if [ ! -f "$DATASETS_FILE" ]; then
    echo "ERROR: $DATASETS_FILE not found."
    echo "Run setup_datasets.py first to generate it."
    exit 1
fi

echo "============================================"
echo "Submitting Kallisto SLURM jobs"
echo "============================================"
echo ""

submitted=0
failed=0

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
    echo "  Submitting: $(basename "$slurm_file")"
    sbatch "$(basename "$slurm_file")"
    cd ../../../

    submitted=$((submitted + 1))
    echo ""
done

echo "============================================"
echo "Done. Submitted: ${submitted}, Skipped: ${failed}"
echo "Monitor with: squeue -u \$USER"
echo "============================================"
