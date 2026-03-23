#!/bin/bash
#
#SBATCH --job-name="kallisto_quant"
#SBATCH --output=out_logs/%A_%a.out
#SBATCH --partition="savio4_htc"
#SBATCH --account="pc_disconasabio"
#SBATCH --mem=18000
#SBATCH --cpus-per-task=2
#SBATCH --time=4:00:00
#SBATCH --array=1-num_samples%4
#
#SBATCH --mail-user=yeshasvikamma@berkeley.edu
#SBATCH --mail-type=END

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../../config.env" 2>/dev/null || true

module load anaconda3/2024.02-1-11.4
source activate "${KALLISTO_ENV:-kallisto_v0.51.1}"

if ! command -v kallisto &>/dev/null; then
    echo "ERROR: kallisto not found -- conda environment activation failed. Exiting."
    exit 1
fi

echo "$SLURM_JOB_NAME"
echo ""

start=$(date +%s)
echo "start time: $start"
echo "$HOSTNAME"
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo ""

dataset=OSD-id
out_dir=../../02-kallisto_counts
REF=../../../kallisto_indices

sample=$(cat ../samples.txt | sed -n ${SLURM_ARRAY_TASK_ID}p)

SCRATCH_WORK="${KALLISTO_SCRATCH:-/global/scratch/users/${USER}/kallisto_tmp}/${dataset}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${SCRATCH_WORK}/fastq"
mkdir -p "${SCRATCH_WORK}/counts"

in_dir="${SCRATCH_WORK}/fastq"
local_out="${SCRATCH_WORK}/counts"

cleanup() {
    echo "Cleaning up scratch: ${SCRATCH_WORK}"
    rm -rf "${SCRATCH_WORK}"
}
trap cleanup EXIT

echo ""
echo "Dataset: ${dataset}"
echo "SAMPLE: ${sample}"
echo "Scratch: ${SCRATCH_WORK}"
echo ""

echo ""
echo "copy trimmed fastq files from s3"
echo ""

version=$(aws s3 ls --no-sign-request --region us-west-2 s3://nasa-osdr/${dataset}/ | grep -oP 'version-\d+' | sort -t'-' -k2 -n | tail -1)
echo "Using dataset version: ${version}"
echo ""

if [ -z "$version" ]; then
    echo "ERROR: Could not determine version for dataset ${dataset}. Exiting."
    exit 1
fi

aws s3 sync --no-sign-request --region us-west-2 s3://nasa-osdr/${dataset}/${version}/rna_seq/ ${in_dir}/ --exclude "*" --include "*${sample}_*trimmed.fastq.gz" --no-progress --cli-connect-timeout 30 --cli-read-timeout 30

echo ""
echo "check for expected number of files"
echo ""

max_attempts=3
attempt=1
download_success=false

while [ $attempt -le $max_attempts ]; do
    echo "File check attempt ${attempt} of ${max_attempts}"

    r1=$(find "${in_dir}/" -name "*${sample}_*trimmed.fastq.gz" | head -1)

    if [ -z "$r1" ]; then
        echo "ERROR: Could not find trimmed file for sample ${sample}"
    else
        echo "Found R1: $r1"
        echo "Running gzip integrity check..."
        gzip -t "$r1" && r1_ok=true || r1_ok=false

        if [ "$r1_ok" = true ]; then
            echo "Integrity check passed for the file"
            download_success=true
            break
        else
            echo "ERROR: Integrity check failed for one or more files"
        fi
    fi

    if [ $attempt -lt $max_attempts ]; then
        echo "Re-attempting download..."
        aws s3 sync --no-sign-request --region us-west-2 s3://nasa-osdr/${dataset}/${version}/rna_seq/ ${in_dir}/ --exclude "*" --include "*${sample}_*trimmed.fastq.gz" --no-progress --cli-connect-timeout 30 --cli-read-timeout 30
    fi

    attempt=$(( attempt + 1 ))
done

if [ "$download_success" = false ]; then
    echo "ERROR: Files for sample ${sample} could not be successfully downloaded and verified after ${max_attempts} attempts. Exiting."
    exit 1
fi


echo ""
echo "rename trimmed fastq files"
echo ""

find "${in_dir}/" -name "*${sample}_*trimmed.fastq.gz" | while read -r filepath; do
    dir_part=$(dirname "$filepath")
    filename=$(basename "$filepath")
    newname="${filename#*${sample}}"
    newname="${sample}${newname}"
    newname="${newname//_GLbulkRNAseq_trimmed.fastq.gz/_trimmed.fastq.gz}"
    mv "$filepath" "${dir_part}/${newname}"
done


echo ""
echo "kallisto version: "
kallisto version
echo ""

echo "run kallisto count command"
echo ""

call="kallisto quant -t 2 \
	-i ${REF}/Mmus-107_cdna_ncrna_ERCC92_index.idx \
	--single \
	-l 200 \
	-s 20 \
	-o ${local_out} \
	${in_dir}/${sample}_trimmed.fastq.gz"


echo $call
echo ""
eval $call

echo ""
echo "check kallisto output"
echo ""

expected_lines=146283

if [ ! -f "${local_out}/abundance.tsv" ]; then
    echo "ERROR: abundance.tsv not found for sample ${sample}. Exiting."
    exit 1
fi
actual_lines=$(wc -l < "${local_out}/abundance.tsv")
if [ "$actual_lines" -ne "$expected_lines" ]; then
    echo "ERROR: abundance.tsv has ${actual_lines} lines but expected ${expected_lines}. Exiting."
    exit 1
else
    echo "abundance.tsv found and contains expected number of lines (${actual_lines})"
fi

echo ""
echo "Run the R script to create the ens_gene_biot_abundance.tsv output file"
echo ""


echo ""
echo "R version: "
R --version
echo ""

Rscript --vanilla ens_genelevel_biot.r ${REF}/mapping_Mmus-107_and_ERCC.rda ${local_out}/abundance.tsv ${local_out}/ens_gene_biot_abundance.tsv

echo ""
echo "check Rscript output"
echo ""

expected_lines=53604

if [ ! -f "${local_out}/ens_gene_biot_abundance.tsv" ]; then
    echo "ERROR: ens_gene_biot_abundance.tsv not found for sample ${sample}. Exiting."
    exit 1
fi
actual_lines=$(wc -l < "${local_out}/ens_gene_biot_abundance.tsv")
if [ "$actual_lines" -ne "$expected_lines" ]; then
    echo "ERROR: ens_gene_biot_abundance.tsv has ${actual_lines} lines but expected ${expected_lines}. Exiting."
    exit 1
else
    echo "ens_gene_biot_abundance.tsv found and contains expected number of lines (${actual_lines})"
fi


echo ""
echo "Copy final results to home directory"
echo ""

mkdir -p ${out_dir}/${sample}
cp ${local_out}/abundance.tsv ${out_dir}/${sample}/
cp ${local_out}/ens_gene_biot_abundance.tsv ${out_dir}/${sample}/


echo "DONE"

echo ""
end=$(date +%s)
echo "end time: $end"
runtime_s=$(( end - start ))
echo "total run time(s): $runtime_s"
sec_per_min=60
sec_per_hr=3600
runtime_m=$(echo "scale=2; $runtime_s / $sec_per_min;" | bc)
echo "total run time(m): $runtime_m"
runtime_h=$(echo "scale=2; $runtime_s / $sec_per_hr;" | bc)
echo "total run time(h): $runtime_h"

echo ""
echo "slurm job ID: ${SLURM_JOB_ID}"
