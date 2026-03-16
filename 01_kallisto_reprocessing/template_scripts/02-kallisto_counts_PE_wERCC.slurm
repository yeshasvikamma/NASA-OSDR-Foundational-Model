#!/bin/bash
#
#SBATCH --job-name="kallisto_quant"
#SBATCH --output=out_logs/%A_%a.out
#SBATCH --partition=savio3_htc
#SBATCH --account=CHANGE_ME
#SBATCH --cpus-per-task=2
#SBATCH --array=1-num_samples
#
#SBATCH --mail-type=END

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../../../../config.env" 2>/dev/null || true

echo "$SLURM_JOB_NAME"
echo ""

start=$(date +%s)
echo "start time: $start"
echo "$HOSTNAME"
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo ""

module load python
if [ -n "$CONDA_INIT_SCRIPT" ] && [ -f "$CONDA_INIT_SCRIPT" ]; then
    source "$CONDA_INIT_SCRIPT"
fi
conda activate "${KALLISTO_ENV:-kallisto_v0.51.1}"

dataset=OSD-id
in_dir=../../01-TG_Preproc/Fastq
out_dir=../../02-kallisto_counts
REF=../../../kallisto_indices

sample=$(cat ../samples.txt | sed -n ${SLURM_ARRAY_TASK_ID}p)

echo ""
echo "Dataset: ${dataset}"
echo "SAMPLE: ${sample}"
echo ""

echo ""
echo "copy trimmed fastq files from s3"
echo ""

# Get most recent version number
version=$(aws s3 ls --no-sign-request --region us-west-2 s3://nasa-osdr/${dataset}/ | grep -oP 'version-\d+' | sort -t'-' -k2 -n | tail -1)
echo "Using dataset version: ${version}"
echo ""

# Check that the version number was defined
if [ -z "$version" ]; then
    echo "ERROR: Could not determine version for dataset ${dataset}. Exiting."
    exit 1
fi

# Run aws command to download the files
aws s3 sync --no-sign-request --region us-west-2 s3://nasa-osdr/${dataset}/${version}/rna_seq/ ${in_dir}/ --exclude "*" --include "*${sample}_*trimmed.fastq.gz" --no-progress --cli-connect-timeout 30 --cli-read-timeout 30

echo ""
echo "check for expected number of files"
echo ""

max_attempts=3
attempt=1
download_success=false

while [ $attempt -le $max_attempts ]; do
    echo "File check attempt ${attempt} of ${max_attempts}"

    r1=$(find "${in_dir}/" -name "*${sample}_*R1_trimmed.fastq.gz" | head -1)
    r2=$(find "${in_dir}/" -name "*${sample}_*R2_trimmed.fastq.gz" | head -1)

    if [ -z "$r1" ] || [ -z "$r2" ]; then
        echo "ERROR: Could not find R1 and/or R2 files for sample ${sample}"
    else
        echo "Found R1: $r1"
        echo "Found R2: $r2"
        echo "Running gzip integrity check..."
        gzip -t "$r1" && r1_ok=true || r1_ok=false
        gzip -t "$r2" && r2_ok=true || r2_ok=false

        if [ "$r1_ok" = true ] && [ "$r2_ok" = true ]; then
            echo "Integrity check passed for both files"
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
    newname="${newname//_GLbulkRNAseq_R1_/_R1_}"
    newname="${newname//_GLbulkRNAseq_R2_/_R2_}"
    mv "$filepath" "${dir_part}/${newname}"
done


echo ""
echo "kallisto version: "
kallisto version
echo ""

echo "run kallisto count command"
echo ""

mkdir ${out_dir}/${sample}

call="kallisto quant -t 2 \
	-i ${REF}/Mmus-107_cdna_ncrna_ERCC92_index.idx \
	-o ${out_dir}/${sample} \
	${in_dir}/${sample}_R1_trimmed.fastq.gz ${in_dir}/${sample}_R2_trimmed.fastq.gz"


echo $call
echo ""
eval $call

echo ""
echo "check kallisto output"
echo ""

expected_lines=146283

if [ ! -f "${out_dir}/${sample}/abundance.tsv" ]; then
    echo "ERROR: abundance.tsv not found for sample ${sample}. Exiting."
    exit 1
fi
actual_lines=$(wc -l < "${out_dir}/${sample}/abundance.tsv")
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

Rscript --vanilla ens_genelevel_biot.r ${REF}/mapping_Mmus-107_and_ERCC.rda ${out_dir}/${sample}/abundance.tsv ${out_dir}/${sample}/ens_gene_biot_abundance.tsv

echo ""
echo "check Rscript output"
echo ""

expected_lines=53604

if [ ! -f "${out_dir}/${sample}/ens_gene_biot_abundance.tsv" ]; then
    echo "ERROR: ens_gene_biot_abundance.tsv not found for sample ${sample}. Exiting."
    exit 1
fi
actual_lines=$(wc -l < "${out_dir}/${sample}/ens_gene_biot_abundance.tsv")
if [ "$actual_lines" -ne "$expected_lines" ]; then
    echo "ERROR: ens_gene_biot_abundance.tsv has ${actual_lines} lines but expected ${expected_lines}. Exiting."
    exit 1
else
    echo "ens_gene_biot_abundance.tsv found and contains expected number of lines (${actual_lines})"
fi


echo ""
echo "Remove trimmed fastq files"
echo ""

rm ${in_dir}/${sample}_*trimmed.fastq.gz


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
