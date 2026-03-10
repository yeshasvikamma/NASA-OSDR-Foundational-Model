# Phase 1: OSDR Kallisto Reprocessing

Reprocess OSDR mouse RNA-seq datasets through Kallisto to produce gene-level expression counts. These counts are used later in Phase 4 for fine-tuning the BulkFormer model.

## Test Datasets

The file `test_mouse_datasets.csv` contains 4 mouse OSDR datasets for testing:

| Dataset | Library Layout | ERCC Spike-in | Template Used |
|---------|---------------|---------------|---------------|
| OSD-47  | Single-end    | No            | `02-kallisto_counts_SE_noERCC.slurm` |
| OSD-104 | Paired-end    | No            | `02-kallisto_counts_PE_noERCC.slurm` |
| OSD-202 | Single-end    | Yes           | `02-kallisto_counts_SE_wERCC.slurm` |
| OSD-244 | Paired-end    | Yes           | `02-kallisto_counts_PE_wERCC.slurm` |

## Prerequisites

Before running, make sure you have:

1. **AWS CLI** installed (`which aws`)
2. **Conda** installed (`which conda`)
3. **`kallisto_indices/`** directory copied to this folder (provided separately -- contains `.idx` and `.rda` files)
4. **`config.env`** filled in with your HPC settings (one directory up)

## Setup Steps

### Step 1: Install the Kallisto conda environment

```bash
conda env create -f kallisto_v0.51.1.yml
```

### Step 2: Make scripts executable

```bash
chmod u+rx make_dirs.sh
chmod u+rx setup_datasets.py
```

### Step 3: Edit SLURM template placeholders

The SLURM templates in `template_scripts/` have placeholder values that need your HPC settings. Open each of the 4 `.slurm` files and update:

```
#SBATCH --partition=normal      <-- replace "normal" with your partition
#SBATCH --mail-user=myemail     <-- replace "myemail" with your email
#SBATCH --mem=18000             <-- can reduce to 12000 if needed
#SBATCH --cpus-per-task=2       <-- can reduce to 1 if needed
```

### Step 4: Run the dataset setup script

This creates per-dataset directory structures, fetches sample metadata from the OSDR API, and generates customized SLURM scripts:

```bash
python setup_datasets.py test_mouse_datasets.csv \
    --template-dir ./template_scripts \
    --make-dirs-script ./make_dirs.sh
```

After this completes, you will see:
- `OSD-47/`, `OSD-104/`, `OSD-202/`, `OSD-244/` directories created
- Each contains `proc_scripts/02-kallisto_counts/` with its SLURM script
- `OSD_datasets.txt` listing all datasets to process
- `duplicate_datasets.txt` and `datasets_w_meta_issues.txt` for QC

### Step 5: Submit all jobs

Use the convenience wrapper:

```bash
bash run_kallisto.sh
```

Or submit manually:

```bash
for osd in $(cat OSD_datasets.txt); do
    echo "Submitting: ${osd}"
    cd ${osd}/proc_scripts/02-kallisto_counts
    sbatch 02-kallisto_counts*.slurm
    cd ../../../
done
```

### Step 6: Monitor jobs

```bash
squeue -u $USER
```

## Output

Each dataset produces per-sample output in `OSD-XXX/02-kallisto_counts/<sample>/`:

- `abundance.tsv` -- Kallisto transcript-level quantification
- `ens_gene_biot_abundance.tsv` -- gene-level counts with Ensembl IDs, symbols, and biotypes

The `ens_gene_biot_abundance.tsv` files are what Phase 4 uses for fine-tuning.

## Pipeline Flow

```
S3 (trimmed FASTQs)
    |
    v
aws s3 sync --> local FASTQ files
    |
    v
kallisto quant --> abundance.tsv (transcript-level)
    |
    v
ens_genelevel_biot.r --> ens_gene_biot_abundance.tsv (gene-level)
    |
    v
Cleanup: remove FASTQ files
```
