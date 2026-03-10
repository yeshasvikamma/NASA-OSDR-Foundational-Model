# Phase 2: ARCHS4 Data Preparation

Download ARCHS4 bulk RNA-seq data and preprocess it into the format expected by the BulkFormer model.

## What This Phase Does

1. **Downloads** ARCHS4 v2.3 HDF5 files (~10-30 GB each) containing gene-level counts for all human and mouse samples
2. **Downloads** gene metadata from the BioFM repository (ortholog mappings, exon lengths)
3. **Preprocesses** the expression data:
   - Maps mouse genes to human orthologs (one-to-one)
   - Filters to protein-coding ortholog genes
   - Normalizes: raw counts -> length-normalize by exon length -> TPM -> log(TPM+1) -> z-score per sample
   - Splits into train/val/test sets (70/15/15)
4. **Saves** parquet files matching BulkFormer's expected input format
5. **(Optional)** Generates ESM2 gene identity embeddings and KNN gene-gene coexpression graph

## Prerequisites

- `config.env` filled in (one directory up)
- `biofm` conda environment installed (`conda env create -f ../environments/biofm.yml`)
- Internet access for downloading (~40 GB total)
- At least 64 GB RAM for preprocessing (ARCHS4 matrices are large)

## Steps

### Step 1: Download ARCHS4 data + gene metadata

```bash
# Option A: Submit as SLURM job (if compute nodes have internet)
sbatch download_archs4.slurm

# Option B: Run directly on login node
bash download_archs4.sh
```

This downloads to `../data/archs4/` and `../data/ensembl/`.

### Step 2: Preprocess

```bash
# Option A: Submit as SLURM job
sbatch preprocess_archs4.slurm

# Option B: Run interactively (needs ~64 GB RAM)
conda activate biofm
python preprocess_archs4.py --data-dir ../data
```

### Step 3: (Optional) Generate ESM2 embeddings

ESM2 embeddings give each gene a biological identity based on its protein sequence. For a quick test, the preprocessing script generates placeholder embeddings. For production:

```bash
# Run with ESM2 (requires fair-esm package + protein FASTA file)
python preprocess_archs4.py --data-dir ../data
```

Or download pre-computed embeddings from the BioFM repository.

## Output Files

After preprocessing, the `../data/` directory will contain:

```
data/
├── archs4/
│   ├── human_gene_v2.3.h5              # Raw ARCHS4 human data
│   ├── mouse_gene_v2.3.h5              # Raw ARCHS4 mouse data
│   └── processed/
│       ├── train_expr_logtpm_short.parquet  # Training set
│       ├── val_expr_logtpm_short.parquet    # Validation set
│       ├── test_expr_logtpm_short.parquet   # Test set
│       └── gene_order.csv                   # Canonical gene ordering
├── embeddings/
│   └── esm2_t6_8M_UR50D_gene_embeddings.pt  # Gene identity embeddings
├── ensembl/
│   ├── orthologs_one2one.txt               # Mouse-human ortholog mapping
│   ├── gencode_v49_gene_exon_lengths.csv   # Human exon lengths
│   ├── gencode_v49_mouse_gene_exon_lengths.csv  # Mouse exon lengths
│   └── protein_coding_ortholog_genes.txt   # Canonical gene list
└── graph/
    └── edge_index_top20.pt                 # KNN gene-gene graph edges
```

## Normalization Details

The normalization follows BioFM's pipeline exactly:

1. **Length-normalize**: `counts / exon_length_kb` (using merged exon lengths from GENCODE v49)
2. **TPM**: `(length_normalized / sum_per_sample) * 1e6`
3. **Log**: `log(TPM + 1)` (natural log)
4. **Z-score**: `(x - mean) / std` per sample, filling NaN with 0
