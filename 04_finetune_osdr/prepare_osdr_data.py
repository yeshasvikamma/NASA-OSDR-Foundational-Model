#!/usr/bin/env python3
"""
prepare_osdr_data.py -- Convert Kallisto gene-level counts to BulkFormer input format.

Reads the ens_gene_biot_abundance.tsv files produced by Phase 1 (Kallisto
reprocessing), applies the same normalization as the ARCHS4 pre-training data,
maps genes to the canonical gene set, and outputs a parquet file ready for
fine-tuning.

Usage:
    python prepare_osdr_data.py \
        --kallisto-dir ../01_kallisto_reprocessing \
        --data-dir ../data \
        --output-dir ../data/osdr/processed

Pipeline:
    1. Discover all ens_gene_biot_abundance.tsv files under the Kallisto output dirs
    2. Load each file, extract gene-level counts
    3. Map Ensembl gene IDs / symbols to the canonical gene set used during pre-training
    4. Normalize: counts -> length-normalize -> TPM -> log(TPM+1) -> z-score
    5. Combine all samples into a single matrix
    6. Save as parquet (same format as pre-training data)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Kallisto output to BulkFormer input format."
    )
    parser.add_argument(
        "--kallisto-dir",
        default="../01_kallisto_reprocessing",
        help="Root of the Kallisto reprocessing directory",
    )
    parser.add_argument(
        "--data-dir",
        default="../data",
        help="Root data directory (for gene metadata / exon lengths)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <data-dir>/osdr/processed)",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Fraction of samples for validation (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def discover_abundance_files(kallisto_dir):
    """Find all ens_gene_biot_abundance.tsv files under Kallisto output directories."""
    kallisto_path = Path(kallisto_dir)
    files = sorted(kallisto_path.rglob("ens_gene_biot_abundance.tsv"))
    print(f"  Found {len(files)} abundance files")
    return files


def load_abundance_file(path):
    """Load a single ens_gene_biot_abundance.tsv file."""
    df = pd.read_csv(path, sep="\t")
    # Expected columns: ensembl_gene, symbol, biotype, counts
    return df


def load_canonical_gene_order(data_dir):
    """Load the canonical gene order used during pre-training."""
    data_path = Path(data_dir)

    # Try the preprocessed gene order first
    gene_order_path = data_path / "archs4" / "processed" / "gene_order.csv"
    if gene_order_path.exists():
        df = pd.read_csv(gene_order_path)
        return df["gene"].tolist()

    # Fallback: protein-coding ortholog genes
    pc_path = data_path / "ensembl" / "protein_coding_ortholog_genes.txt"
    if pc_path.exists():
        with open(pc_path) as f:
            return [line.strip() for line in f if line.strip()]

    return None


def load_exon_lengths(data_dir):
    """Load exon lengths for TPM normalization."""
    # Try mouse exon lengths first (OSDR data is mouse)
    mouse_path = Path(data_dir) / "ensembl" / "gencode_v49_mouse_gene_exon_lengths.csv"
    human_path = Path(data_dir) / "ensembl" / "gencode_v49_gene_exon_lengths.csv"

    for path in [mouse_path, human_path]:
        if path.exists():
            print(f"  Loading exon lengths from {path}")
            return pd.read_csv(path)

    print("  WARNING: No exon length file found")
    return None


def normalize_counts(counts_df, exon_lengths_df=None):
    """
    Normalize raw counts to log(TPM+1) then z-score per sample.

    Same pipeline as preprocess_archs4.py to ensure consistency.
    """
    values = counts_df.values.astype(np.float64)

    if exon_lengths_df is not None:
        # Build symbol -> length mapping
        if "gene_name" in exon_lengths_df.columns:
            name_col = "gene_name"
        elif "symbol" in exon_lengths_df.columns:
            name_col = "symbol"
        else:
            name_col = exon_lengths_df.columns[0]

        if "merged_exon_length" in exon_lengths_df.columns:
            len_col = "merged_exon_length"
        elif "length" in exon_lengths_df.columns:
            len_col = "length"
        else:
            len_col = exon_lengths_df.columns[1]

        length_map = dict(zip(exon_lengths_df[name_col], exon_lengths_df[len_col]))
        lengths = np.array([length_map.get(g, np.nan) for g in counts_df.columns])
        valid = ~np.isnan(lengths)

        if valid.sum() < len(lengths):
            print(f"    {valid.sum()}/{len(lengths)} genes have exon lengths")

        counts_df = counts_df.iloc[:, valid]
        values = counts_df.values.astype(np.float64)
        lengths = lengths[valid]
        length_kb = lengths / 1000.0
        rpk = values / length_kb[np.newaxis, :]
    else:
        rpk = values

    # TPM
    rpk_sum = rpk.sum(axis=1, keepdims=True)
    rpk_sum[rpk_sum == 0] = 1
    tpm = rpk / rpk_sum * 1e6

    # log(TPM + 1)
    log_tpm = np.log1p(tpm)

    # z-score per sample
    means = log_tpm.mean(axis=1, keepdims=True)
    stds = log_tpm.std(axis=1, keepdims=True)
    stds[stds == 0] = 1
    z = (log_tpm - means) / stds

    return pd.DataFrame(z, index=counts_df.index, columns=counts_df.columns)


def main():
    args = parse_args()

    data_dir = Path(args.data_dir).resolve()
    kallisto_dir = Path(args.kallisto_dir).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "osdr" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Prepare OSDR Data for Fine-tuning")
    print("=" * 70)
    print(f"  Kallisto dir: {kallisto_dir}")
    print(f"  Data dir:     {data_dir}")
    print(f"  Output dir:   {output_dir}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Discover abundance files
    # ------------------------------------------------------------------
    print("[1/5] Discovering Kallisto output files ...")
    abundance_files = discover_abundance_files(kallisto_dir)
    if not abundance_files:
        print("ERROR: No ens_gene_biot_abundance.tsv files found.")
        print(f"  Searched in: {kallisto_dir}")
        print("  Make sure Phase 1 (Kallisto reprocessing) has completed.")
        sys.exit(1)
    print()

    # ------------------------------------------------------------------
    # Step 2: Load all abundance files into a single matrix
    # ------------------------------------------------------------------
    print("[2/5] Loading abundance files ...")
    sample_data = {}

    for fpath in abundance_files:
        # Extract sample name from path: .../OSD-XXX/02-kallisto_counts/<sample>/ens_gene_biot_abundance.tsv
        sample_name = fpath.parent.name
        dataset_name = None
        for part in fpath.parts:
            if part.startswith("OSD-"):
                dataset_name = part
                break

        sample_id = f"{dataset_name}_{sample_name}" if dataset_name else sample_name

        df = load_abundance_file(fpath)
        # Use gene symbol as index, counts as value
        gene_counts = dict(zip(df["symbol"], df["counts"]))
        sample_data[sample_id] = gene_counts

    counts_df = pd.DataFrame(sample_data).T.fillna(0)
    print(f"  Combined matrix: {counts_df.shape} (samples x genes)")
    print()

    # ------------------------------------------------------------------
    # Step 3: Map to canonical gene set
    # ------------------------------------------------------------------
    print("[3/5] Mapping to canonical gene set ...")
    canonical_genes = load_canonical_gene_order(data_dir)

    if canonical_genes is not None:
        common = sorted(set(counts_df.columns) & set(canonical_genes))
        missing = len(canonical_genes) - len(common)
        print(f"  Canonical genes: {len(canonical_genes)}")
        print(f"  Overlap with OSDR: {len(common)}")
        print(f"  Missing (will be zero-filled): {missing}")

        # Reindex to canonical gene order, filling missing with 0
        counts_df = counts_df.reindex(columns=canonical_genes, fill_value=0)
    else:
        print("  WARNING: No canonical gene order found. Using all genes from Kallisto output.")
        print("  (Pre-training gene order will be applied during fine-tuning if available)")
    print()

    # ------------------------------------------------------------------
    # Step 4: Normalize
    # ------------------------------------------------------------------
    print("[4/5] Normalizing ...")
    exon_lengths = load_exon_lengths(data_dir)
    norm_df = normalize_counts(counts_df, exon_lengths)
    print(f"  Normalized matrix: {norm_df.shape}")
    print()

    # ------------------------------------------------------------------
    # Step 5: Split and save
    # ------------------------------------------------------------------
    print("[5/5] Splitting and saving ...")
    np.random.seed(args.seed)

    n = len(norm_df)
    val_size = max(1, int(n * args.val_frac))
    indices = np.random.permutation(n)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_df = norm_df.iloc[train_idx]
    val_df = norm_df.iloc[val_idx]

    print(f"  Train: {train_df.shape}")
    print(f"  Val:   {val_df.shape}")

    # Save transposed (genes as rows) to match BulkFormer format
    train_df.T.to_parquet(output_dir / "train_expr_logtpm_short.parquet")
    val_df.T.to_parquet(output_dir / "val_expr_logtpm_short.parquet")

    # Save sample metadata
    meta = pd.DataFrame({"sample_id": norm_df.index.tolist()})
    meta["dataset"] = meta["sample_id"].str.extract(r"(OSD-\d+)")
    meta["split"] = "train"
    meta.loc[val_idx, "split"] = "val"
    meta.to_csv(output_dir / "sample_metadata.csv", index=False)

    print()
    print("=" * 70)
    print("OSDR data preparation complete!")
    print(f"  Output files:")
    print(f"    {output_dir / 'train_expr_logtpm_short.parquet'}")
    print(f"    {output_dir / 'val_expr_logtpm_short.parquet'}")
    print(f"    {output_dir / 'sample_metadata.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
