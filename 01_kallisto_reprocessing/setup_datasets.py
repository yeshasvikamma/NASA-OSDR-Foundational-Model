#!/usr/bin/env python3
"""
setup_datasets.py

Takes a CSV file structured like all_mouse_datasets.csv and:
1. Identifies duplicate vs unique id.accession values
2. Creates directory structures for unique datasets
3. Calls the OSDR API to get sample metadata for each dataset
4. Creates samples.txt files
5. Flags datasets with metadata inconsistencies
6. Copies and modifies appropriate template SLURM scripts into each dataset directory

Usage:
    python3 setup_datasets.py <input_csv> [--template-dir ./template_scripts] [--make-dirs-script ./make_dirs.sh]
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import urllib.error
from collections import Counter


# ─────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Set up dataset directory structures and scripts.")
    parser.add_argument("input_csv", help="Path to the input CSV file (e.g. all_mouse_datasets.csv)")
    parser.add_argument(
        "--template-dir",
        default="./template_scripts",
        help="Path to the template_scripts directory (default: ./template_scripts)",
    )
    parser.add_argument(
        "--make-dirs-script",
        default="./make_dirs.sh",
        help="Path to the make_dirs.sh script (default: ./make_dirs.sh)",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# Read input CSV
# ─────────────────────────────────────────────
def read_input_csv(path):
    """
    Returns a list of dicts with keys from the header row.
    Strips surrounding quotes and whitespace from all values.
    """
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {k.strip(): v.strip().strip('"') for k, v in row.items()}
            rows.append(cleaned)
    return rows


# ─────────────────────────────────────────────
# Identify duplicates
# ─────────────────────────────────────────────
def identify_duplicates(rows):
    counts = Counter(r["id.accession"] for r in rows)
    unique = [ds for ds, cnt in counts.items() if cnt == 1]
    duplicates = [ds for ds, cnt in counts.items() if cnt > 1]
    return sorted(unique), sorted(duplicates)


# ─────────────────────────────────────────────
# Create directory structure via make_dirs.sh
# ─────────────────────────────────────────────
def create_dirs(dataset, make_dirs_script):
    env = os.environ.copy()
    env["dataset"] = dataset
    result = subprocess.run(["bash", make_dirs_script], env=env)
    if result.returncode != 0:
        print(f"  WARNING: make_dirs.sh returned non-zero exit code for {dataset}")


# ─────────────────────────────────────────────
# Fetch sample metadata from OSDR API
# ─────────────────────────────────────────────
def fetch_samples(dataset):
    url = (
        f"https://visualization.osdr.nasa.gov/biodata/api/metadata/"
        f"?id.accession={dataset}"
        f"&id.assay%20name=/rna-seq/"
        f"&assay.parameter%20value.library%20layout"
        f"&assay.parameter%20value.Spike-in%20Quality%20Control"
        f"&format=csv"
    )
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            content = resp.read().decode("utf-8")
        return content
    except urllib.error.URLError as e:
        print(f"  WARNING: Could not fetch API data for {dataset}: {e}")
        return None


# ─────────────────────────────────────────────
# Parse API CSV response
# Row 0: #id row (ignored)
# Row 1: column headers
# Row 2+: data
# ─────────────────────────────────────────────
def parse_api_csv(content):
    lines = content.splitlines()
    # Find the header row (second row, index 1)
    if len(lines) < 2:
        return [], []

    reader = csv.DictReader(lines[1:])  # use row index 1 as header
    rows = []
    for row in reader:
        cleaned = {k.strip().strip('"'): v.strip().strip('"') for k, v in row.items()}
        rows.append(cleaned)
    return rows


def find_column(row, candidates):
    """Case-insensitive column lookup among candidate substrings."""
    for key in row:
        key_lower = key.lower()
        for c in candidates:
            if c.lower() in key_lower:
                return key
    return None


# ─────────────────────────────────────────────
# Check metadata consistency across samples
# ─────────────────────────────────────────────
def check_meta_consistency(api_rows):
    """Returns True if all samples have consistent library layout and spike-in values."""
    if not api_rows:
        return True

    layout_col = find_column(api_rows[0], ["library layout"])
    spikein_col = find_column(api_rows[0], ["spike-in", "spike in"])

    layouts = set()
    spikeins = set()
    for row in api_rows:
        if layout_col:
            layouts.add(row.get(layout_col, "").lower())
        if spikein_col:
            spikeins.add(row.get(spikein_col, "").lower())

    return len(layouts) <= 1 and len(spikeins) <= 1


# ─────────────────────────────────────────────
# Determine which SLURM template to use
# Based on input CSV values (not API values)
# ─────────────────────────────────────────────
def determine_template(library_layout, spike_in):
    layout = library_layout.strip().lower()
    spike = spike_in.strip().lower()
    has_spike = "spike-in" in spike or "spike in" in spike

    if layout == "single" and has_spike:
        return "02-kallisto_counts_SE_wERCC.slurm"
    elif layout == "paired" and has_spike:
        return "02-kallisto_counts_PE_wERCC.slurm"
    elif layout == "single" and not has_spike:
        return "02-kallisto_counts_SE_noERCC.slurm"
    elif layout == "paired" and not has_spike:
        return "02-kallisto_counts_PE_noERCC.slurm"
    else:
        return None


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    args = parse_args()

    print(f"Reading input CSV: {args.input_csv}")
    rows = read_input_csv(args.input_csv)

    unique_datasets, duplicate_datasets = identify_duplicates(rows)

    print(f"\nFound {len(unique_datasets)} unique datasets and {len(duplicate_datasets)} duplicate datasets.")

    # Write duplicate_datasets.txt
    with open("duplicate_datasets.txt", "w") as f:
        for ds in duplicate_datasets:
            f.write(ds + "\n")
    print(f"Wrote duplicate_datasets.txt ({len(duplicate_datasets)} entries)")

    # Build a lookup from id.accession -> first row (for unique datasets)
    dataset_meta = {}
    for row in rows:
        ds = row["id.accession"]
        if ds not in dataset_meta:
            dataset_meta[ds] = row

    meta_issues = []

    # Process unique datasets only
    all_datasets_to_process = unique_datasets  # dir creation only for unique
    # But we fetch samples and check meta for all unique datasets
    for dataset in unique_datasets:
        print(f"\n── {dataset} ──")

        # 1. Create directory structure
        print(f"  Creating directories...")
        create_dirs(dataset, args.make_dirs_script)

        # 2. Fetch sample metadata from API
        print(f"  Fetching sample metadata from API...")
        content = fetch_samples(dataset)

        samples = []
        if content:
            api_rows = parse_api_csv(content)
            if api_rows:
                # Find sample name column
                sample_col = find_column(api_rows[0], ["sample name"])
                if sample_col:
                    samples = [r[sample_col] for r in api_rows if r.get(sample_col, "").strip()]
                else:
                    print(f"  WARNING: Could not find 'sample name' column in API response for {dataset}")

                # Check metadata consistency
                if not check_meta_consistency(api_rows):
                    print(f"  WARNING: Metadata inconsistency detected for {dataset}")
                    meta_issues.append(dataset)
            else:
                print(f"  WARNING: No data rows returned from API for {dataset}")
        else:
            print(f"  WARNING: No content from API for {dataset}")

        # 3. Write samples.txt
        samples_path = os.path.join(dataset, "proc_scripts", "samples.txt")
        os.makedirs(os.path.dirname(samples_path), exist_ok=True)
        with open(samples_path, "w") as f:
            for s in samples:
                f.write(s + "\n")
        print(f"  Wrote {len(samples)} samples to {samples_path}")

        # 4. Copy ens_genelevel_biot.r (unmodified)
        r_src = os.path.join(args.template_dir, "ens_genelevel_biot.r")
        r_dst = os.path.join(dataset, "proc_scripts", "02-kallisto_counts", "ens_genelevel_biot.r")
        os.makedirs(os.path.dirname(r_dst), exist_ok=True)
        if os.path.exists(r_src):
            shutil.copy2(r_src, r_dst)
            print(f"  Copied ens_genelevel_biot.r")
        else:
            print(f"  WARNING: {r_src} not found")

        # 5. Determine and copy/modify SLURM template
        meta_row = dataset_meta.get(dataset, {})
        library_layout = meta_row.get("assay.parameter value.library layout", "")
        spike_in = meta_row.get("assay.parameter value.spike-in quality control", "")

        template_name = determine_template(library_layout, spike_in)
        if template_name:
            slurm_src = os.path.join(args.template_dir, template_name)
            slurm_dst = os.path.join(dataset, "proc_scripts", "02-kallisto_counts", template_name)
            if os.path.exists(slurm_src):
                shutil.copy2(slurm_src, slurm_dst)
                # Replace placeholders
                num_samples = len(samples)
                with open(slurm_dst, "r") as f:
                    content_slurm = f.read()
                content_slurm = content_slurm.replace("num_samples", str(num_samples))
                content_slurm = content_slurm.replace("OSD-id", dataset)
                with open(slurm_dst, "w") as f:
                    f.write(content_slurm)
                print(f"  Copied and modified {template_name} (num_samples={num_samples}, dataset={dataset})")
            else:
                print(f"  WARNING: Template {slurm_src} not found")
        else:
            print(f"  WARNING: Could not determine SLURM template for {dataset} (layout='{library_layout}', spike-in='{spike_in}')")

    # Write datasets_w_meta_issues.txt
    with open("datasets_w_meta_issues.txt", "w") as f:
        for ds in meta_issues:
            f.write(ds + "\n")
    print(f"\nWrote datasets_w_meta_issues.txt ({len(meta_issues)} entries)")

    # Write OSD_datasets.txt
    with open("OSD_datasets.txt", "w") as f:
        for ds in unique_datasets:
            f.write(ds + "\n")
    print(f"Wrote OSD_datasets.txt ({len(unique_datasets)} entries)")

    print("\nDone.")


if __name__ == "__main__":
    main()
