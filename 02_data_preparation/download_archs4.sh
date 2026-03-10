#!/bin/bash
# ==============================================================================
# download_archs4.sh -- Download ARCHS4 HDF5 gene expression files
# ==============================================================================
#
# Downloads human and mouse gene-level count matrices from ARCHS4.
# Files are ~10-30 GB each. Run on a node with internet access.
#
# Usage: bash download_archs4.sh [output_dir]
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/../config.env"
if [ -f "$CONFIG" ]; then
    source "$CONFIG"
fi

OUTPUT_DIR="${1:-${DATA_DIR:-${SCRIPT_DIR}/../data}/archs4}"
mkdir -p "$OUTPUT_DIR"

ARCHS4_BASE="https://s3.dev.maayanlab.cloud/archs4/files"

HUMAN_FILE="human_gene_v2.3.h5"
MOUSE_FILE="mouse_gene_v2.3.h5"

echo "============================================"
echo "ARCHS4 Data Download"
echo "============================================"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Download human gene counts
if [ -f "${OUTPUT_DIR}/${HUMAN_FILE}" ]; then
    echo "Human file already exists: ${OUTPUT_DIR}/${HUMAN_FILE}"
    echo "  Size: $(du -h "${OUTPUT_DIR}/${HUMAN_FILE}" | cut -f1)"
    echo "  Skipping download. Delete the file to re-download."
else
    echo "Downloading ${HUMAN_FILE} ..."
    wget -c -P "$OUTPUT_DIR" "${ARCHS4_BASE}/${HUMAN_FILE}"
    echo "  Done. Size: $(du -h "${OUTPUT_DIR}/${HUMAN_FILE}" | cut -f1)"
fi

echo ""

# Download mouse gene counts
if [ -f "${OUTPUT_DIR}/${MOUSE_FILE}" ]; then
    echo "Mouse file already exists: ${OUTPUT_DIR}/${MOUSE_FILE}"
    echo "  Size: $(du -h "${OUTPUT_DIR}/${MOUSE_FILE}" | cut -f1)"
    echo "  Skipping download. Delete the file to re-download."
else
    echo "Downloading ${MOUSE_FILE} ..."
    wget -c -P "$OUTPUT_DIR" "${ARCHS4_BASE}/${MOUSE_FILE}"
    echo "  Done. Size: $(du -h "${OUTPUT_DIR}/${MOUSE_FILE}" | cut -f1)"
fi

echo ""

# Download ortholog mapping from BioFM repo
ENSEMBL_DIR="${DATA_DIR:-${SCRIPT_DIR}/../data}/ensembl"
mkdir -p "$ENSEMBL_DIR"

ORTHO_URL="https://raw.githubusercontent.com/alwalt/BioFM/main/utils/orthologs_one2one.txt"
EXON_HUMAN_URL="https://raw.githubusercontent.com/alwalt/BioFM/main/utils/gencode_v49_gene_exon_lengths.csv"
EXON_MOUSE_URL="https://raw.githubusercontent.com/alwalt/BioFM/main/utils/gencode_v49_mouse_gene_exon_lengths.csv"
PC_ORTHO_URL="https://raw.githubusercontent.com/alwalt/BioFM/main/utils/protein_coding_ortholog_genes.txt"

echo "Downloading gene metadata from BioFM repository..."

for url in "$ORTHO_URL" "$EXON_HUMAN_URL" "$EXON_MOUSE_URL" "$PC_ORTHO_URL"; do
    fname=$(basename "$url")
    if [ -f "${ENSEMBL_DIR}/${fname}" ]; then
        echo "  Already exists: ${fname}"
    else
        echo "  Downloading: ${fname}"
        wget -q -P "$ENSEMBL_DIR" "$url"
    fi
done

echo ""
echo "============================================"
echo "Download complete."
echo ""
echo "Files:"
ls -lh "${OUTPUT_DIR}/"
echo ""
echo "Gene metadata:"
ls -lh "${ENSEMBL_DIR}/"
echo "============================================"
