library("plyr")

# Get command line arguments
args <- commandArgs(TRUE)

mapping_file <- args[1]      # First argument: path to mapping .rda file
f <- args[2]                 # Second argument: path to abundance.tsv file
biot_output_file <- args[3]  # Third argument: path to ens_gene_biot_abundance.tsv file

# Load the mapping file
load(mapping_file)

# Rename to 'cb' to match original script
cb <- mapping

# Read abundance file
abu <- read.table(f, sep="\t", stringsAsFactors=F, header=TRUE)

# Get gene symbols, gene IDs, and gene biotypes
ugene <- cb[,2]      # gene_symbol
ugeneid <- cb[,1]    # gene_id
ubiotype <- cb[,4]   # gene_biotype

# Match transcripts to genes
m3 <- match(abu[,1], cb[,3])

# Combine abundance with gene symbols, gene IDs, and biotypes
cco <- cbind(abu, ugene[m3], ugeneid[m3], ubiotype[m3])

# Extract gene symbol (column 6), gene_id (column 7), gene_biotype (column 8), and estimated counts (column 4)
co <- cco[,c(6,7,8,4)]
co[,1] <- as.character(co[,1])
co[,2] <- as.character(co[,2])
co[,3] <- as.character(co[,3])

# Create data frame
df <- data.frame(gene = co[,1], gene_id = co[,2], gene_biotype = co[,3], value = as.numeric(co[,4]))

# Remove NA values (transcripts not found in mapping)
#df <- df[!is.na(df$gene),]

# Sum counts by gene_id (the unique identifier)
dd_symbol <- ddply(df, .(gene_id, gene), summarize, sum=sum(value), number=length(gene))

# SORT by gene_id (ensembl_gene)
dd_symbol <- dd_symbol[order(dd_symbol$gene_id),]

# Format output for gene_abundance.tsv (gene_symbol and counts only)
gene_output <- data.frame(gene_symbol = dd_symbol[,2], counts = dd_symbol[,3])

# Write gene-level counts to specified output file (gene_symbol, counts)
#write.table(gene_output, file=output_file, quote=F, col.names=F, row.names=F, sep="\t")

# Sum counts by BOTH gene_symbol AND gene_id for ens_gene_abundance.tsv
dd_ensembl <- ddply(df, .(gene_id, gene), summarize, sum=sum(value), number=length(gene))

# SORT by gene_id (ensembl_gene)
dd_ensembl <- dd_ensembl[order(dd_ensembl$gene_id),]

# Create output for ens_gene_abundance.tsv (gene_id, gene_symbol, counts)
ens_output <- data.frame(gene_id = dd_ensembl[,1], gene_symbol = dd_ensembl[,2], counts = dd_ensembl[,3])

# Write the Ensembl gene abundance file
#write.table(ens_output, file=ens_output_file, quote=F, col.names=TRUE, row.names=F, sep="\t")

# For biotype file: use the SAME aggregation as ens_gene_abundance.tsv (by gene and gene_id only)
# Then add biotype by taking the first biotype for each gene_id
dd_biotype <- ddply(df, .(gene_id, gene), summarize, 
                    sum=sum(value), 
                    number=length(gene),
                    gene_biotype=gene_biotype[1])  # Take first biotype for this gene_id

# SORT by gene_id (ensembl_gene)
dd_biotype <- dd_biotype[order(dd_biotype$gene_id),]

# Create output for ens_gene_biot_abundance.tsv (gene_id, gene_symbol, gene_biotype, counts)
# This should have the SAME rows and order as ens_output
biot_output <- data.frame(ensembl_gene = dd_biotype[,1], 
                         symbol = dd_biotype[,2], 
                         biotype = dd_biotype[,5],  # Column 5 is gene_biotype
                         counts = dd_biotype[,3])

# Write the biotype gene abundance file
write.table(biot_output, file=biot_output_file, quote=F, col.names=TRUE, row.names=F, sep="\t")



## print session info ##
print(" ")
print("Session Info below: ")
sessionInfo()
