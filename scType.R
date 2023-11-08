
# load libraries and functions
lapply(c("dplyr","Seurat","HGNChelper","openxlsx"), library, character.only = T)
source("https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/R/gene_sets_prepare.R"); source("https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/R/sctype_score_.R")

# get cell-type-specific gene sets from our in-built database (DB)
gs_list = gene_sets_prepare("https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/ScTypeDB_short.xlsx", "Immune system") # e.g. Immune system, Liver, Pancreas, Kidney, Eye, Brain

# assign cell types
scRNAseqData = readRDS(gzcon(url('https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/exampleData.RDS'))); #load example scRNA-seq matrix
es.max = sctype_score(scRNAseqData = scRNAseqData, scaled = TRUE, gs = gs_list$gs_positive, gs2 = gs_list$gs_negative)

# View results, cell-type by cell matrix. See the complete example below
View(es.max)



# load libraries
lapply(c("dplyr","Seurat","HGNChelper"), library, character.only = T)

# Load the PBMC dataset
pbmc.data <- Read10X(data.dir = "/Users/hyunjaecho/Downloads/pbmc1k/")
# Initialize the Seurat object with the raw (non-normalized data).
#gene_expression_matrix <- pbmc.data$`Gene Expression`
gene_expression_matrix <- pbmc.data

pbmc <- CreateSeuratObject(counts = gene_expression_matrix, project = "pbmc1k", min.cells = 3, min.features = 200)



# normalize data
pbmc[["percent.mt"]] <- PercentageFeatureSet(pbmc, pattern = "^MT-")
# pbmc <- subset(pbmc, subset = nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 5) # make some filtering based on QC metrics visualizations, see Seurat tutorial: https://satijalab.org/seurat/articles/pbmc3k_tutorial.html
pbmc <- NormalizeData(pbmc, normalization.method = "LogNormalize", scale.factor = 10000)
pbmc <- FindVariableFeatures(pbmc, selection.method = "vst", nfeatures = 2000)

# scale and run PCA
pbmc <- ScaleData(pbmc, features = rownames(pbmc))
pbmc <- RunPCA(pbmc, features = VariableFeatures(object = pbmc))

# Check number of PC components (we selected 10 PCs for downstream analysis, based on Elbow plot)
ElbowPlot(pbmc)

# cluster and visualize
pbmc <- FindNeighbors(pbmc, dims = 1:10)
pbmc <- FindClusters(pbmc, resolution = 0.8)
pbmc <- RunUMAP(pbmc, dims = 1:10)
DimPlot(pbmc, reduction = "umap")




# load gene set preparation function
source("https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/R/gene_sets_prepare.R")
# load cell type annotation function
source("https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/R/sctype_score_.R")


# DB file
db_ = "https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/ScTypeDB_full.xlsx";
tissue = "Immune system" # e.g. Immune system,Pancreas,Liver,Eye,Kidney,Brain,Lung,Adrenal,Heart,Intestine,Muscle,Placenta,Spleen,Stomach,Thymus 

# prepare gene sets
gs_list = gene_sets_prepare(db_, tissue)



# get cell-type by cell matrix
es.max = sctype_score(scRNAseqData = pbmc[["RNA"]]@scale.data, scaled = TRUE, 
                      gs = gs_list$gs_positive, gs2 = gs_list$gs_negative) 

# NOTE: scRNAseqData parameter should correspond to your input scRNA-seq matrix. 
# In case Seurat is used, it is either pbmc[["RNA"]]@scale.data (default), pbmc[["SCT"]]@scale.data, in case sctransform is used for normalization,
# or pbmc[["integrated"]]@scale.data, in case a joint analysis of multiple single-cell datasets is performed.

# merge by cluster
cL_resutls = do.call("rbind", lapply(unique(pbmc@meta.data$seurat_clusters), function(cl){
  es.max.cl = sort(rowSums(es.max[ ,rownames(pbmc@meta.data[pbmc@meta.data$seurat_clusters==cl, ])]), decreasing = !0)
  head(data.frame(cluster = cl, type = names(es.max.cl), scores = es.max.cl, ncells = sum(pbmc@meta.data$seurat_clusters==cl)), 10)
}))
sctype_scores = cL_resutls %>% group_by(cluster) %>% top_n(n = 1, wt = scores)  

# set low-confident (low ScType score) clusters to "unknown"
sctype_scores$type[as.numeric(as.character(sctype_scores$scores)) < sctype_scores$ncells/4] = "Unknown"
print(sctype_scores[,1:3])



pbmc@meta.data$customclassif = ""
for(j in unique(sctype_scores$cluster)){
  cl_type = sctype_scores[sctype_scores$cluster==j,]; 
  pbmc@meta.data$customclassif[pbmc@meta.data$seurat_clusters == j] = as.character(cl_type$type[1])
}

DimPlot(pbmc, reduction = "umap", label = TRUE, repel = TRUE, group.by = 'customclassif')        



barcodes <- read.table("/Users/hyunjaecho/Downloads/pbmc1k/barcodes.tsv", header = TRUE, sep = "\t")

seurat_barcodes <- rownames(pbmc)  # assuming `pbmc` is your Seurat object


all(barcodes$barcodes %in% seurat_barcodes)  # Replace `Barcode` with the column name in `barcodes.tsv` containing the barcodes


cell_types <- pbmc@meta.data$customclassif


barcodes_df <- data.frame(Barcode = rownames(pbmc))
cell_types_df <- data.frame(Barcode = rownames(pbmc@meta.data), CellType = pbmc@meta.data$customclassif)

# Assuming `pbmc` is your Seurat object and `customclassif` is the column with cell type annotations
mapping <- data.frame(
  Barcode = rownames(pbmc@meta.data),
  CellType = pbmc@meta.data$customclassif
)

# Checking the first few rows of the mapping to ensure it's correct
head(mapping)

# Save as a CSV file
write.table(mapping, file = "/Users/hyunjaecho/Downloads/barcode_to_celltype.csv", sep = ",", row.names = FALSE, quote = FALSE)

# Or save as a TSV file
write.table(mapping, file = "/Users/hyunjaecho/Downloads/barcode_to_celltype.tsv", sep = "\t", row.names = FALSE, quote = FALSE)
