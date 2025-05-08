library(SingleCellExperiment)
library(tidyverse)
library(rjson)

Args = commandArgs(T)
seed = Args[1]
dirName = Args[2]

setwd(dirName)
pa = {}
pa$pca_dim = 10
pa$ridge_penalty = .1
pa$gene_embedding = 'training_data'
pa$pert_embedding = 'training_data'


solve_y_axb <- function(Y, A = NULL, B = NULL, A_ridge = 0.01, B_ridge = 0.01){
  stopifnot(is.matrix(Y) || is(Y, "Matrix"))
  stopifnot(is.null(A) || is.matrix(A) || is(A, "Matrix"))
  stopifnot(is.null(B) || is.matrix(B) || is(B, "Matrix"))
  
  center <- rowMeans(Y)
  Y <- Y - center
  
  if(! is.null(A) && ! is.null(B)){
    stopifnot(nrow(Y) == nrow(A))
    stopifnot(ncol(Y) == ncol(B))
    # fit <- lm.fit(kronecker(t(B), A), as.vector(Y))
    tmp <- as.matrix(Matrix::solve(t(A) %*% A + Matrix::Diagonal(ncol(A)) * A_ridge) %*% t(A) %*% Y %*% t(B) %*% Matrix::solve(B %*% t(B) + Matrix::Diagonal(nrow(B)) * B_ridge))
  }else if(is.null(B)){
    fit <- lm.fit(A, Y)
    tmp <- as.matrix(Matrix::solve(t(A) %*% A + Matrix::Diagonal(ncol(A)) * A_ridge) %*% t(A) %*% Y)
  }else if(is.null(A)){
    fit <- lm.fit(t(B), t(Y))
    tmp <- as.matrix(Y %*% t(B) %*% Matrix::solve(B %*% t(B) + Matrix::Diagonal(nrow(B)) * B_ridge))
  }else{
    stop("Either A or B must be non-null")
  }
  tmp[is.na(tmp)] <- 0
  list(K = tmp, center = center)
}

filein = 'data/train/perturb_processed.h5ad'
sce <- zellkonverter::readH5AD(filein)

filein = paste0('data/train/splits/set2conditions_', seed, '.tsv')
set2condition = rjson::fromJSON(file = filein)
if(! "ctrl" %in% set2condition$train){
  set2condition$train <- c(set2condition$train, "ctrl")
}

sce <- sce[,sce$condition %in% unlist(set2condition)]

# Clean up the colData(sce) a bit
sce$condition <- droplevels(sce$condition)
sce$clean_condition <- stringr::str_remove(sce$condition, "\\+ctrl")
training_df <- tibble(training = names(set2condition), condition = set2condition) %>%
  unnest(condition)
colData(sce) <- colData(sce) %>%
  as_tibble() %>%
  tidylog::left_join(training_df, by = "condition") %>%
  DataFrame()

gene_names <- rowData(sce)[["gene_name"]]
rownames(sce) <- gene_names

baseline <- MatrixGenerics::rowMeans2(assay(sce, "X")[,sce$condition == "ctrl",drop=FALSE])

# Pseudobulk everything
psce <- glmGamPoi::pseudobulk(sce, group_by = vars(condition, clean_condition, training))
assay(psce, "change") <- assay(psce, "X") - baseline

train_data <- psce[,psce$training == "train"]

gene_emb <- if(pa$gene_embedding == "training_data"){
  pca <- irlba::prcomp_irlba(as.matrix(assay(train_data, "X")), n = pa$pca_dim)
  rownames(pca$x) <- rownames(train_data)
  pca$x
}

pert_emb <- if(pa$pert_embedding == "training_data"){
  pca <- irlba::prcomp_irlba(as.matrix(assay(train_data, "X")), n = pa$pca_dim)
  rownames(pca$x) <- rownames(train_data)
  t(pca$x)
}


if(! "ctrl" %in% colnames(pert_emb)){
  pert_emb <- cbind(pert_emb, ctrl = rep(0, nrow(pert_emb)))
}
pert_matches <- match(colnames(pert_emb), train_data$clean_condition)
gene_matches <- match(rownames(gene_emb), rownames(train_data))
if(sum(! is.na(pert_matches)) <= 1){
  stop("Too few matches between clean_conditions and pert_embedding")
}
if(sum(! is.na(gene_matches)) <= 1){
  stop("Too few matches between gene names and gene_embedding")
}

gene_emb_sub <- gene_emb[! is.na(gene_matches),,drop=FALSE]
pert_emb_training <- pert_emb[,! is.na(pert_matches),drop=FALSE]
Y <-  assay(train_data, "change")[na.omit(gene_matches), na.omit(pert_matches),drop=FALSE]
coefs <- solve_y_axb(Y = Y, A = gene_emb_sub, B = pert_emb_training,
                     A_ridge = pa$ridge_penalty, B_ridge = pa$ridge_penalty)


pert_matches_all <- match(psce$clean_condition, colnames(pert_emb))
pert_emb_all <- pert_emb[,pert_matches_all,drop=FALSE]
colnames(pert_emb_all) <- psce$clean_condition

baseline <- baseline[na.omit(gene_matches)]

pred <- as.matrix(gene_emb_sub %*% coefs$K %*% pert_emb_all + coefs$center + baseline)

rownames(pred) <- rownames(psce)[na.omit(gene_matches)]

fileout = paste0('savedModels', seed, '/pred.tsv')

write.table(pred, fileout, quote = FALSE, sep = '\t', col.names = NA, row.names = TRUE)