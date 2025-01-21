import copy
import json
import os
from pathlib import Path
import sys
import warnings
import numpy as np
import torch
import scgpt as scg
from scgpt.tasks import GeneEmbedding, embed_data
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.preprocess import Preprocessor
from scgpt.utils import set_seed 
import joblib
import scanpy as sc

set_seed(42)
model_dir = Path("./data/scGPT_human")

data = sc.read_h5ad(f"sciplex3_dataset.h5ad")
data.X = data.layers['counts']
cellemb_data = embed_data(data, model_dir, max_length=512, cell_embedding_mode="cls")
celltype_dict = {}
for i in np.unique(cellemb_data.obs['condition1']):
    tmp_data = cellemb_data[np.array(cellemb_data.obs['condition1']==i) & np.array(cellemb_data.obs['condition2']=='control')]
    celltype_dict[i] = tmp_data.obsm['X_scGPT'].mean(axis=0)
joblib.dump(celltype_dict,f"sciplex3_cell_embs.pkl")