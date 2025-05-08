import sys
from scpram import models
import scanpy as sc
import warnings
import os
sys.path.append('/home/project/Pertb_benchmark')
from myUtil import *
import yaml
import pandas as pd
import numpy as np
import torch
import shutil
from anndata import AnnData
import scvi


def perturbation_prediction(DataSet, outSample): 
    basePath = f'/home/project/Pertb_benchmark/DataSet/{DataSet}/outSample/hvg5000/scPRAM/'
    tmp = '{}/{}'.format(basePath, outSample)
    if not os.path.isdir(tmp): os.makedirs(tmp)
    os.chdir(tmp)
    fileout = f'/home/project/Pertb_benchmark/DataSet/{DataSet}/filter_hvg5000_logNor.h5ad'
    adata0 = sc.read_h5ad(fileout)
    
    adata0.X = adata0.layers['logNor']
    
    perturbations = list(adata0.obs['condition2'].unique())  
    filtered_per = list(filter(lambda x: x != 'control', perturbations))
    
    for perturbation in filtered_per:
        adata = adata0[(adata0.obs['condition2'] == perturbation) | (adata0.obs['condition2'] == 'control')]
        model = models.SCPRAM(input_dim=adata.n_vars, device='cuda:0')
        model = model.to(model.device)
        key_dic = {'condition_key': 'condition2',
           'cell_type_key': 'condition1',
           'ctrl_key': 'control',
           'stim_key': perturbation,  
           'pred_key': 'predict',
           }
        cell_to_pred = outSample
        train = adata[~((adata.obs[key_dic['cell_type_key']] == cell_to_pred) &
                   (adata.obs[key_dic['condition_key']] == key_dic['stim_key']))]
        model.train_SCPRAM(train, epochs=100)
    
        adata_to_pred = adata[((adata.obs[key_dic['cell_type_key']] == cell_to_pred) &
                       (adata.obs[key_dic['condition_key']] == key_dic['ctrl_key']))]
        pred = model.predict(train_adata=train,
                         cell_to_pred=cell_to_pred,
                         key_dic=key_dic,
                         ratio=0.005)  
        print(pred)
        pred.obs["condition2"] = 'imputed'
        ctrl_adata = adata[((adata.obs['condition1'] == outSample) & (adata.obs["condition2"] == "control"))]
        treat_adata = adata[((adata.obs['condition1'] == outSample) & (adata.obs["condition2"] == perturbation))] 
        eval_adata1 = ctrl_adata.concatenate(treat_adata, pred)
        eval_adata1.obs=eval_adata1.obs[['condition2']]
        eval_adata1.obs = eval_adata1.obs.rename(columns={'condition2':'perturbation'})
        eval_adata1.obs['perturbation'] = eval_adata1.obs['perturbation'].replace([perturbation], 'stimulated')
        del eval_adata1.var
        eval_adata1.write_h5ad('{}_imputed.h5ad'.format(perturbation)) 
   


def outSample():
    filein_tmp = '/home/project/Pertb_benchmark/DataSet/{}/filter_hvg5000_logNor.h5ad'.format(DataSet)
    adata_tmp = sc.read_h5ad(filein_tmp)
    outSamples = list(adata_tmp.obs['condition1'].unique())
    for outSample in outSamples:
        perturbation_prediction(DataSet, outSample)


DataSet = 'kangCrossCell'


### conda activate cpa

if __name__ == '__main__':
    outSample()  
    
