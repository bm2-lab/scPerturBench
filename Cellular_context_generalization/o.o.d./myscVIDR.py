import sys
sys.path.insert(1, '/home/software/scVIDR/vidr/')
sys.path.insert(1, '/home/software/scVIDR/')
from vidr.vidr import VIDR
from utils import *
sys.path.append('/home/project/Pertb_benchmark')

import os
from myUtil import *
import scanpy as sc
import pandas as pd
import numpy as np
import torch


def perturbation_prediction(DataSet, outSample): 
    basePath = f'/home/project/Pertb_benchmark/DataSet/{DataSet}/outSample/hvg5000/scVIDR/'
    tmp = '{}/{}'.format(basePath, outSample)
    if not os.path.isdir(tmp): os.makedirs(tmp)
    os.chdir(tmp)
    path = f'/home/project/Pertb_benchmark/DataSet/{DataSet}/filter_hvg5000_logNor.h5ad'
    adata0 = sc.read_h5ad(path)
    adata0.X = adata0.layers['logNor']
    
    perturbations = list(adata0.obs['condition2'].unique())  ### perturbations
    filtered_per = list(filter(lambda x: x != 'control', perturbations))
    
    for perturbation in filtered_per:
        adata = adata0[(adata0.obs['condition2'] == perturbation) | (adata0.obs['condition2'] == 'control')]
        train_adata, test_adata = prepare_data(adata, 
                                       "condition1", 
                                       "condition2", 
                                       outSample, 
                                       perturbation, 
                                       normalized = True)                  
        model = VIDR(train_adata, linear_decoder = False)
        print(perturbation)
        print(outSample)
        model.train(max_epochs=100,
                batch_size=128,
                early_stopping=True,
                early_stopping_patience=25)
        model.save(f"./{perturbation}_{outSample}.pt")
    
        
        pred, delta = model.predict(ctrl_key = "control",
                                treat_key = perturbation,
                                cell_type_to_predict = outSample,
                                regression = False)
        pred.obs["condition2"] = 'imputed'
        ctrl_adata = adata[((adata.obs['condition1'] == outSample) & (adata.obs["condition2"] == "control"))]
        treat_adata = adata[((adata.obs['condition1'] == outSample) & (adata.obs["condition2"] == perturbation))]
        eval_adata1 = ctrl_adata.concatenate(treat_adata, pred)
        eval_adata1.obs=eval_adata1.obs[['condition2']]
        eval_adata1.obs = eval_adata1.obs.rename(columns={'condition2':'perturbation'})
        eval_adata1.obs['perturbation'] = eval_adata1.obs['perturbation'].replace([perturbation], 'stimulated')
        del eval_adata1.var
        eval_adata1.write_h5ad(f'./{perturbation}_imputed.h5ad')                        


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