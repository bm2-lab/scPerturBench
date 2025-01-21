import sys
sys.path.insert(1, '/NFS_home/NFS_home_1/wangyiheng/perturbation_benchmark/algorithms/DL/scVIDR/vidr/')
from utils import *
from vidr.vidr import VIDR
import os
sys.path.append('/home/wzt/project/Pertb_benchmark')
from myUtil import *
from nvitop import Device
import yaml
import scanpy as sc
#import scgen as scg
import pandas as pd
import numpy as np
import torch
import seaborn as sns
import gseapy as gp
from scipy import stats
from scipy import linalg
from scipy import spatial
from anndata import AnnData
from scipy import sparse
from statannotations.Annotator import Annotator
from matplotlib import pyplot as plt
import scvi



def perturbation_prediction(X):
    outSample, hvg, DataLayer, redo = X   
    basePath = '/home/wzt/project/Pertb_benchmark/DataSet/{}/outSample/hvg{}/scVIDR/'.format(DataSet, hvg)
    
    tmp = '{}/{}'.format(basePath, outSample)
    if not os.path.isdir(tmp): os.makedirs(tmp)
    os.chdir(tmp)
    print(os.getcwd())
    path = '/home/wzt/project/Pertb_benchmark/DataSet/{}/filter_hvg{}_logNor.h5ad'.format(DataSet, hvg)
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
        print(hvg)
        model.train(max_epochs=100,
                batch_size=128,
                early_stopping=True,
                early_stopping_patience=25)
        model.save(f"./{perturbation}_{hvg}g_{outSample}.pt")
    
        
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
    mylist = []
    filein_tmp = '/home/wzt/project/Pertb_benchmark/DataSet/{}/filter_hvg5000_logNor.h5ad'.format(DataSet)
    adata_tmp = sc.read_h5ad(filein_tmp)
    outSamples = list(adata_tmp.obs['condition1'].unique())
    for hvg in [5000]:
        for outSample in outSamples:
            mylist.append([outSample, hvg, 'logNor', True])   
    myPool(perturbation_prediction, mylist, processes=1)


DataSet = 'crossSpeciesJointFactor'
print(DataSet)


if __name__ == '__main__':
    outSample() 