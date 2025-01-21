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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print('p')


### kang data out sample test 
def inSample(X):
    outSample, hvg, DataLayer, redo = X   #hvg,datalayer,outsample celltype
    basePath = '/home/wzt/project/Pertb_benchmark/DataSet/{}/inSample/hvg{}/scVIDR/'.format(DataSet, hvg)
    outSample_test = outSample + '_test'
    print(outSample_test)
    
    tmp = '{}/{}'.format(basePath, outSample)
    if not os.path.isdir(tmp): os.makedirs(tmp)
    os.chdir(tmp)
    print(os.getcwd())
    path = '/home/wzt/project/Pertb_benchmark/DataSet/{}/filter_hvg{}_logNor.h5ad'.format(DataSet, hvg)
    adata0 = sc.read_h5ad(path)
    adata0.X = adata0.layers['logcounts']
    adata0.obs['condition_iid'] = adata0.obs['condition1'].astype(str) + '_' + adata0.obs['iid_test'].astype(str)
    
    perturbations = list(adata0.obs['condition2'].unique())  ### perturbations
    filtered_per = list(filter(lambda x: x != 'control', perturbations))
    
    for perturbation in filtered_per:
        adata = adata0[(adata0.obs['condition2'] == perturbation) | (adata0.obs['condition2'] == 'control')]
            
        train_adata, test_adata = prepare_data(adata, 
                                       "condition_iid", #cell_type_key
                                       "condition2", #treatment_key
                                       outSample_test, #cell_type_to_predict
                                       perturbation, #treatment_to_predict
                                       normalized = True)
                                       
                                       
        model = VIDR(train_adata, linear_decoder = False)
        #train_adata.obs["cell_dose"] = [f"{j}_{str(i)}" for (i,j) in zip(train_adata.obs["condition2"], train_adata.obs["dose_value"])]
        print(perturbation)
        model.train(max_epochs=100,
                batch_size=128,
                early_stopping=True,
                early_stopping_patience=25)
        model.save(f"./{perturbation}_{outSample_test}.pt")
    
        
        pred, delta = model.predict(ctrl_key = "control",
                                treat_key = perturbation,
                                cell_type_to_predict = outSample_test,
                                regression = False)
        pred.obs["condition2"] = 'imputed'
        ctrl_adata = adata[((adata.obs['condition_iid'] == outSample_test) & (adata.obs["condition2"] == "control"))]
        treat_adata = adata[((adata.obs['condition_iid'] == outSample_test) & (adata.obs["condition2"] == perturbation))]
        eval_adata1 = ctrl_adata.concatenate(treat_adata, pred)
        eval_adata1.obs=eval_adata1.obs[['condition2']]
        eval_adata1.obs = eval_adata1.obs.rename(columns={'condition2':'perturbation'})
        eval_adata1.obs['perturbation'] = eval_adata1.obs['perturbation'].replace([perturbation], 'stimulated')
        
        #eval_adata1.obs.columns=['perturbation']
        del eval_adata1.var
        eval_adata1.write_h5ad(f'./{perturbation}_imputed.h5ad')                        
        sc.tl.pca(eval_adata1)
        sc.pl.pca(eval_adata1, color=['perturbation'], frameon=False, save = f'{perturbation}.pdf', show = False)                        
    
    pid= os.getpid()
    print(pid)
    gpu_memory = pd.DataFrame(1, index=['gpu_memory'], columns=['gpu_memory'], dtype=str)
    devices = Device.all()
    for device in devices:
        processes = device.processes()
        if pid in processes.keys():
            p=processes[pid]
            gpu_memory['gpu_memory'] = p.gpu_memory_human()
    gpu_memory.to_csv('gpu_memory.csv', sep='\t', index=False)
       
        



def Main():
    mylist = []
    filein_tmp = '/home/wzt/project/Pertb_benchmark/DataSet/{}/filter_hvg5000_logNor.h5ad'.format(DataSet)
    adata_tmp = sc.read_h5ad(filein_tmp)
    outSamples = list(adata_tmp.obs['condition1'].unique())
    for hvg in [5000]:
        for outSample in outSamples:
            mylist.append([outSample, hvg, 'logNor', True])   ### outSample, hvg, 'logNor', True
    myPool(inSample, mylist, processes=1)


DataSet = 'Afriat'
print(DataSet)

if __name__ == '__main__':
    Main() 