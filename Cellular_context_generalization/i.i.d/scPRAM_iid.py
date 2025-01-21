import sys
from scpram import models
import scanpy as sc
import warnings
import os
sys.path.append('/home/wzt/project/Pertb_benchmark')
from myUtil import *
import yaml
import pandas as pd
import numpy as np
import torch
import shutil
from nvitop import Device
from anndata import AnnData
from matplotlib import pyplot as plt
import scvi
print('p')


### kang data out sample test 
def inSample(X):
    outSample, hvg, DataLayer, redo = X   #hvg,datalayer,outsample celltype
    basePath = '/home/wzt/project/Pertb_benchmark/DataSet/{}/inSample/hvg{}/scPRAM/'.format(DataSet, hvg)
    outSample_test = outSample + '_test' ### 1111
    
    tmp = '{}/{}'.format(basePath, outSample)
    if not os.path.isdir(tmp): os.makedirs(tmp)
    os.chdir(tmp)
    #kang_preData(hvg, DataLayer)
    fileout = '/home/wzt/project/Pertb_benchmark/DataSet/{}/filter_hvg{}_logNor.h5ad'.format(DataSet, hvg)
    adata0 = sc.read_h5ad(fileout)
    adata0.obs['condition_iid'] = adata0.obs['condition1'].astype(str) + '_' + adata0.obs['iid_test'].astype(str) ### 2222
    adata0.X = adata0.layers['logcounts'].astype(np.float32)
    print(hvg)
    
    perturbations = list(adata0.obs['condition2'].unique())  ### perturbations
    filtered_per = list(filter(lambda x: x != 'control', perturbations))
    #if outSample in ['PW029']: filtered_per = 'Etoposide'
    #elif outSample in ['PW030', 'PW032', 'PW034', 'PW036']: filtered_per = ['Etoposide', 'Panobinostat']
    #elif outSample in ['PW040']: filtered_per = ['Panobinostat']
    for perturbation in filtered_per:
        adata = adata0[(adata0.obs['condition2'] == perturbation) | (adata0.obs['condition2'] == 'control')]
        model = models.SCPRAM(input_dim=adata.n_vars, device='cuda:0')
        model = model.to(model.device)
        key_dic = {'condition_key': 'condition2',
           'cell_type_key': 'condition_iid',
           'ctrl_key': 'control',
           'stim_key': perturbation,  ###
           'pred_key': 'predict',
           }
        cell_to_pred = outSample_test
        train = adata[~((adata.obs[key_dic['cell_type_key']] == cell_to_pred) &
                   (adata.obs[key_dic['condition_key']] == key_dic['stim_key']))]
        model.train_SCPRAM(train, epochs=100)
    
        adata_to_pred = adata[((adata.obs[key_dic['cell_type_key']] == cell_to_pred) &
                       (adata.obs[key_dic['condition_key']] == key_dic['ctrl_key']))]
        pred = model.predict(train_adata=train,
                         cell_to_pred=cell_to_pred,
                         key_dic=key_dic,
                         ratio=0.005)  # The ratio need to vary with the size of dataset
        print(pred)
    
    
        pred.obs["condition2"] = 'imputed'
        ctrl_adata = adata[((adata.obs['condition_iid'] == outSample_test) & (adata.obs["condition2"] == "control"))]
        treat_adata = adata[((adata.obs['condition_iid'] == outSample_test) & (adata.obs["condition2"] == perturbation))] ###
        eval_adata1 = ctrl_adata.concatenate(treat_adata, pred)
        
        eval_adata1.obs=eval_adata1.obs[['condition2']]
        eval_adata1.obs = eval_adata1.obs.rename(columns={'condition2':'perturbation'})
        eval_adata1.obs['perturbation'] = eval_adata1.obs['perturbation'].replace([perturbation], 'stimulated')
        #eval_adata1.obs = eval_adata1.obs[['condition2']]
        del eval_adata1.var
        eval_adata1.write_h5ad('{}_imputed.h5ad'.format(perturbation)) ###
        sc.tl.pca(eval_adata1)
        sc.pl.pca(eval_adata1, color=["perturbation"], frameon=False, save = "iid_{}.pdf".format(perturbation), show=False)
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
    #outSamples = ['PW029']
    for hvg in [5000]:
        for outSample in outSamples:
            mylist.append([outSample, hvg, 'logNor', True])
    myPool(inSample, mylist, processes=1)

DataSet = 'Afriat'
#multiprocessing.set_start_method('spawn', force=True)
#torch.cuda.set_device(1)

    
    
if __name__ == '__main__':
    Main()  ###  multi condition    out of sample test
    
