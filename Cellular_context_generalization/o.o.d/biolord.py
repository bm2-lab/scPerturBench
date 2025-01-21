import numpy as np
import pandas as pd
import scanpy as sc
import biolord
import os, sys
sys.path.append('/home/wzt/project/Pertb_benchmark')
from myUtil import *
from nvitop import Device
import yaml
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from scipy.stats import ttest_rel




def perturbation_prediction(X):
    outSample, hvg, DataLayer, redo = X  
    basePath = '/home/wzt/project/Pertb_benchmark/DataSet/{}/outSample/hvg{}/bioLord/'.format(DataSet, hvg)
    tmp = '{}/{}'.format(basePath, outSample)
    if not os.path.isdir(tmp): os.makedirs(tmp)
    os.chdir(tmp)
    path = '/home/wzt/project/Pertb_benchmark/DataSet/{}/filter_hvg{}_logNor.h5ad'.format(DataSet, hvg)
    adata = sc.read_h5ad(path)
    adata.X = adata.layers['logNor'].copy()
  
    np.random.seed(1116)
    meta = adata.obs
    meta['ct_con'] = meta['condition1'].astype(str) + '_' + meta['condition2'].astype(str)
    cn = 'split_{}'.format(outSample) 
    meta[cn] = None
    meta.loc[(meta['condition1'] == outSample) & (meta['condition2'] != 'control'), cn] = 'ood'
    ood_count = len(meta[meta[cn] == 'ood'])
    meta.loc[(meta[cn] != 'ood'), cn] =  np.random.choice(['train', 'valid'], size = (adata.shape[0] - ood_count), replace=True, p=[0.9, 0.1])
    adata.obs = meta
    

    biolord.Biolord.setup_anndata(adata = adata,
    ordered_attributes_keys = 'dose',  # order
    categorical_attributes_keys = ['condition1', 'condition2'], # class
    layer = "logNor")
    
    ### Run Biolord
    module_params = {"decoder_width": 1024, 
                     "decoder_depth": 4, "attribute_nn_width": 512, "attribute_nn_depth": 2,
        "n_latent_attribute_categorical": 4, "gene_likelihood": "normal", #raw counts,nb  default normal
        "reconstruction_penalty": 1e2, # 
        "unknown_attribute_penalty": 1e1,#
        "unknown_attribute_noise_param": 1e-1, #
        "attribute_dropout_rate": 0.1, "use_batch_norm": False, "use_layer_norm": False, "seed": 42,}
    
    model = biolord.Biolord(
        adata=adata,
        n_latent=256,
        model_name=outSample,
        module_params=module_params,
        train_classifiers=False,  
        split_key= 'split_{}'.format(outSample),
        train_split='train', valid_split='valid', test_split='ood')
    ### Train the model
    trainer_params = {"n_epochs_warmup": 0, "latent_lr": 1e-4, "latent_wd": 1e-4, "decoder_lr": 1e-4, "decoder_wd": 1e-4,
        "attribute_nn_lr": 1e-2, "attribute_nn_wd": 4e-8, "step_size_lr": 45, "cosine_scheduler": True, "scheduler_final_lr": 1e-5,}
    
    print('model start!')
    model.train(
        max_epochs=500,
        batch_size=128,
        plan_kwargs=trainer_params,
        early_stopping=True,
        early_stopping_patience=20,
        check_val_every_n_epoch=5,
        enable_checkpointing=False,)
    
    idx_source = np.where(((adata.obs["condition2"] == 'control') & (adata.obs["condition1"] == outSample)))[0]     
    adata_source = adata[idx_source].copy()
    adata_preds = model.compute_prediction_adata(adata, adata_source, target_attributes = ['condition2', 'dose'])

    
    cat_adata = adata[(adata.obs['condition1'] == outSample)]  
    ctrl_adata = cat_adata[(cat_adata.obs['condition2'] == 'control')]
    ctrl_adata.obs['group']='control'
    stim_adata = cat_adata[(cat_adata.obs['condition2'] != 'control')] 
    stim_adata.obs['group']='stimulated'
    adata_preds = adata_preds[(adata_preds.obs['condition2'] != 'control')]
    adata_preds.obs['group'] = 'imputed'
    result = ad.concat([ctrl_adata, stim_adata, adata_preds])  
    perturbations = list(adata.obs['condition2'].unique())  
    filtered_per = list(filter(lambda x: x != 'control', perturbations))
    for perturbation in filtered_per:
        print(perturbation)
        result2 = result[(result.obs['condition2'] == perturbation) | (result.obs['condition2'] == 'control')]
        result2.obs['condition2'] = result2.obs['group']
        del result2.obs['group']
        result2.obs = result2.obs.rename(columns={'condition2':'perturbation'})
        result2.write_h5ad('{}_imputed.h5ad'.format(perturbation))
   


def outSample():
    mylist = []
    filein_tmp = '/home/wzt/project/Pertb_benchmark/DataSet/{}/filter_hvg5000_logNor.h5ad'.format(DataSet)
    adata_tmp = sc.read_h5ad(filein_tmp)
    outSamples = list(adata_tmp.obs['condition1'].unique())
    for hvg in [5000]:
        for outSample in outSamples:
            mylist.append([outSample, hvg, 'logNor', True])   
    myPool(perturbation_prediction, mylist, processes=1)


DataSet = 'TCDDJointFactor'
print(DataSet)



if __name__ == '__main__':
    outSample()

