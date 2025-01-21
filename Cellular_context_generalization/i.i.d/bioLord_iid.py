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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ['MKL_THREADING_LAYER'] = 'GNU'  #!!
print('p')

### kang data out sample test 
def inSample(X):
    outSample, hvg, DataLayer, redo = X   #hvg,datalayer,outsample celltype
    basePath = '/home/wzt/project/Pertb_benchmark/DataSet/{}/inSample/hvg{}/bioLord/'.format(DataSet, hvg) #### 0000
    outSample_test = outSample + '_test' ### 11111
    print(outSample_test)
    
    
    tmp = '{}/{}'.format(basePath, outSample)
    if not os.path.isdir(tmp): os.makedirs(tmp)
    os.chdir(tmp)
    path = '/home/wzt/project/Pertb_benchmark/DataSet/{}/filter_hvg{}_logNor.h5ad'.format(DataSet, hvg)
    adata = sc.read_h5ad(path)
    adata.X = adata.layers['logNor'].copy()
    adata.obs['condition_iid'] = adata.obs['condition1'].astype(str) + '_' + adata.obs['iid_test'].astype(str)

    meta = adata.obs
    meta['ct_con'] = meta['condition_iid'].astype(str) + '_' + meta['condition2'].astype(str)
    cn = 'split_{}'.format(outSample_test) #col_name
    meta[cn] = None
    meta.loc[(meta['condition_iid'] == outSample_test) & (meta['condition2'] != 'control'), cn] = 'ood'
    ood_count = len(meta[meta[cn] == 'ood'])
    meta.loc[(meta[cn] != 'ood'), cn] =  np.random.choice(['train', 'valid'], size = (adata.shape[0] - ood_count), replace=True, p=[0.9, 0.1])
    adata.obs = meta
    
    if DataSet in ['kang_pbmc', 'PRJNA419230', 'salmonella', 'kangCrossPatient', 'covid19']:
        ok = 'None'
        ck = ["condition_iid", "condition2"]
    if DataSet in ['haber', 'crossPatientKaggle', 'crossPatient', 'crossPatientKaggleCell']:
        ok = 'None'
        ck = ["condition_iid", "condition2", 'batch']
    if DataSet in ['crossSpecies', 'Afriat', 'McFarland']:
        ok = 'None'
        ck = ["condition_iid", "condition2", 'condition3']
    if DataSet in ['TCDD', 'sciplex3']:
        ok = 'None'
        ck = ["condition_iid", "condition2", 'batch','dose']
    
    print(DataSet)
    print(ok)
    print(ck)
    
    biolord.Biolord.setup_anndata(adata = adata,
    ordered_attributes_keys = ok,  # order, eg.age,drug time,None, ['', '']
    categorical_attributes_keys = ck, # class, eg.celltype,tissue
    layer = "logNor")
    
    ### Run Biolord
    module_params = {"decoder_width": 1024, # scicplex3:4096, default 1024, 
                     "decoder_depth": 4, "attribute_nn_width": 512, "attribute_nn_depth": 2,
        "n_latent_attribute_categorical": 4, "gene_likelihood": "normal", #raw counts,nb  default normal
        "reconstruction_penalty": 1e2, # 
        "unknown_attribute_penalty": 1e1,#
        "unknown_attribute_noise_param": 1e-1, #
        "attribute_dropout_rate": 0.1, "use_batch_norm": False, "use_layer_norm": False, "seed": 42,}
    
    model = biolord.Biolord(
        adata=adata,
        n_latent=256, #sciplex3:256,tcdd 64, mcf/haber/kang/salm:32
        model_name=outSample,
        module_params=module_params,
        train_classifiers=False,  #classify elseif
        split_key= 'split_{}'.format(outSample_test),
        train_split='train', valid_split='valid', test_split='ood')
    ### Train the model
    trainer_params = {"n_epochs_warmup": 0, "latent_lr": 1e-4, "latent_wd": 1e-4, "decoder_lr": 1e-4, "decoder_wd": 1e-4,
        "attribute_nn_lr": 1e-2, "attribute_nn_wd": 4e-8, "step_size_lr": 45, "cosine_scheduler": True, "scheduler_final_lr": 1e-5,}
    
    print('model start!')
    #os.environ['MKL_THREADING_LAYER'] = 'GNU'
    model.train(
        max_epochs=500,
        batch_size=128,
        plan_kwargs=trainer_params,
        early_stopping=True,
        early_stopping_patience=20,
        check_val_every_n_epoch=5,
        #num_workers=2,
        enable_checkpointing=False,)
    
    ### adata
    #targets = list(filter(lambda x: x != 'condition_iid', ck))
    #print(targets)
    
    outSample_train = outSample + '_train'
    idx_source = np.where(((adata.obs["condition2"] == 'control') & (adata.obs["condition_iid"] == outSample_train)))[0]     # != ==
    

    ### index downsample
    if DataSet in ['TCDD', 'sciplex3'] and len(idx_source) > 1000:#
        downsampled_indices = np.random.choice(idx_source, size=1000, replace=False)#
    else:#1
        downsampled_indices = idx_source#
    
    adata_source = adata[downsampled_indices].copy()#
    

    
    adata_preds = model.compute_prediction_adata(adata, adata_source, target_attributes = ["condition2", 'condition3'])
    #, add_attributes = ["condition3"])
    
    cat_adata = adata[(adata.obs['condition_iid'] == outSample_test)]  ## Extract the celltype==outsample
    ctrl_adata = cat_adata[(cat_adata.obs['condition2'] == 'control')]
    ctrl_adata.obs['group']='control'
    stim_adata = cat_adata[(cat_adata.obs['condition2'] != 'control')] 
    stim_adata.obs['group']='stimulated'
    adata_preds = adata_preds[(adata_preds.obs['condition2'] != 'control')]
    adata_preds.obs['group'] = 'imputed'
    #adata_preds.obs['perturbation'] = 'imputed'
    result = ad.concat([ctrl_adata, stim_adata, adata_preds])  
    #result.write_h5ad('stimulated_imputed.h5ad')
    perturbations = list(adata.obs['condition2'].unique())  ### perturbations
    filtered_per = list(filter(lambda x: x != 'control', perturbations))
    for perturbation in filtered_per:
        print(perturbation)
        result2 = result[(result.obs['condition2'] == perturbation) | (result.obs['condition2'] == 'control')]
        result2.obs['condition2'] = result2.obs['group']
        del result2.obs['group']
        result2.obs = result2.obs.rename(columns={'condition2':'perturbation'})
        if 'condition3' in result2.obs.columns: result2.obs = result2.obs.rename(columns = {'condition3':'Time'})
        result2.write_h5ad('{}_imputed.h5ad'.format(perturbation))
        sc.tl.pca(result2)
        sc.pl.pca(result2, color=['perturbation'], frameon=False, save = "{}_iid2.pdf".format(perturbation), show=False)
   
   
   
    
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
    #outSamples = ['HepatocytesPortal']
    for hvg in [5000]:
        for outSample in outSamples:
            mylist.append([outSample, hvg, 'logNor', True])   ### outSample, hvg, 'logNor', True
    myPool(inSample, mylist, processes=1)


DataSet = 'McFarland'
print(DataSet)


#multiprocessing.set_start_method('spawn', force=True)
#torch.cuda.set_device(1)
if __name__ == '__main__':
    Main()  ###  multi condition    out of sample test

