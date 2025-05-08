import os, sys
sys.path.append('/home/project/Pertb_benchmark')
from myUtil import *
import yaml
import scanpy as sc
import scgen
import anndata as ad
import torch

def perturbation_prediction(DataSet, outSample):  
    basePath = f'/home/project/Pertb_benchmark/DataSet/{DataSet}/outSample/hvg5000/scGen/'
    tmp = '{}/{}'.format(basePath, outSample)
    if not os.path.isdir(tmp): os.makedirs(tmp)
    os.chdir(tmp)
    path = f'/home/project/Pertb_benchmark/DataSet/{DataSet}/filter_hvg5000_logNor.h5ad'
    adata0 = sc.read_h5ad(path)
    
    perturbations = list(adata0.obs['condition2'].unique())  
    filtered_per = list(filter(lambda x: x != 'control', perturbations))
    
    for perturbation in filtered_per:
        adata = adata0[(adata0.obs['condition2'] == perturbation) | (adata0.obs['condition2'] == 'control')]
        adata_train = adata[~((adata.obs["condition1"] == outSample) & (adata.obs["condition2"] == perturbation))].copy()
        print(perturbation)
        print(outSample)
        scgen.SCGEN.setup_anndata(adata_train, batch_key="condition2", labels_key="condition1")
        model = scgen.SCGEN(adata_train)
        print('model start')
        model.train(max_epochs=200, batch_size=64, early_stopping=True, early_stopping_patience=25)
        print('model done')
        pred, delta = model.predict(ctrl_key='control', stim_key=perturbation, celltype_to_predict=outSample)
        pred.obs['condition2'] = 'imputed'
        ctrl_adata = adata[((adata.obs['condition1'] == outSample) & (adata.obs['condition2'] == 'control'))]
        stim_adata = adata[((adata.obs['condition1'] == outSample) & (adata.obs['condition2'] == perturbation))]
        eval_adata = ctrl_adata.concatenate(stim_adata, pred) 
        eval_adata.obs=eval_adata.obs[['condition2']]
        eval_adata.obs = eval_adata.obs.rename(columns={'condition2':'perturbation'})
        eval_adata.obs['perturbation'] = eval_adata.obs['perturbation'].replace([perturbation], 'stimulated')
        del eval_adata.var
        eval_adata.write_h5ad('{}_imputed.h5ad'.format(perturbation))

def outSample():
    filein_tmp = '/home/project/Pertb_benchmark/DataSet/{}/filter_hvg5000_logNor.h5ad'.format(DataSet)
    adata_tmp = sc.read_h5ad(filein_tmp)
    outSamples = list(adata_tmp.obs['condition1'].unique())
    for outSample in outSamples:
        perturbation_prediction(DataSet, outSample)


DataSet = 'kangCrossCell'
print(DataSet)

### conda activate scarches

if __name__ == '__main__':
    outSample() 