import os, sys
sys.path.append('/home/wzt/project/Pertb_benchmark')
from myUtil import *
import yaml
import shutil
import anndata as ad
import torch
import warnings
from nvitop import Device
warnings.filterwarnings('ignore')


def OutSample(X):
    def fun1(x):
        if x == '{}_Ctrl'.format(outSample):
              return "control"
        elif x == '{}_Real'.format(outSample):
              return "stimulated"
        elif x == '{}_SCREEN'.format(outSample):
              return "imputed"
        else:
             pass
    outSample, hvg, DataLayer, redo = X
    basePath = '/home/wzt/project/Pertb_benchmark/DataSet/{}/outSample/hvg{}/SCREEN/'.format(DataSet, hvg)
    tmp = '{}/{}'.format(basePath, outSample)
    if not os.path.isdir(tmp): os.makedirs(tmp)
    os.chdir(tmp)
    
    path = '/home/wzt/project/Pertb_benchmark/DataSet/{}/filter_hvg{}_{}.h5ad'.format(DataSet, hvg, DataLayer)
    adata = sc.read_h5ad(path)
    perturbations = list(adata.obs['condition2'].unique())  
    perturbations = [i for i in perturbations if i != 'control']
    for perturbation in perturbations:
        ncell = adata[(adata.obs['condition1'] == outSample) & (adata.obs['condition2'] == perturbation)]
        if ncell.shape[0] == 0: continue    #### 
        try:
            if not redo and os.path.isfile('{}_imputed.h5ad'.format(perturbation)): continue

            cmd = 'python /home/wzt/software/SCREEN/screen/screen.py  -in  {}  -ou  ./'\
            ' --label {}  --condition_key  condition2  --cell_type_key condition1 --ctrl_key control --stim_key {}'\
            ' --latent_dim 100 --batch_size 64 --epochs 40  --full_quadratic False --activation leaky_relu'\
            ' --optimizer Adam  --GPU_NUM  {}'.format(path, outSample, perturbation, GPU_NUM)
            subprocess.call(cmd, shell=True)
            filein = 'SCREEN_{}.h5ad'.format(outSample)
            adata1 = sc.read_h5ad(filein)
            os.remove(filein)  #### 
            adata1.obs['perturbation'] = adata1.obs['condition2'].apply(fun1)
            adata1.write_h5ad('{}_imputed.h5ad'.format(perturbation))

        except Exception as e:
            print ('myError********************************************************\n')
            print (e)
            print (X, perturbation)


DataSet = 'TCDDJointFactor'
GPU_NUM = 0

