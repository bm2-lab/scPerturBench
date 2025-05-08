import os, sys
sys.path.append('/home//project/Pertb_benchmark')
from myUtil import *
import yaml
import shutil
import anndata as ad
import torch
import warnings
warnings.filterwarnings('ignore')


def Kang_OutSample(DataSet, outSample):
    def fun1(x):
        if x == '{}_Ctrl'.format(outSample):
              return "control"
        elif x == '{}_Real'.format(outSample):
              return "stimulated"
        elif x == '{}_SCREEN'.format(outSample):
              return "imputed"
        else:
             pass
    basePath = f'/home//project/Pertb_benchmark/DataSet/{DataSet}/outSample/hvg5000/SCREEN/'
    tmp = '{}/{}'.format(basePath, outSample)
    if not os.path.isdir(tmp): os.makedirs(tmp)
    os.chdir(tmp)
    
    path = f'/home//project/Pertb_benchmark/DataSet/{DataSet}/filter_hvg5000_logNor.h5ad'
    adata = sc.read_h5ad(path)
    perturbations = list(adata.obs['condition2'].unique())
    perturbations = [i for i in perturbations if i != 'control']
    for perturbation in perturbations:
        ncell = adata[(adata.obs['condition1'] == outSample) & (adata.obs['condition2'] == perturbation)]
        if ncell.shape[0] == 0: continue
        cmd = 'python /home//software/SCREEN/screen/screen.py  -in  {}  -ou  ./'\
        ' --label {}  --condition_key  condition2  --cell_type_key condition1 --ctrl_key control --stim_key {}'\
        ' --latent_dim 100 --batch_size 64 --epochs 40  --full_quadratic False --activation leaky_relu'\
        ' --optimizer Adam  --GPU_NUM  {}'.format(path, outSample, perturbation, GPU_NUM)
        subprocess.call(cmd, shell=True)
        filein = 'SCREEN_{}.h5ad'.format(outSample)
        adata1 = sc.read_h5ad(filein)
        os.remove(filein)  #### 删除防止后续重复利用
        adata1.obs['perturbation'] = adata1.obs['condition2'].apply(fun1)
        adata1.write_h5ad('{}_imputed.h5ad'.format(perturbation))

def KangMain():
    mylist = []
    filein_tmp = '/home//project/Pertb_benchmark/DataSet/{}/filter_hvg5000_logNor.h5ad'.format(DataSet)
    adata_tmp = sc.read_h5ad(filein_tmp)
    outSamples = list(adata_tmp.obs['condition1'].unique())
    for outSample in outSamples:
        Kang_OutSample(DataSet, outSample)


DataSet = 'kangCrossCell'
GPU_NUM = 0

## conda activate cpa

if __name__ == '__main__':
    KangMain()