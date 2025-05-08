import os, sys
sys.path.append('/home/project/Pertb_benchmark')
from myUtil import *
import yaml
import shutil
import anndata as ad
import torch
import warnings
from nvitop import Device
warnings.filterwarnings('ignore')

'''
conda activate cpa 
/home/software/SCREEN/screen/screen_iid.py的175行
'''

### 对kang数据进行out sample test
def Kang_inSample(DataSet):
    basePath = f'/home/project/Pertb_benchmark/DataSet/{DataSet}/inSample/hvg5000/SCREEN/'
    if not os.path.isdir(basePath): os.makedirs(basePath)
    os.chdir(basePath)
    path = f'/home/project/Pertb_benchmark/DataSet/{DataSet}/filter_hvg5000_logNor.h5ad'
    adata = sc.read_h5ad(path)
    perturbations = list(adata.obs['condition2'].unique())
    perturbations = [i for i in perturbations if i != 'control']
    for perturbation in perturbations:
        try:
            cmd = 'python /home/software/SCREEN/screen/screen_iid.py  -in  {} '\
            ' --condition_key  condition2  --cell_type_key condition1 --ctrl_key control --stim_key {}'\
            ' --latent_dim 100 --batch_size 64 --epochs 40  --full_quadratic False --activation leaky_relu'\
            ' --optimizer Adam  --GPU_NUM  {}  --epoch_vae 200'.format(path, perturbation, GPU_NUM)  ### 默认40 和 200 个epoch
            subprocess.call(cmd, shell=True)
        except Exception as e:
            print ('myError********************************************************\n')
            print (e)
            print (X, perturbation)

DataSet = 'kangCrossCell'
GPU_NUM = 0

if __name__ == '__main__':
    Kang_inSample(DataSet)