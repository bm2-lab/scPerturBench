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

def inSample(X):
    hvg, DataLayer = X
    basePath = '/home/wzt/project/Pertb_benchmark/DataSet/{}/inSample/hvg{}/SCREEN/'.format(DataSet, hvg)
    if not os.path.isdir(basePath): os.makedirs(basePath)
    os.chdir(basePath)
    
    path = '/home/wzt/project/Pertb_benchmark/DataSet/{}/filter_hvg{}_{}.h5ad'.format(DataSet, hvg, DataLayer)
    adata = sc.read_h5ad(path)

    perturbations = list(adata.obs['condition2'].unique())
    perturbations = [i for i in perturbations if i != 'control']
    for perturbation in perturbations:
        try:
            cmd = 'python /home/wzt/software/SCREEN/screen/screen_iid.py  -in  {} '\
            ' --condition_key  condition2  --cell_type_key condition1 --ctrl_key control --stim_key {}'\
            ' --latent_dim 100 --batch_size 64 --epochs 40  --full_quadratic False --activation leaky_relu'\
            ' --optimizer Adam  --GPU_NUM  {}  --epoch_vae 200'.format(path, perturbation, GPU_NUM)
            subprocess.call(cmd, shell=True)
        except Exception as e:
            print ('myError********************************************************\n')
            print (e)
            print (X, perturbation)

    pid= os.getpid()
    gpu_memory = pd.DataFrame(1, index=['gpu_memory'], columns=['gpu_memory'], dtype=str)
    devices = Device.all()
    for device in devices:
        processes = device.processes()
        if pid in processes.keys():
            p=processes[pid]
            gpu_memory['gpu_memory'] = p.gpu_memory_human()
    gpu_memory.to_csv('gpu_memory.csv', sep='\t', index=False)




DataSet = 'kang_pbmc'
GPU_NUM = 0
