import os, sys
sys.path.append('/home/project/Pertb_benchmark')
from myUtil import *
import anndata as ad
import torch
import warnings
warnings.filterwarnings('ignore')


def Kang_OutSample(DataSet, outSample):
    basePath = '/home//project/Pertb_benchmark/DataSet/{}/outSample/hvg5000/controlMean/'.format(DataSet)
    tmp = '{}/{}'.format(basePath, outSample)
    if not os.path.isdir(tmp): os.makedirs(tmp)
    os.chdir(tmp)

    path = '/home//project/Pertb_benchmark/DataSet/{}/filter_hvg5000_logNor.h5ad'.format(DataSet)
    adata = sc.read_h5ad(path)
    perturbations = list(adata.obs['condition2'].unique())
    perturbations = [i for i in perturbations if i != 'control']
    for perturbation in perturbations:
        control_adata = adata[ (adata.obs['condition2'].isin(['control'])) & (adata.obs['condition1'].isin([outSample]))]
        perturb_adata = adata[ (adata.obs['condition2'].isin([perturbation])) & (adata.obs['condition1'].isin([outSample]))]
        imputed_adata = control_adata.copy() 

        control_adata.obs['perturbation'] = 'control'

        imputed_adata.obs['perturbation'] = 'imputed'

        perturb_adata.obs['perturbation'] = 'stimulated'
        result = ad.concat([control_adata, imputed_adata, perturb_adata])
        result.write_h5ad('{}_imputed.h5ad'.format(perturbation))

def KangMain(DataSet):
    filein_tmp = '/home//project/Pertb_benchmark/DataSet/{}/filter_hvg5000_logNor.h5ad'.format(DataSet)
    adata_tmp = sc.read_h5ad(filein_tmp)
    outSamples = list(adata_tmp.obs['condition1'].unique())
    for outSample in tqdm(outSamples):
        Kang_OutSample(DataSet, outSample)


DataSets = ["kangCrossCell", "kangCrossPatient", "Parekh",  "Haber", "crossPatient", "KaggleCrossPatient",
           "KaggleCrossCell", "crossSpecies", "McFarland", "Afriat", "TCDD", "sciplex3"]

DataSets = ['kangCrossCell']


## conda activate  cpa

if __name__ == '__main__':
    for DataSet in tqdm(DataSets):
        print (DataSet)
        KangMain(DataSet)