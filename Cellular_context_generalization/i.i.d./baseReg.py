import os, sys
sys.path.append('/home/project/Pertb_benchmark')
from myUtil import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA

import anndata as ad

import warnings
warnings.filterwarnings('ignore')

def generatePairedSample(adata, perturbation):
    cellTypes = list(adata.obs['condition1'].unique())
    annList_Pertb = []; annList_control = []
    for celltype in cellTypes:
        perturb_cells = adata[ (adata.obs['condition1'] == celltype) & (adata.obs['condition2'] == perturbation)]
        control_cells = adata[ (adata.obs['condition1'] == celltype) & (adata.obs['condition2'] ==  'control')]
        Nums = min(perturb_cells.shape[0], control_cells.shape[0])
        perturb_cells = perturb_cells[:Nums]
        control_cells = control_cells[:Nums]
        annList_Pertb.append(perturb_cells)
        annList_control.append(control_cells)
    annList_Pertb = ad.concat(annList_Pertb)
    annList_control = ad.concat(annList_control)
    return annList_Pertb, annList_control

def Kang_OutSample(DataSet, outSample):
    basePath = f'/home//project/Pertb_benchmark/DataSet/{DataSet}/inSample/hvg5000/baseReg/'
    tmp = '{}/{}'.format(basePath, outSample)
    if not os.path.isdir(tmp): os.makedirs(tmp)
    os.chdir(tmp)
    
    path = f'/home/project/Pertb_benchmark/DataSet/{DataSet}/filter_hvg5000_logNor.h5ad'
    tmp = sc.read_h5ad(path)
    perturbations = list(tmp.obs['condition2'].unique())
    perturbations = [i for i in perturbations if i != 'control']
    for perturbation in perturbations:

        fileout = '{}_imputed.h5ad'.format(perturbation)
        if os.path.isfile(fileout): continue

        adata = tmp[tmp.obs['condition2'].isin([perturbation, 'control'])]
        adata_train = adata[adata.obs['iid_test'] == 'train']
        adata_test  = adata[adata.obs['iid_test'] == 'test']

        Xtr_anndata, ytr_anndata = generatePairedSample(adata_train, perturbation)
        Xtr = Xtr_anndata.X;  ytr  = ytr_anndata.X
        Xte_anndata = adata_test[ (adata_test.obs['condition2'].isin(['control'])) & (adata_test.obs['condition1'].isin([outSample]))]
        yte_anndata = adata_test[ (adata_test.obs['condition2'].isin([perturbation])) & (adata_test.obs['condition1'].isin([outSample]))]
        Xte = Xte_anndata.X

        n_components = min(100, Xtr.shape[0])
        pca = PCA(n_components=n_components)
        Xtr = pca.fit_transform(Xtr)
        Xte = pca.transform(Xte)
        reg = Ridge(random_state=42).fit(Xtr, ytr)
        ypred = reg.predict(Xte)
        Xte_anndata.obs['perturbation'] = 'control'
        imputed = Xte_anndata.copy()
        imputed.X = ypred
        imputed.obs['perturbation'] = 'imputed'
        yte_anndata.obs['perturbation'] = 'stimulated'
        result = ad.concat([Xte_anndata, yte_anndata, imputed])
        result.write_h5ad(fileout)

 
def KangMain(DataSet):
    mylist = []
    filein_tmp = '/home/project/Pertb_benchmark/DataSet/{}/filter_hvg5000_logNor.h5ad'.format(DataSet)
    adata_tmp = sc.read_h5ad(filein_tmp)
    outSamples = list(adata_tmp.obs['condition1'].unique())
    for outSample in outSamples:
        Kang_OutSample(DataSet, outSample)

DataSets = ["kangCrossCell", "kangCrossPatient",  "Parekh", "Haber", "crossPatient", "KaggleCrossPatient",
            "KaggleCrossCell", "crossSpecies", "McFarland", "Afriat", "TCDD", "sciplex3"]

if __name__ == '__main__':
    for DataSet in tqdm(DataSets):
        KangMain(DataSet)