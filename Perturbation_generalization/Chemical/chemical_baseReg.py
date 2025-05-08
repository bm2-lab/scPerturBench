import sys
sys.path.append('/home/project/Pertb_benchmark')
from myUtil import *
from collections import OrderedDict
from itertools import chain
import pickle
import torch.nn as nn
import numpy as np
from sklearn.linear_model import Ridge
from tqdm import tqdm
import os

def clean_condition(condition):
    return condition.replace('+ctrl', '').replace('ctrl+', '').strip()

def getData(tmp_train, adata_train, embeddings):
    ytr = np.vstack([
        adata_train[adata_train.obs['cov_drug_dose_name'] == drug].X.mean(axis=0) 
        for drug in tmp_train
    ])

    Xtr = np.vstack([embeddings[drug] for drug in tmp_train])
    return Xtr, np.array(ytr)



def doLinearModel_single(DataSet, seed = 1):
    dirName = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/baseReg'.format(DataSet)
    if not os.path.isdir(dirName): os.makedirs(dirName)
    os.chdir(dirName)

    dirOut = 'savedModels{}'.format(seed)
    fileout = '{}/pred.tsv'.format(dirOut)
    if os.path.isfile(fileout): return

    dataset_path = "/home/project/Pertb_benchmark/DataSet2/{}/filter_hvg5000_logNor.h5ad".format(DataSet)
    embedding_path =  "/home/project/Pertb_benchmark/DataSet2/{}/chemicalEmbedding.pkl".format(DataSet)
    with open(embedding_path, "rb") as fp:
        embeddings = pickle.load(fp)
    adata = sc.read_h5ad(dataset_path)
    adata = adata[adata.obs['condition'] != 'control']

    tmp = 'split_ood_multi_task{}'.format(seed)
    adata_train = adata[adata.obs[tmp] == 'train']
    adata_test = adata[adata.obs[tmp] == 'ood']

    tmp_train = list(adata_train.obs['cov_drug_dose_name'].unique())
    tmp_test = list(adata_test.obs['cov_drug_dose_name'].unique())
    Xtr, ytr = getData(tmp_train, adata_train, embeddings)
    Xte, yte = getData(tmp_test, adata_test, embeddings)

    ridge_model = Ridge(random_state=42)
    ridge_model.fit(Xtr, ytr)
    ypred = ridge_model.predict(Xte)
    
    result = pd.DataFrame(ypred, columns=adata.var_names, index= tmp_test)
    if not os.path.isdir(dirOut): os.makedirs(dirOut)
    result.to_csv(fileout, sep='\t')




def doLinearModel_comb(DataSet, seed = 1):
    dirName = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/baseReg'.format(DataSet)
    if not os.path.isdir(dirName): os.makedirs(dirName)
    os.chdir(dirName)

    dirOut = 'savedModels{}'.format(seed)
    fileout = '{}/pred.tsv'.format(dirOut)
    if os.path.isfile(fileout): return

    dataset_path = "/home/project/Pertb_benchmark/DataSet2/{}/filter_hvg5000_logNor.h5ad".format(DataSet)
    embedding_path =  "/home/project/Pertb_benchmark/DataSet2/{}/chemicalEmbedding.pkl".format(DataSet)
    with open(embedding_path, "rb") as fp:
        embeddings = pickle.load(fp)
    adata = sc.read_h5ad(dataset_path)
    adata = adata[adata.obs['condition'] != 'control']

    tmp = 'split_ood_multi_task{}'.format(seed)
    adata_train = adata[adata.obs[tmp] == 'train']
    adata_test = adata[adata.obs[tmp] == 'test']

    tmp_train = list(adata_train.obs['cov_drug_dose_name'].unique())
    tmp_test = list(adata_test.obs['cov_drug_dose_name'].unique())
    Xtr, ytr = getData(tmp_train, adata_train, embeddings)
    Xte, yte = getData(tmp_test, adata_test, embeddings)

    ridge_model = Ridge(random_state=42)
    ridge_model.fit(Xtr, ytr)
    ypred = ridge_model.predict(Xte)
    
    result = pd.DataFrame(ypred, columns=adata.var_names, index= tmp_test)
    if not os.path.isdir(dirOut): os.makedirs(dirOut)
    result.to_csv(fileout, sep='\t')





def generateExp(cellNum, means, std):
    expression_matrix = np.array([
    np.random.normal(loc=means[i], scale=std[i], size=cellNum) 
    for i in range(len(means))]).T
    return expression_matrix


def generateH5ad(DataSet, seed = 1, senario='trainMean'):
    import anndata
    dirName = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/{}'.format(DataSet, senario)
    os.chdir(dirName)
    filein = 'savedModels{}/pred.tsv'.format(seed)
    exp = pd.read_csv(filein, sep='\t', index_col=0)
    filein = '../../filter_hvg5000.h5ad'
    adata = sc.read_h5ad(filein)
    expGene = np.intersect1d(adata.var_names, exp.columns)
    pertGenes = np.intersect1d(adata.obs['cov_drug_dose_name'].unique(), exp.index)
    adata = adata[:, expGene]; exp = exp.loc[:, expGene]

    control_exp = adata[adata.obs['perturbation'] == 'control'].to_df()
    control_std = list(np.std(control_exp))
    control_std = [i if not np.isnan(i) else 0 for i in control_std]

    pred_list = []
    for pertGene in tqdm(pertGenes):
        cellNum = adata[adata.obs['cov_drug_dose_name'] == pertGene].shape[0]
        means = list(exp.loc[pertGene, ])
        expression_matrix = generateExp(cellNum, means, control_std)
        pred = anndata.AnnData(expression_matrix, var=adata.var)
        pred.obs['cov_drug_dose_name'] = pertGene
        pred_list.append(pred)
    pred_results = anndata.concat(pred_list)
    pred_results.obs['Expcategory'] = 'imputed'

    control_adata = adata[adata.obs['perturbation'] == 'control'].copy()
    control_adata.obs['Expcategory'] = 'control'

    stimulated_adata = adata[adata.obs['cov_drug_dose_name'].isin(pertGenes)]
    stimulated_adata.obs['Expcategory'] = 'stimulated'
    pred_fi = anndata.concat([pred_results, control_adata, stimulated_adata])
    pred_fi.write('savedModels{}/result.h5ad'.format(seed))

### conda activate cpa
SinglePertDataSets = ['sciplex3_MCF7', "sciplex3_A549", "sciplex3_K562"]
CombPertDataSets = ['sciplex3_comb']

if __name__ == '__main__':
    print ('hello, world')

#### single
    for DataSet in tqdm(['sciplex3_A549']):
        seeds = [1, 2, 3]
        print (DataSet)
        for seed in tqdm(seeds):
            doLinearModel_single(DataSet, seed)
            #generateH5ad(DataSet, seed, senario='baseReg')


### comb
    # for DataSet in tqdm(CombPertDataSets):
    #     seeds = [1, 2, 3]
    #     print (DataSet)
    #     for seed in tqdm(seeds):
    #         doLinearModel_comb(DataSet, seed)
    #         generateH5ad(DataSet, seed, senario='baseReg')

