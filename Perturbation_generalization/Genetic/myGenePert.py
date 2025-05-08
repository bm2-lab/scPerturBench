import sys
sys.path.append('/home/project/Pertb_benchmark')
sys.path.append('//home/software/GenePert')
from myUtil import *
import torch
from collections import OrderedDict
from itertools import chain
import importlib
import matplotlib.pyplot as plt
import pickle, sklearn, umap                              
# Reload the module
import utils # type: ignore
import GenePertExperiment  #type: ignore
importlib.reload(utils)
# Reload the module
importlib.reload(GenePertExperiment)
from utils import get_best_overall_mse_corr, run_experiments_with_embeddings, plot_mse_corr_comparison, compare_embedding_correlations #type: ignore

import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from tqdm import tqdm
from sklearn.model_selection import KFold
from matplotlib.patches import Patch
from sklearn.metrics.pairwise import cosine_similarity
import json
import os



embedding_path =  "/home/software/GenePert/GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle"


def getcondition(DataSet, seed = 1):
    filein = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/GEARS/data/train/splits/train_simulation_{}_0.8.pkl'.format(DataSet, seed)
    with open(filein, 'rb') as fin:
        mydict = pickle.load(fin)
    train_conditions = mydict['train']
    test_conditions = mydict['test']
    return train_conditions, test_conditions

def clean_condition(condition):
    return condition.replace('+ctrl', '').replace('ctrl+', '').strip()

def populate_dicts(adata_subset, mean_dict):
    for condition in adata_subset.obs['condition'].unique():
        condition_mask = adata_subset.obs['condition'] == condition
        condition_data = adata_subset[condition_mask].X
        clean_cond = clean_condition(condition)
        mean_dict[clean_cond] = np.mean(condition_data, axis=0)

def doLinearModel(DataSet, seed = 1):
    dirName = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/GenePert'.format(DataSet)
    if not os.path.isdir(dirName): os.makedirs(dirName)
    os.chdir(dirName)

    dataset_path = "/home/project/Pertb_benchmark/DataSet2/{}/filter_hvg5000_logNor.h5ad".format(DataSet)
    experiment = GenePertExperiment.GenePertExperiment(embeddings=None)
    experiment.load_dataset(dataset_path)
    with open(embedding_path, "rb") as fp:
        embeddings = pickle.load(fp)
    experiment.embeddings = embeddings
    train_conditions, test_conditions = getcondition(DataSet, seed)
    embedding_size = len(next(iter(experiment.embeddings.values())))
    X_train, y_train, X_test, y_test = [], [], [], []
    train_mask = experiment.adata.obs["condition"].isin(train_conditions)
    test_mask = experiment.adata.obs["condition"].isin(test_conditions)
    adata_train = experiment.adata[train_mask]
    adata_test = experiment.adata[test_mask]
    mean_dict_train, mean_dict_test = {}, {}
    populate_dicts(adata_train, mean_dict_train)
    populate_dicts(adata_test, mean_dict_test)
    train_gene_name_X_map = experiment.populate_X_y(mean_dict_train, X_train, y_train, embedding_size)
    test_gene_name_X_map = experiment.populate_X_y(mean_dict_test, X_test, y_test, embedding_size)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    ridge_model = Ridge(alpha=1,  random_state=42)
    ridge_model.fit(X_train, y_train)
    y_pred = ridge_model.predict(X_test)
    result = pd.DataFrame(y_pred, columns=experiment.adata.var_names, index=mean_dict_test.keys())
    dirOut = 'savedModels{}'.format(seed)
    if not os.path.isdir(dirOut): os.makedirs(dirOut)
    result.to_csv('{}/pred.tsv'.format(dirOut),  sep='\t')


def generateExp(cellNum, means, std):
    expression_matrix = np.array([
    np.random.normal(loc=means[i], scale=std[i], size=cellNum) 
    for i in range(len(means))]).T
    return expression_matrix


### 根据预测的生成表达量
def generateH5ad(DataSet, seed = 1):
    dirName = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/GenePert'.format(DataSet)
    os.chdir(dirName)
    filein = 'savedModels{}/pred.tsv'.format(seed)
    exp = pd.read_csv(filein, sep='\t', index_col=0)
    filein = '../GEARS/savedModels{}/result.h5ad'.format(seed)
    adata = sc.read_h5ad(filein)
    expGene = np.intersect1d(adata.var_names, exp.columns)
    pertGenes = np.intersect1d(adata.obs['perturbation'].unique(), exp.index)
    adata = adata[:, expGene]; exp = exp.loc[:, expGene]

    control_exp = adata[adata.obs['perturbation'] == 'control'].to_df()
    control_std = list(np.std(control_exp))
    control_std = [i if not np.isnan(i) else 0 for i in control_std]
    for pertGene in pertGenes:
        cellNum = adata[(adata.obs['perturbation'] == pertGene) & (adata.obs['Expcategory']=='imputed')].shape[0]
        means = list(exp.loc[pertGene, ])
        expression_matrix = generateExp(cellNum, means, control_std)
        adata[(adata.obs['perturbation'] == pertGene) & (adata.obs['Expcategory']=='imputed')].X = expression_matrix
    adata.write('savedModels{}/result.h5ad'.format(seed))



### conda activate gears
seeds = [1, 2, 3]

SinglePertDataSets = ['Adamson', "Frangieh", "TianActivation", "TianInhibition", "Replogle_exp7", "Replogle_exp8", "Papalexi", "Replogle_RPE1essential", "Replogle_K562essential"]
CombPertDataSets = ['Norman', 'Wessels', 'Schmidt', "Replogle_exp6", "Replogle_exp10"]


if __name__ == '__main__':
    print ('hello, world')
    for myDataSet in ['Papalexi']:
        print (myDataSet)
        for seed in tqdm(seeds):
            doLinearModel(myDataSet, seed)
            #generateH5ad(myDataSet, seed)