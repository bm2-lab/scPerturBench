import subprocess, os, sys, re, glob
from collections import defaultdict
import numpy as np, pandas as pd
from multiprocessing import Pool
import multiprocessing
from sklearn.utils import shuffle
from tqdm import tqdm
from scipy.sparse import csr_matrix, issparse, spmatrix
from anndata import AnnData
from scipy import sparse
import anndata as ad
import scanpy as sc
import logging
from collections import defaultdict, OrderedDict
from itertools import chain
import warnings
warnings.filterwarnings('ignore')
import pickle

def clean_condition(condition):
    return condition.replace('+ctrl', '').replace('ctrl+', '').strip()

def subSample(adata, n_samples):
    if adata.shape[0] <= n_samples:
        return adata
    else:
        sampled_indices = np.random.choice(adata.n_obs, n_samples, replace=False)
        adata_sampled = adata[sampled_indices, :]
        return adata_sampled


def preData(adata, domaxNumsPerturb=0, domaxNumsControl=0, minNums = 50, min_cells= 10):
    adata.var_names.astype(str)
    adata.var_names_make_unique()
    adata = adata[~adata.obs.index.duplicated()]
    adata = adata[adata.obs["perturbation"] != "None"]
    filterNoneNums = adata.shape[0]
    sc.pp.filter_cells(adata, min_genes= 200)
    sc.pp.filter_genes(adata, min_cells= min_cells)
    filterCells = adata.shape[0]

    if np.any([True if i.startswith('mt-') else False for i in adata.var_names]):
        adata.var['mt'] = adata.var_names.str.startswith('mt-')
    else:
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    if sum(adata.obs['pct_counts_mt'] < 10) / adata.shape[0] <=0.5:
        adata = adata[adata.obs.pct_counts_mt < 15, :]
    else:
        adata = adata[adata.obs.pct_counts_mt < 10, :]
    filterMT = adata.shape[0]
    tmp = adata.obs['perturbation'].value_counts()
    tmp_bool = tmp >= minNums
    genes = list(tmp[tmp_bool].index)
    if 'control' not in genes: genes += ['control']
    adata = adata[adata.obs['perturbation'].isin(genes), :]
    filterMinNums = adata.shape[0]

    if domaxNumsPerturb:
        adata1 = adata[adata.obs['perturbation'] == 'control']
        perturbations = adata.obs['perturbation'].unique()
        perturbations = [i for i in perturbations if i != 'control']
        adata_list = []
        for perturbation in perturbations:
            adata_tmp = adata[adata.obs['perturbation'] == perturbation]
            adata_tmp = subSample(adata_tmp, domaxNumsPerturb)
            adata_list.append(adata_tmp)
        adata2 = ad.concat(adata_list)
        adata = ad.concat([adata1, adata2])
        adata.var = adata1.var.copy()
    if domaxNumsControl:
        adata1 = adata[adata.obs['perturbation'] == 'control']
        adata2 = adata[adata.obs['perturbation'] != 'control']
        adata1 = subSample(adata1, domaxNumsControl)
        adata = ad.concat([adata1, adata2])
        adata.var = adata1.var.copy()

    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4) 
    sc.pp.log1p(adata)
    adata.layers['logNor'] = adata.X.copy()
    sc.pp.highly_variable_genes(adata, n_top_genes=5000, subset=False)
    adata.var['highly_variable_5000'] = adata.var['highly_variable']
    adata = adata[adata.obs.sort_values(by='perturbation').index,:]
    return filterNoneNums, filterCells, filterMT, filterMinNums, adata

def calDEG(DataSet = 'Adamson', condition_column = 'perturbation', control_tag = 'control'):
    os.chdir('/home/project/Pertb_benchmark/DataSet2/{}'.format(DataSet))
    filein = 'filter_hvg5000_logNor.h5ad'
    adata = sc.read_h5ad(filein)
    adata.uns['log1p'] = {}
    adata.uns['log1p']['base'] = None
    mydict = defaultdict(dict)
    perturbations = adata.obs[condition_column].unique()
    perturbations = [i for i in perturbations if i != control_tag]
    sc.tl.rank_genes_groups(adata, condition_column, groups=perturbations, reference= control_tag, method= 't-test')
    result = adata.uns['rank_genes_groups']
    for perturbation in perturbations:
        final_result = pd.DataFrame({key: result[key][perturbation] for key in ['names', 'pvals_adj', 'logfoldchanges', 'scores']})
        tmp1 = 'foldchanges'
        tmp2 = 'logfoldchanges'
        final_result[tmp1] = 2 ** final_result[tmp2]
        final_result.drop(labels=[tmp2], inplace=True, axis=1)
        final_result.set_index('names', inplace=True)
        final_result['abs_scores'] = np.abs(final_result['scores'])
        final_result.sort_values('abs_scores', ascending=False, inplace=True)
        mydict[perturbation] = final_result
    import pickle
    fileout= 'DEG_hvg5000.pkl'
    with open(fileout,'wb') as fout:
        pickle.dump(mydict, fout)


SinglePertDataSets = ['Adamson', "Frangieh", "TianActivation", "TianInhibition", "Replogle_exp7", "Replogle_exp8", "Papalexi", "Replogle_RPE1essential", "Replogle_K562essential"]
CombPertDataSets = ['Norman', 'Wessels', 'Schmidt', "Replogle_exp6"]

chemicalDataSets_single = ['sciplex3_MCF7', 'sciplex3_A549', 'sciplex3_K562']
chemicalDataSets_comb = ['sciplex3_comb']


if __name__ == "__main__":
    print ('hello, world')
