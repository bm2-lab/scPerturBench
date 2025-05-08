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
import warnings
warnings.filterwarnings('ignore')

def myPool(func, mylist, processes):
    with Pool(processes) as pool:
        results = list(tqdm(pool.imap(func, mylist), total=len(mylist)))
    return results

def transID(adata, species):
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    if species in ['Mmu', 'mouse']:
        dat = pd.read_csv('/home//database/ENSEMBL2SYMBOL_Mm.tsv', sep='\t')
    elif species == 'Hsa':
        dat = pd.read_csv('/home//database/ENSEMBL2SYMBOL_Hs.tsv', sep='\t')
    elif species == 'pig':
        dat = pd.read_csv('/home//database/ENSEMBL2SYMBOL_pig.tsv', sep='\t')
    elif species == 'rabbit':
        dat = pd.read_csv('/home//database/ENSEMBL2SYMBOL_rabbit.tsv', sep='\t')
    elif species == 'rat':
        dat = pd.read_csv('/home//database/ENSEMBL2SYMBOL_rat.tsv', sep='\t')
    if adata.var_names[0].startswith('ENS'):
        dat.set_index('ENSEMBL', inplace=True)
    else:
        dat.set_index('SYMBOL', inplace=True)
    dat = dat[~dat.index.duplicated()]
    adata.var = pd.merge(adata.var, dat, left_index=True, right_index=True,how='left')
    if adata.var_names[0].startswith('ENS'):
        adata.var['ENSEMBL'] = adata.var.index
        adata.var.set_index('SYMBOL', inplace=True)
        adata = adata[:, ~adata.var_names.isna()]
    adata.var = adata.var[['ENSEMBL', 'ENTREZID']]
    return adata

         
def preData(adata, filterNone=True, minNums = 30, filterCom=False, mtpercent = 10,  min_genes = 200, domaxNums=False, min_cells= 10):
    adata.var_names.astype(str)
    adata.var_names_make_unique()
    adata = adata[~adata.obs.index.duplicated()]
    if filterCom:
        tmp = adata.obs['perturbation'].apply(lambda x: True if ',' not in x else False);  adata = adata[tmp]
    if filterNone:
        adata = adata[adata.obs["perturbation"] != "None"]
    filterNoneNums = adata.shape[0]
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells= min_cells)
    filterCells = adata.shape[0]

    if np.any([True if i.startswith('mt-') else False for i in adata.var_names]):
        adata.var['mt'] = adata.var_names.str.startswith('mt-')
    else:
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    if sum(adata.obs['pct_counts_mt'] < 10) / adata.shape[0] <=0.5: mtpercent = 15
    adata = adata[adata.obs.pct_counts_mt < mtpercent, :]
    filterMT = adata.shape[0]
    tmp = adata.obs['perturbation'].value_counts()
    tmp_bool = tmp >= minNums
    genes = list(tmp[tmp_bool].index)
    if 'control' not in genes: genes += ['control']
    adata = adata[adata.obs['perturbation'].isin(genes), :]
    filterMinNums = adata.shape[0]
    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers['logNor'] = adata.X.copy()
    sc.pp.highly_variable_genes(adata, n_top_genes=500, subset=False)
    adata.var['highly_variable_500'] = adata.var['highly_variable']
    sc.pp.highly_variable_genes(adata, n_top_genes=1000, subset=False)
    adata.var['highly_variable_1000'] = adata.var['highly_variable']
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=False)
    adata.var['highly_variable_2000'] = adata.var['highly_variable']
    sc.pp.highly_variable_genes(adata, n_top_genes=5000, subset=False)
    adata.var['highly_variable_5000'] = adata.var['highly_variable']
    adata = adata[adata.obs.sort_values(by='perturbation').index,:]
    return filterNoneNums, filterCells, filterMT, filterMinNums, adata

def calDEG(DataSet = 'kang_pbmc', hvg=5000):
    os.chdir('/home/project/Pertb_benchmark/DataSet/{}'.format(DataSet))
    filein = 'filter_hvg{}_logNor.h5ad'.format(hvg)
    adata = sc.read_h5ad(filein)
    adata.uns['log1p']['base'] = None
    mydict = defaultdict(dict)
    outSamples = list(adata.obs['condition1'].unique())
    for outSample in outSamples:
        adata_tmp = adata[(adata.obs['condition1'] == outSample)].copy()
        perturbations = list(adata_tmp.obs['condition2'].unique())
        perturbations = [i for i in perturbations if i != 'control']
        sc.tl.rank_genes_groups(adata_tmp, 'condition2', groups=perturbations, reference='control', method= 't-test')
        result = adata_tmp.uns['rank_genes_groups']
        for perturbation in perturbations:
            final_result = pd.DataFrame({key: result[key][perturbation] for key in ['names', 'pvals_adj', 'logfoldchanges', 'scores']})
            tmp1 = 'foldchanges'
            tmp2 = 'logfoldchanges'
            final_result[tmp1] = 2 ** final_result[tmp2]
            final_result.drop(labels=[tmp2], inplace=True, axis=1)
            final_result.set_index('names', inplace=True)
            final_result['abs_scores'] = np.abs(final_result['scores'])
            final_result.sort_values('abs_scores', ascending=False, inplace=True)
            mydict[outSample][perturbation] = final_result
    import pickle
    with open('DEG_hvg{}.pkl'.format(hvg),'wb') as fout:
        pickle.dump(mydict, fout)


def getDEG(hvg, outSample, perturb, DEG):
    import pickle
    with open('DEG_hvg{}.pkl'.format(hvg), 'rb') as fin:
        mydict = pickle.load(fin)
        DegList = list(mydict[outSample][perturb].index[:DEG])
    return DegList



if __name__ == "__main__":
    print ('hello, world')
    pass