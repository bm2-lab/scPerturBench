import sys
sys.path.append('/home/wzt/project/Pertb_benchmark')
from myUtil1 import *
import torch
import pickle
import torch
import anndata as ad
import pandas as pd
import numpy as np
import random
from scouter import Scouter, ScouterData


embedding_path =  "/home/wzt/software/GenePert/GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle"


def set_seeds(seed=24):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clean_condition(condition):
    return condition.replace('+ctrl', '').replace('ctrl+', '').strip()

def condition_sort(x):
    return '+'.join(sorted(x.split('+')))


def getPerturbs(adata):
    mylist  = []
    for perturb in adata.obs['perturbation']:
        perturbs = perturb.split('+')
        mylist.extend(perturbs)
    mylist = list(set(mylist))
    mylist = [i for i in mylist if i != 'control']
    return mylist

def getFullEmbd(embd, allPerturbs):
    WOEmbd = [i for i in allPerturbs if i not in list(embd.index)]
    np.random.seed(42)
    if len(WOEmbd) >= 1:
        print ('gene without embedding')
        print (WOEmbd)
        tmp = [pd.DataFrame([np.random.random(embd.shape[1])], columns=embd.columns, index=[tmp]) for tmp in WOEmbd]
        return pd.concat(tmp)
    else:
        return


def trainModel(DataSet, seed):
    dirName = '/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/{}/hvg5000/scouter'.format(DataSet)
    if not os.path.isdir(dirName): os.makedirs(dirName)
    os.chdir(dirName)
    adata = sc.read_h5ad('../GEARS/data/train/perturb_processed.h5ad')
    adata.obs['condition'] = adata.obs['condition'].astype(str).apply(lambda x: condition_sort(x)).astype('category')
    adata.uns = {}; adata.obs.drop('condition_name', axis=1, inplace=True)

    with open(embedding_path, 'rb') as f:
        embd = pd.DataFrame(pickle.load(f)).T
    allPerturbs = getPerturbs(adata)
    tmp = getFullEmbd(embd, allPerturbs)
    ctrl_row = pd.DataFrame([np.zeros(embd.shape[1])], columns=embd.columns, index=['ctrl'])
    embd_all = pd.concat([ctrl_row, tmp, embd])
    scouterdata = ScouterData(adata=adata, embd=embd_all, key_label='condition', key_var_genename='gene_name')
    scouterdata.setup_ad('embd_index')
    scouterdata.gene_ranks()
    scouterdata.get_dropout_non_zero_genes()

    filein = '../GEARS/data/train/splits/train_simulation_{}_0.8.pkl'.format(seed)
    with open(filein, 'rb') as fin:
        splits = pickle.load(fin)
    scouterdata.split_Train_Val_Test(val_conds = splits['val'], test_conds = splits['test'])

    scouter_model = Scouter(scouterdata)
    scouter_model.model_init()
    scouter_model.train(n_epochs = 40)
    pred_dict = scouter_model.pred(pert_list = splits['test'], n_pred = 500)

    adata_list = []
    for pertGene in pred_dict:
        tmp_adata = ad.AnnData(pred_dict[pertGene])
        tmp_adata.obs['perturbation'] = clean_condition(pertGene)
        adata_list.append(tmp_adata)
    adata_pred = ad.concat(adata_list)
    adata_pred.obs['Expcategory'] = 'imputed'
    adata_pred.var = adata.var

    adata_control = adata[adata.obs['perturbation'] == 'control']
    adata_control.obs['Expcategory'] = 'control'
    adata_truth = adata[adata.obs['perturbation'].isin(list(adata_pred.obs['perturbation'].unique()))]
    adata_truth.obs['Expcategory'] = 'stimulated'
    result = ad.concat([adata_pred, adata_control, adata_truth])
    dirOut = 'savedModels{}'.format(seed)
    if not os.path.isdir(dirOut): os.makedirs(dirOut)
    result.write_h5ad('{}/result.h5ad'.format(dirOut))


### conda activate cpa
seeds = [1, 2, 3]
set_seeds(24)
SinglePertDataSets = ['Adamson', "Frangieh", "TianActivation", "TianInhibition", "Replogle_exp7", "Replogle_exp8", "Papalexi", "Replogle_RPE1essential", "Replogle_K562essential"]
CombPertDataSets = ['Norman', 'Wessels', 'Schmidt', "Replogle_exp6", "Replogle_exp10"]
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")