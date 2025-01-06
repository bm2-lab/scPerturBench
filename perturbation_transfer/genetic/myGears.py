#coding:utf-8
import sys, subprocess, os
sys.path.append('/home/wzt/project/Pertb_benchmark')
from scipy import sparse
from myUtil import *
from gears import PertData, GEARS
from gears.inference import evaluate, compute_metrics
import anndata as ad
import torch
import shutil
from itertools import chain


sc.settings.verbosity = 3

def trainModel(adata, issplit = False, seed = 1):
    pert_data = PertData('./data') # specific saved folder   下载 gene2go_all.pkl
    if not os.path.isfile('data/train/data_pyg/cell_graphs.pkl'):
        pert_data.new_data_process(dataset_name = 'train', adata = adata) # specific dataset name and adata object
    pert_data.load(data_path = './data/train') # load the processed data, the path is saved folder + dataset_name
    pert_data.prepare_split(split = 'simulation', seed = seed, train_gene_set_size=.8) # get data split with seed
    if issplit: return
    pert_data.get_dataloader(batch_size = 32, test_batch_size = 128) # prepare data loader

    # set up and train a model
    gears_model = GEARS(pert_data, device = device)
    gears_model.model_initialize(hidden_size = 64)
    gears_model.train(epochs = 15)  ### epochs
    gears_model.save_model('savedModels{}'.format(seed))
    return gears_model

###  为gears格式做准备
def doGearsFormat(adata):
    def fun1(x):
        if x == 'control': return 'ctrl'
        elif '+' in x:
            genes = x.split('+')
            return genes[0] + '+' + genes[1]
        else: return x + '+' + 'ctrl'
    adata.obs['cell_type'] = 'K562'
    adata.obs['condition'] = adata.obs['perturbation'].apply(lambda x: fun1(x))
    if 'gene_name' not in adata.var.columns:
        adata.var['gene_name'] = adata.var_names
    if not sparse.issparse(adata.X): adata.X = sparse.csr_matrix(adata.X)
    return adata

def runGears(DataSet, issplit=False, redo=False, seed = 1):
    dirName = '/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/{}/hvg5000/GEARS'.format(DataSet)
    if not os.path.isdir(dirName): os.makedirs(dirName)
    os.chdir(dirName)
    if redo and os.path.isdir('data/train'):
        shutil.rmtree('data/train')

    if os.path.isfile('savedModels{}/model.pt'.format(seed)): return
    if not os.path.isfile('data/train/data_pyg/cell_graphs.pkl'):
        adata = sc.read_h5ad('../../filter_hvg5000_logNor.h5ad')
        adata.uns['log1p'] = {}; adata.uns['log1p']["base"] = None
        adata = doGearsFormat(adata)
        trainModel(adata, issplit = issplit, seed = seed)
    else:
        trainModel(adata='', issplit = issplit, seed = seed)

def loadModel(dirName, seed):
    os.chdir(dirName)
    pert_data = PertData('./data')
    pert_data.load(data_path = './data/train') #
    pert_data.prepare_split(split = 'simulation', seed = seed, train_gene_set_size=.8)
    pert_data.get_dataloader(batch_size = 32, test_batch_size = 128)
    gears_model = GEARS(pert_data, device = device)
    gears_model.load_pretrained('savedModels{}'.format(seed))
    return gears_model

def remove_duplicates_and_preserve_order(input_list):
    deduplicated_dict = OrderedDict.fromkeys(input_list)
    deduplicated_list = list(deduplicated_dict.keys())
    return deduplicated_list

def getPredict(DataSet, seed):
    dirName = '/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/{}/hvg5000/GEARS'.format(DataSet)
    os.chdir(dirName)
    gears_model = loadModel(dirName, seed)
    adata = gears_model.adata
    test_loader = gears_model.dataloader['test_loader']
    test_res = evaluate(test_loader, gears_model.best_model, gears_model.config['uncertainty'], gears_model.device)  
    pert_cats = remove_duplicates_and_preserve_order(test_res['pert_cat'])

    adata_list = [adata[adata.obs['condition'] == pert_cat].copy()  for pert_cat in pert_cats ]
    adata2 = ad.concat(adata_list)
    adata2.obs['Expcategory'] = 'stimulated'

    adata_list = [adata[adata.obs['condition'] == pert_cat].copy()  for pert_cat in pert_cats ]
    adata1 = ad.concat(adata_list)
    adata1.X = test_res['pred']    ###
    adata1.obs['Expcategory'] = 'imputed'

    adata_ctrl = gears_model.ctrl_adata
    adata_ctrl.obs['Expcategory'] = 'control'
    adata_fi = ad.concat([adata1, adata2, adata_ctrl])
    adata_fi.write('savedModels{}/result.h5ad'.format(seed))


seeds = [1, 2, 3]
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


DataSet = 'Schmidt'
SinglePertDataSets = ['Adamson', "Frangieh", "TianActivation", "TianInhibition", "Replogle_exp7", "Replogle_exp8", "Papalexi", "Replogle_RPE1essential", "Replogle_K562essential"]
CombPertDataSets = ['Norman', 'Wessels', 'Schmidt', "Replogle_exp6", "Replogle_exp10"]

'''
conda activate gears  0.1.0   version
'''
