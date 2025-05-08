#coding:utf-8
import sys, subprocess, os
tmp = '/home/software/scELMo/PerturbationAnalysis'
sys.path.insert(0, tmp)

from gears import PertData, GEARS
from gears.inference import evaluate
import pickle

sys.path.append('/home/project/Pertb_benchmark')
from scipy import sparse
from myUtil import *
import anndata as ad
import torch
import shutil
from itertools import chain


sc.settings.verbosity = 3

def trainModel(adata, seed = 1):
    if not os.path.isdir('data'): os.makedirs('data')
    cmd = 'cp ../GEARS/data/gene2go_all.pkl                     data/'; subprocess.call(cmd, shell=True)
    cmd = 'cp ../GEARS/data/essential_all_data_pert_genes.pkl   data/'; subprocess.call(cmd, shell=True)
    cmd = 'cp -r ../GEARS/data/go_essential_all   data/'; subprocess.call(cmd, shell=True)

    pert_data = PertData('./data') # specific saved folder   下载 gene2go_all.pkl
    if not os.path.isfile('data/train/data_pyg/cell_graphs.pkl'):
        pert_data.new_data_process(dataset_name = 'train', adata = adata) # specific dataset name and adata object
    cmd = 'cp -r ../GEARS/data/train/splits/    data/train/'; subprocess.call(cmd, shell=True)
    
    pert_data.load(data_path = './data/train') # load the processed data, the path is saved folder + dataset_name
    pert_data.prepare_split(split = 'simulation', seed = seed, train_gene_set_size=.8) # get data split with seed
    pert_data.get_dataloader(batch_size = 32, test_batch_size = 128) # prepare data loader


    with open("/home/software/GeneCompass/human_emb_768.pickle", "rb") as fp:
        GPT_3_5_gene_embeddings = pickle.load(fp)

    gene_names= list(pert_data.adata.var['gene_name'].values)
    count_missing = 0
    EMBED_DIM = 768 # embedding dim from GPT-3.5
    lookup_embed = np.zeros(shape=(len(gene_names),EMBED_DIM))
    for i, gene in enumerate(gene_names):
        if gene in GPT_3_5_gene_embeddings:
            lookup_embed[i,:] = GPT_3_5_gene_embeddings[gene].flatten()
        else:
            count_missing+=1

    # set up and train a model
    gears_model = GEARS(pert_data, device = device, gene_emb = lookup_embed)
    gears_model.model_initialize(hidden_size = 64)
    gears_model.train(epochs = 15)  ### epochs
    gears_model.save_model('savedModels{}'.format(seed))
    return gears_model

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

def runGears(DataSet, redo=False, seed = 1):
    dirName = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/GeneCompass'.format(DataSet)
    if not os.path.isdir(dirName): os.makedirs(dirName)
    os.chdir(dirName)
    if redo and os.path.isdir('data/train'):
        shutil.rmtree('data/train')

    if os.path.isfile('savedModels{}/model.pt'.format(seed)): return
    if not os.path.isfile('data/train/data_pyg/cell_graphs.pkl'):
        adata = sc.read_h5ad('../../filter_hvg5000_logNor.h5ad')
        adata.uns['log1p'] = {}; adata.uns['log1p']["base"] = None
        adata = doGearsFormat(adata)
        trainModel(adata, seed = seed)
    else:
        trainModel(adata='', seed = seed)

def loadModel(dirName, seed):
    os.chdir(dirName)
    pert_data = PertData('./data')
    pert_data.load(data_path = './data/train') #

    with open("/home/software/GeneCompass/human_emb_768.pickle", "rb") as fp:
        GPT_3_5_gene_embeddings = pickle.load(fp)

    gene_names= list(pert_data.adata.var['gene_name'].values)
    count_missing = 0
    EMBED_DIM = 768 # embedding dim from GPT-3.5
    lookup_embed = np.zeros(shape=(len(gene_names),EMBED_DIM))
    for i, gene in enumerate(gene_names):
        if gene in GPT_3_5_gene_embeddings:
            lookup_embed[i,:] = GPT_3_5_gene_embeddings[gene].flatten()
        else:
            count_missing+=1

    pert_data.prepare_split(split = 'simulation', seed = seed, train_gene_set_size=.8)
    pert_data.get_dataloader(batch_size = 32, test_batch_size = 128)
    gears_model = GEARS(pert_data, device = device, gene_emb = lookup_embed)
    gears_model.load_pretrained('savedModels{}'.format(seed), gene_emb = lookup_embed)
    return gears_model

def remove_duplicates_and_preserve_order(input_list):
    deduplicated_dict = OrderedDict.fromkeys(input_list)
    deduplicated_list = list(deduplicated_dict.keys())
    return deduplicated_list

def getPredict(DataSet, seed):
    dirName = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/GeneCompass'.format(DataSet)
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
    adata1.X = test_res['pred']    ### 最为重要，更改为预测值，其余metadata 不变
    adata1.obs['Expcategory'] = 'imputed'

    adata_ctrl = gears_model.ctrl_adata
    adata_ctrl.obs['Expcategory'] = 'control'
    adata_fi = ad.concat([adata1, adata2, adata_ctrl])
    adata_fi.write('savedModels{}/result.h5ad'.format(seed))


seeds = [1, 2, 3]
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


SinglePertDataSets = ['Adamson', "Frangieh", "TianActivation", "TianInhibition", "Replogle_exp7", "Replogle_exp8", "Papalexi", "Replogle_RPE1essential", "Replogle_K562essential"]
CombPertDataSets = ['Norman', 'Wessels', 'Schmidt', "Replogle_exp6", "Replogle_exp10"]

'''
conda activate gears
'''

if __name__ == '__main__':
    print ('hello, world')

    for DataSet in ['Papalexi']:
        for seed in seeds:
            runGears(DataSet, redo=False, seed = seed)
            getPredict(DataSet, seed)