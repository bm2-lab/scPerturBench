import sys
sys.path.append('/home/project/Pertb_benchmark')
sys.path.insert(0, '/home/software/scFoundation/GEARS')
from myUtil1 import *
import torch
import pickle
from collections import OrderedDict
import os
from gears import PertData, GEARS
from gears.inference import evaluate
from scipy import sparse

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
    if not sparse.issparse(adata.X): adata.X = sparse.csr_matrix(adata.X) ### 转换为稀疏矩阵
    return adata


def runFoundation(DataSet, seed):
    dirName = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/scFoundation'.format(DataSet)
    if not os.path.isdir(dirName): os.makedirs(dirName)
    os.chdir(dirName)
    h5ad_tmp = 'savedModels{}/result.h5ad'.format(seed)
    if os.path.isfile(h5ad_tmp):
        print ("already trained!")
        return
    if not os.path.isdir('data'): os.makedirs('data')
    if not os.path.isdir('data/train'): os.makedirs('data/train')
    if not os.path.isdir('data/train/splits'): os.makedirs('data/train/splits')
    cmd = 'ln -sf /home/software/scFoundation/GEARS/data/gene2go.pkl  data/gene2go.pkl'; subprocess.call(cmd, shell=True)
    cmd = 'ln -sf  /home/software/scFoundation/GEARS/data/go.csv  data/train/go.csv'; subprocess.call(cmd, shell=True)

    if not os.path.isfile('data/train/data_pyg/cell_graphs.pkl'):
        adata = sc.read_h5ad('../../filter_hvgall_logNor.h5ad')
        adata.uns['log1p'] = {}; adata.uns['log1p']["base"] = None
        adata = doGearsFormat(adata)
        pert_data = PertData('./data') # specific saved folder   download gene2go_all.pkl
        pert_data.new_data_process(dataset_name = 'train', adata = adata) # specific dataset name and adata object
    else:
        pert_data = PertData('./data') # specific saved folder   download gene2go_all.pkl
    pert_data.load(data_path = './data/train') # load the processed data, the path is saved folder + dataset_name

    tmp_dir1 = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/scFoundation/data/train/splits'.format(DataSet)
    tmp_dir2 = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/GEARS/data/train/splits/'.format(DataSet)
    if not os.path.isdir(tmp_dir1): os.makedirs(tmp_dir1)
    if not os.path.isdir(tmp_dir2): return
    cmd = 'cp   {}/*  {}'.format(tmp_dir2, tmp_dir1)
    subprocess.call(cmd, shell= True)

    pert_data.prepare_split(split = 'simulation', seed = seed, train_gene_set_size=.8) # get data split with seed
    pert_data.get_dataloader(batch_size = batch_size, test_batch_size = 16) # prepare data loader

    # set up and train a model
    gears_model = GEARS(pert_data, device = device)
    gears_model.model_initialize(hidden_size = 512, 
                                model_type = 'maeautobin',
                                bin_set= "autobin_resolution_append",
                                load_path= "/home/software/scFoundation/models.ckpt",
                                finetune_method="frozen",
                                accumulation_steps= 5,
                                mode= 'v1',
                                highres=0)

    gears_model.train(epochs = epochs, lr=0.0005)  ### epochs
    gears_model.save_model('savedModels{}'.format(seed))


def loadModel(dirName, seed):
    os.chdir(dirName)
    pert_data = PertData('./data')
    pert_data.load(data_path = './data/train') #
    pert_data.prepare_split(split = 'simulation', seed = seed, train_gene_set_size=.8)
    pert_data.get_dataloader(batch_size = batch_size, test_batch_size = 2)
    gears_model = GEARS(pert_data, device = device)
    gears_model.load_pretrained('savedModels{}'.format(seed))
    return gears_model

def remove_duplicates_and_preserve_order(input_list):
    deduplicated_dict = OrderedDict.fromkeys(input_list)
    deduplicated_list = list(deduplicated_dict.keys())
    return deduplicated_list

def getPredict(DataSet, seed):
    dirName = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/scFoundation'.format(DataSet)
    os.chdir(dirName)
    h5ad_tmp = 'savedModels{}/result.h5ad'.format(seed)
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
    adata1.X = test_res['pred']
    adata1.obs['Expcategory'] = 'imputed'

    adata_ctrl = gears_model.ctrl_adata
    adata_ctrl.obs['Expcategory'] = 'control'
    adata_fi = ad.concat([adata1, adata2, adata_ctrl])
    adata_fi.write_h5ad('savedModels{}/result.h5ad'.format(seed))


SinglePertDataSets = ['Adamson', "Frangieh", "TianActivation", "TianInhibition", "Replogle_exp7", "Replogle_exp8", "Papalexi", "Replogle_RPE1essential", "Replogle_K562essential"]
CombPertDataSets = ['Norman', 'Wessels', 'Schmidt', "Replogle_exp6", 'Replogle_exp10']

'''
conda activate gears
'''

epochs = 15
batch_size = 2
seeds = [1, 2, 3]
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
DataSet = 'Papalexi'

if __name__ == '__main__':
    print ('hello, world')
    for seed in seeds:
        print (DataSet, seed)
        runFoundation(DataSet, seed = seed)    ####
        getPredict(DataSet, seed = seed)