import sys, subprocess, os
sys.path.insert(0, '/home/software/PRnet')
from data._utils import rank_genes_groups_by_cov #type: ignore
from trainer.PRnetTrainer import PRnetTrainer  #type: ignore

sys.path.append('/home/project/Pertb_benchmark')
from myUtil1 import *
from scipy import sparse
import anndata as ad
import anndata
from sklearn.metrics import mean_squared_error
import pickle
import torch.nn as nn
import joblib
import torch
from collections import defaultdict, OrderedDict
sc.settings.verbosity = 3


def preData4PRnet_single(adata):
    adata.obs['dose'] = adata.obs['dose_value']
    adata.obs.loc[adata.obs['control'] == 1, 'dose'] = 0
    adata.obs_names = [str(i) for i in range(adata.shape[0])]
    control_index = list(adata.obs[adata.obs['control'] == 1].index.unique())

    adata.obs["split_ood_multi_task1"] = (adata.obs["split_ood_multi_task1"].astype(str).replace("test", "valid"))
    adata.obs["split_ood_multi_task1"] = adata.obs["split_ood_multi_task1"].astype("category")
    adata.obs["split_ood_multi_task2"] = (adata.obs["split_ood_multi_task2"].astype(str).replace("test", "valid"))
    adata.obs["split_ood_multi_task2"] = adata.obs["split_ood_multi_task2"].astype("category")
    adata.obs["split_ood_multi_task3"] = (adata.obs["split_ood_multi_task2"].astype(str).replace("test", "valid"))
    adata.obs["split_ood_multi_task3"] = adata.obs["split_ood_multi_task2"].astype("category")

    adata.obs["split_ood_multi_task1"] = (adata.obs["split_ood_multi_task1"].astype(str).replace("ood", "test"))
    adata.obs["split_ood_multi_task1"] = adata.obs["split_ood_multi_task1"].astype("category")
    adata.obs["split_ood_multi_task2"] = (adata.obs["split_ood_multi_task2"].astype(str).replace("ood", "test"))
    adata.obs["split_ood_multi_task2"] = adata.obs["split_ood_multi_task2"].astype("category")
    adata.obs["split_ood_multi_task3"] = (adata.obs["split_ood_multi_task2"].astype(str).replace("ood", "test"))
    adata.obs["split_ood_multi_task3"] = adata.obs["split_ood_multi_task2"].astype("category")

    import random
    random.seed(42)
    tmp_index=  []
    for index, dose in enumerate(adata.obs['dose']):
        if dose == 0:
            tmp_index.append(str(index))
        else:
            tmp_index.append(random.choice(control_index))
    adata.obs['paired_control_index'] = tmp_index
    return adata

def preData4PRnet_comb(adata):
    adata.obs['dose'] = 1000.0
    adata.obs['cov_drug'] = adata.obs['cov_drug_dose_name'].apply(lambda x: x.split('_')[0] + '_' + x.split('_')[1])
    adata.obs['SMILES'] = adata.obs['smiles_rdkit']

    adata.obs.loc[adata.obs['control'] == 1, 'dose'] = 0
    adata.obs_names = [str(i) for i in range(adata.shape[0])]
    control_index = list(adata.obs[adata.obs['control'] == 1].index.unique())

    adata.obs["split_ood_multi_task1"] = (adata.obs["split_ood_multi_task1"].astype(str).replace("val", "valid"))
    adata.obs["split_ood_multi_task1"] = adata.obs["split_ood_multi_task1"].astype("category")
    adata.obs["split_ood_multi_task2"] = (adata.obs["split_ood_multi_task2"].astype(str).replace("val", "valid"))
    adata.obs["split_ood_multi_task2"] = adata.obs["split_ood_multi_task2"].astype("category")
    adata.obs["split_ood_multi_task3"] = (adata.obs["split_ood_multi_task2"].astype(str).replace("val", "valid"))
    adata.obs["split_ood_multi_task3"] = adata.obs["split_ood_multi_task2"].astype("category")

    import random
    random.seed(42)
    tmp_index=  []
    for index, dose in enumerate(adata.obs['dose']):
        if dose == 0:
            tmp_index.append(str(index))
        else:
            tmp_index.append(random.choice(control_index))
    adata.obs['paired_control_index'] = tmp_index
    return adata



def TrainData(DataSet, seed, isComb = False):
    dirName = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/PRnet'.format(DataSet)
    if not os.path.isdir(dirName): os.makedirs(dirName)
    os.chdir(dirName)
    modeldir = 'savedModels{}'.format(seed)
    if os.path.isfile('savedModels{}/result.h5ad'.format(seed)): return
    if not os.path.isdir(modeldir): os.makedirs(modeldir)
    adata = sc.read_h5ad('../../filter_hvg5000.h5ad')
    adata.uns['log1p'] = {}
    adata.uns['log1p']['base'] = None
    if isComb:
        adata = preData4PRnet_comb(adata)
    else:
        adata = preData4PRnet_single(adata)

    config_kwargs = {
        'batch_size' : 512,
        'comb_num' : 1,
        'save_dir' :  modeldir,
        'n_epochs' : 100,
        'split_key' : "split_ood_multi_task{}".format(seed),
        'x_dimension' : adata.shape[1],
        'hidden_layer_sizes' : [128],
        'z_dimension' : 64,
        'adaptor_layer_sizes' : [128],
        'comb_dimension' : 64, 
        'drug_dimension': 1024,
        'dr_rate' : 0.05,
        'n_epochs' : 100,
        'lr' : 1e-3, 
        'weight_decay' : 1e-8,
        'scheduler_factor' : 0.5,
        'scheduler_patience' : 5,
        'n_genes' : 20,
        'loss' : ['GUSS'], 
        'obs_key' : 'cov_drug'
    }

    rank_genes_groups_by_cov(adata, groupby=config_kwargs['obs_key'], covariate='cell_type', control_group='control', n_genes=config_kwargs['n_genes'])

    Trainer = PRnetTrainer(
                        adata,
                        batch_size=config_kwargs['batch_size'],
                        comb_num=1,
                        split_key=config_kwargs['split_key'],
                        model_save_dir=config_kwargs['save_dir'],
                        x_dimension=config_kwargs['x_dimension'],
                        hidden_layer_sizes=config_kwargs['hidden_layer_sizes'],
                        z_dimension=config_kwargs['z_dimension'],
                        adaptor_layer_sizes=config_kwargs['adaptor_layer_sizes'],
                        comb_dimension=config_kwargs['comb_dimension'],
                        drug_dimension=config_kwargs['drug_dimension'],
                        dr_rate=config_kwargs['dr_rate'],
                        n_genes=config_kwargs['n_genes'],
                        loss = config_kwargs['loss'],
                        obs_key = config_kwargs['obs_key'])
    
    Trainer.train(n_epochs = config_kwargs['n_epochs'],
    lr = config_kwargs['lr'], 
    weight_decay= config_kwargs['weight_decay'], 
    scheduler_factor=config_kwargs['scheduler_factor'],
    scheduler_patience=config_kwargs['scheduler_patience'])
    Xtr, ytr, ypred, cov_drug_list, cov_drug_dose_name_list = Trainer.test(config_kwargs['save_dir'] + '/best.pt', return_dict=True)

    tmp_adata = sc.read_h5ad('../trainMean/savedModels{}/result.h5ad'.format(seed))
    tmp_imputed = tmp_adata[tmp_adata.obs['Expcategory'] == 'imputed']
    tmp_Noimputed = tmp_adata[tmp_adata.obs['Expcategory'] != 'imputed']
    
    df = pd.DataFrame({"cov_drug_dose_name": cov_drug_dose_name_list, "Expcategory":["imputed"] * len(cov_drug_dose_name_list)})

    tmp = ad.AnnData(X = ypred, obs=df, var = tmp_imputed.var)
    result_adata = ad.concat([tmp, tmp_Noimputed])
    result_adata.write('savedModels{}/result.h5ad'.format(seed))

SinglePertDataSets = ['sciplex3_A549', 'sciplex3_MCF7', 'sciplex3_K562']
CombPertDataSets = ['sciplex3_comb']

seeds = [1, 2, 3]
#torch.cuda.set_device('cuda:1')
# conda activate cpa


if __name__ == '__main__':
    print ('hello, world')
    for DataSet in ['sciplex3_A549']:
        for seed in tqdm(seeds):
            TrainData(DataSet, seed, isComb=False)

    # for DataSet in CombPertDataSets:
    #     for seed in tqdm(seeds):
    #         TrainData(DataSet, seed, isComb=True)


    