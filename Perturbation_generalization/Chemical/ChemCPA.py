import os, json, torch, sys
import numpy as np
import pandas as pd
import scanpy as sc

# /NFS2_home/NFS2_home_1/wzt/software/chemCPA/chemCPA/experiments_run.py
os.chdir('/NFS_home/NFS_home_2/wzt/software/chemCPA')
sys.path.append('/NFS_home/NFS_home_2/wzt/software/chemCPA')
from chemCPA.data import load_dataset_splits #type: ignore
from tqdm.auto import tqdm
#from chemCPA.paths import FIGURE_DIR, ROOT, PROJECT_DIR
PROJECT_DIR = '/NFS_home/NFS_home_2/wzt/software/chemCPA/project_folder'

from chemCPA.train import bool2idx, compute_prediction, compute_r2, repeat_n #type: ignore
from MyUtils import *  #type: ignore

from pathlib import Path
from pprint import pprint

from seml.config import generate_configs, read_config  #type: ignore
from chemCPA.experiments_run import ExperimentWrapper  #type: ignore



'''
[True if '+' in i else False for i in adata.obs['condition']]
'''


### 训练模型
def runChemCPA(DataSet, seed):
    os.chdir('/home/wzt/software/chemCPA')  ###
    exp = ExperimentWrapper(init_all=False)
    # this is how seml loads the config file internally
    *_, experiment_config = read_config("/home/wzt/software/chemCPA/manual_run.yaml")
    filein_h5ad = '/home/wzt/project/Pertb_benchmark/DataSet2/{}/filter_hvg5000.h5ad'.format(DataSet)
    experiment_config['rdkit_model']['fixed']['dataset.data_params.dataset_path'] = filein_h5ad
    experiment_config['fixed']['dataset.data_params.split_key'] = 'split_ood_multi_task{}'.format(seed)
    # we take the first config generated
    configs = generate_configs(experiment_config)
    if len(configs) > 1:
        print("Careful, more than one config generated from the yaml file")
    args = configs[0]
    args['training']['file_name'] = 'output.pt'  #
    save_dir = '/home/wzt/project/Pertb_benchmark/DataSet2/{}/hvg5000/chemCPA/savedModels{}'.format(DataSet, seed)
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    args['training']['save_dir'] = save_dir

    exp.seed = 1337
    # loads the dataset splits
    exp.init_dataset(**args["dataset"])

    exp.init_drug_embedding(embedding=args["model"]["embedding"])
    exp.init_model(
        hparams=args["model"]["hparams"],
        additional_params=args["model"]["additional_params"],
        load_pretrained=args["model"]["load_pretrained"],
        append_ae_layer=args["model"]["append_ae_layer"],
        enable_cpa_mode=args["model"]["enable_cpa_mode"],
        pretrained_model_path=args["model"]["pretrained_model_path"],
        pretrained_model_hashes=args["model"]["pretrained_model_hashes"],
    )
    # setup the torch DataLoader
    exp.update_datasets()
    exp.train(**args["training"])


###  predict ood
def load_config(seml_collection, model_hash):
    file_path = '{}/datasets/{}.json'.format(PROJECT_DIR, seml_collection)  # Provide path to json

    with open(file_path) as f:
        file_data = json.load(f)

    for _config in tqdm(file_data):
        if _config["config_hash"] == model_hash:
            config = _config["config"]
            config["config_hash"] = _config["config_hash"]
    return config

def predictChemCPA(DataSet, seed = 1):
    import anndata
    config = load_config("chemCPA_configs", "c824e42f7ce751cf9a8ed26f0d9e0af7")
    filein_h5ad = '/home/wzt/project/Pertb_benchmark/DataSet2/{}/filter_hvg5000.h5ad'.format(DataSet)
    config["dataset"]["data_params"]["dataset_path"] = filein_h5ad
    config['dataset']['data_params']['degs_key'] = 'all_DEGs'
    config['dataset']['data_params']['split_key'] = 'split_ood_multi_task{}'.format(seed)

    config['config_hash']= 'output'
    config["model"]["append_ae_layer"] = True
    save_dir = '/home/wzt/project/Pertb_benchmark/DataSet2/{}/hvg5000/chemCPA/savedModels{}'.format(DataSet, seed)
    config['model']['pretrained_model_path'] = save_dir
    config['model']['pretrained_model_hashes']['rdkit'] = 'output'
    config["model"]["embedding"]["directory"] = "/home/wzt/software/chemCPA/project_folder/embeddings"
    config['training']['save_dir'] =  save_dir

    dataset, key_dict = load_dataset(config)  #type: ignore
    config['dataset']['n_vars'] = dataset.n_vars

    canon_smiles_unique_sorted, smiles_to_pathway_map, smiles_to_drug_map = load_smiles(config, dataset, key_dict, True)  #type: ignore
    ood_drugs = dataset.obs.condition[dataset.obs[config["dataset"]["data_params"]["split_key"]].isin(['ood'])].unique().to_list()

    data_params = config['dataset']['data_params']
    datasets = load_dataset_splits(**data_params, return_dataset=False)

    #dosages = [0.001, 0.01, 0.1, 1.0]   ### 预测的浓度列表，即如果ood不在这个列表里面，不进行预测
    #cell_lines = ["A549", "K562", "MCF7"]  ### ### 预测的细胞系，即如果ood不在这个列表里面，不进行预测
    dosages = list(np.unique(datasets['training'].dosages))
    dosages = [i for i in dosages if i !=0]
    cell_lines = list(np.unique(datasets['training'].covariate_names['cell_type']))
    
    model_pretrained, embedding_pretrained = load_model(config, canon_smiles_unique_sorted)  #type: ignore
    results = compute_pred1(model = model_pretrained,     #type: ignore
                            dataset = datasets['ood'],
                            genes_control=datasets['test_control'].genes,  ### 对照的表达谱.genes访问表达谱
                            dosages=dosages,
                            cell_lines=cell_lines,
                            use_DEGs=False,
                            verbose=False,
                        )

    adata = sc.read_h5ad(filein_h5ad)
    adata_control = adata[adata.obs['perturbation'] == 'control'].copy()
    adata_control.obs['Expcategory'] = 'control'

    adata_treat = adata[adata.obs['split_ood_multi_task{}'.format(seed)] == 'ood'].copy()
    adata_treat.obs['Expcategory'] = 'stimulated'

    h5ad_list = []
    for cov_drug_dose_name in results:
        tmp = anndata.AnnData(np.array(results[cov_drug_dose_name]), var=adata.var)
        tmp.obs['cov_drug_dose_name'] = cov_drug_dose_name
        h5ad_list.append(tmp)
    adata1 = anndata.concat(h5ad_list)
    adata1.obs['Expcategory'] = 'imputed'
    pred = anndata.concat([adata_control, adata_treat, adata1])
    pred.var = adata.var
    pred.write_h5ad('{}/result.h5ad'.format(save_dir))
    

torch.cuda.set_device('cuda:1')
DataSets = ['sciplex3_MCF7', 'sciplex3_K562', 'sciplex3_A549']
seeds = [1, 2, 3]

'''
conda activate chemCPA
'''

if __name__ == '__main__':
    print ('hello, world')
    for DataSet in DataSets:
        for seed in seeds:
            runChemCPA(DataSet, seed)      
            predictChemCPA(DataSet, seed)
