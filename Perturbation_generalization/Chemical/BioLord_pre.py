import sys
import pandas as pd
import pickle
import numpy as np
import anndata
from gears import PertData, GEARS
import torch
sys.path.append('/home/wzt/project/Pertb_benchmark')
from myUtil1 import *

def prepareData_single(DataSet):
    dirName = '/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/{}/hvg5000/GEARS'.format(DataSet)
    os.chdir(dirName)
    pert_data = PertData('./data') # specific saved folder   下载 gene2go_all.pkl
    pert_data.load(data_path = './data/train') # load the processed data, the path is saved folder + dataset_name
    adata = pert_data.adata.copy()

    for seed in [1, 2, 3]:
        with open("data/train/splits/train_simulation_{}_0.8.pkl".format(seed), "rb") as f:
            split_data = pickle.load(f)
        
        pert2set = {}
        for i,j in split_data.items():
            for x in j:
                pert2set[x] = i
        
        with open("data/train/splits/train_simulation_{}_0.8_subgroup.pkl".format(seed), "rb") as f:
            subgroup = pickle.load(f)
        adata.obs[f"split{seed}"] = [pert2set[i] for i in adata.obs["condition"].values]
        pert2subgroup = {}
        for i,j in subgroup["test_subgroup"].items():
            for x in j:
                pert2subgroup[x] = i
        
        adata.obs[f"subgroup{seed}"] = adata.obs["condition"].apply(lambda x: pert2subgroup[x] if x in pert2subgroup else 'Train/Val')
        rename = {
            'train': 'train',
            'test': 'ood',
            'val': 'test'
        }
        adata.obs[f'split{seed}'] = adata.obs[f'split{seed}'].apply(lambda x: rename[x])

    adata.obs["perturbation_bioLord"] = [cond.split("+")[0] for cond in adata.obs["condition"]]
    adata.obs["perturbation_bioLord"] = adata.obs["perturbation_bioLord"].astype("category")

    go_path = 'data/go_essential_all/go_essential_all.csv'
    gene_path = 'data/essential_all_data_pert_genes.pkl'
    df = pd.read_csv(go_path)
    df = df.groupby('target').apply(lambda x: x.nlargest(20 + 1, ['importance'])).reset_index(drop = True)
    with open(gene_path, 'rb') as f:
        gene_list = pickle.load(f)
    df = df[df["source"].isin(gene_list)]

    def get_map(pert):
        tmp = pd.DataFrame(np.zeros(len(gene_list)), index=gene_list)
        tmp.loc[df[df.target == pert].source.values, :] = df[df.target == pert].importance.values[:, np.newaxis]
        return tmp.values.flatten()    
    pert2neighbor =  {i: get_map(i) for i in list(adata.obs["perturbation_bioLord"].cat.categories)}    
    adata.uns["pert2neighbor"] = pert2neighbor

    pert2neighbor = np.asarray([val for val in adata.uns["pert2neighbor"].values()])
    keep_idx = pert2neighbor.sum(0) > 0
    name_map = dict(adata.obs[["condition", "condition_name"]].drop_duplicates().values)
    ctrl = np.asarray(adata[adata.obs["condition"].isin(["ctrl"])].X.mean(0)).flatten()

    df_perts_expression = pd.DataFrame(adata.X.A, index=adata.obs_names, columns=adata.var_names)
    df_perts_expression["condition"] = adata.obs["condition"]
    df_perts_expression = df_perts_expression.groupby(["condition"]).mean()
    df_perts_expression = df_perts_expression.reset_index()

    single_perts_condition = []
    single_pert_val = []
    double_perts = []
    for pert in adata.obs["condition"].cat.categories:
        if len(pert.split("+")) == 1:
            continue
        elif "ctrl" in pert:
            single_perts_condition.append(pert)
            p1, p2 = pert.split("+")
            if p2 == "ctrl":
                single_pert_val.append(p1)
            else:
                single_pert_val.append(p2)
    single_perts_condition.append("ctrl")
    single_pert_val.append("ctrl")

    df_singleperts_expression = pd.DataFrame(df_perts_expression.set_index("condition").loc[single_perts_condition].values, index=single_pert_val)
    df_singleperts_emb = np.asarray([adata.uns["pert2neighbor"][p1][keep_idx] for p1 in df_singleperts_expression.index])

    df_singleperts_condition = pd.Index(single_perts_condition)
    df_single_pert_val = pd.Index(single_pert_val)

    adata_single = anndata.AnnData(X=df_singleperts_expression.values, var=adata.var.copy(), dtype=df_singleperts_expression.values.dtype)
    adata_single.obs_names = df_singleperts_condition
    adata_single.obs["condition"] = df_singleperts_condition
    adata_single.obs["perts_name"] = df_single_pert_val
    adata_single.obsm["perturbation_neighbors"] = df_singleperts_emb

    for split_seed in range(1,4):
        adata_single.obs[f"split{split_seed}"] = None
        adata_single.obs[f"subgroup{split_seed}"] = "Train/Val"
        for cat in ["train","test","ood"]:
            cat_idx = adata_single.obs["condition"].isin(adata[adata.obs[f"split{split_seed}"] == cat].obs["condition"].cat.categories)
            adata_single.obs.loc[cat_idx ,f"split{split_seed}"] = cat
            if cat == "ood":
                adata_single.obs.loc[cat_idx ,f"subgroup{split_seed}"] = "unseen_single"


    dirName = '/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/{}/hvg5000/bioLord'.format(DataSet)
    if not os.path.isdir(dirName): os.makedirs(dirName)
    os.chdir(dirName)

    adata_single.write("single_biolord.h5ad")
    adata.write("biolord.h5ad")





def prepareData_comb(DataSet):
    dirName = '/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/{}/hvg5000/GEARS'.format(DataSet)
    os.chdir(dirName)
    pert_data = PertData('./data') # specific saved folder   下载 gene2go_all.pkl
    pert_data.load(data_path = './data/train') # load the processed data, the path is saved folder + dataset_name
    adata = pert_data.adata.copy()

    for seed in [1, 2, 3]:
        with open("data/train/splits/train_simulation_{}_0.8.pkl".format(seed), "rb") as f:
            split_data = pickle.load(f)
        
        pert2set = {}
        for i,j in split_data.items():
            for x in j:
                pert2set[x] = i
        
        with open("data/train/splits/train_simulation_{}_0.8_subgroup.pkl".format(seed), "rb") as f:
            subgroup = pickle.load(f)
        adata.obs[f"split{seed}"] = [pert2set[i] for i in adata.obs["condition"].values]
        pert2subgroup = {}
        for i,j in subgroup["test_subgroup"].items():
            for x in j:
                pert2subgroup[x] = i
        
        adata.obs[f"subgroup{seed}"] = adata.obs["condition"].apply(lambda x: pert2subgroup[x] if x in pert2subgroup else 'Train/Val')
        rename = {
            'train': 'train',
            'test': 'ood',
            'val': 'test'
        }
        adata.obs[f'split{seed}'] = adata.obs[f'split{seed}'].apply(lambda x: rename[x])

    adata.obs["perturbation"] = [cond.split("+")[0] for cond in adata.obs["condition"]]
    adata.obs["perturbation_rep"] = [cond.split("+")[1] if len(cond.split("+")) > 1 else "ctrl" for cond in adata.obs["condition"]]
    adata.obs["perturbation"] = adata.obs["perturbation"].astype("category")
    adata.obs["perturbation_rep"] = adata.obs["perturbation_rep"].astype("category")
    new_cats = adata.obs["perturbation_rep"].cat.categories[~adata.obs["perturbation_rep"].cat.categories.isin(adata.obs["perturbation"].cat.categories)]
    adata.obs["perturbation"] = adata.obs["perturbation"].cat.add_categories(new_cats)
    new_cats_rep =  adata.obs["perturbation"].cat.categories[~adata.obs["perturbation"].cat.categories.isin(adata.obs["perturbation_rep"].cat.categories)]
    adata.obs["perturbation_rep"] = adata.obs["perturbation_rep"].cat.add_categories(new_cats_rep)
    adata.obs["perturbation"] = adata.obs["perturbation"].cat.reorder_categories(adata.obs["perturbation_rep"].cat.categories)

    go_path = 'data/go_essential_all/go_essential_all.csv'
    gene_path = 'data/essential_all_data_pert_genes.pkl'
    df = pd.read_csv(go_path)
    df = df.groupby('target').apply(lambda x: x.nlargest(20 + 1, ['importance'])).reset_index(drop = True)
    with open(gene_path, 'rb') as f:
        gene_list = pickle.load(f)
    df = df[df["source"].isin(gene_list)]

    def get_map(pert):
        tmp = pd.DataFrame(np.zeros(len(gene_list)), index=gene_list)
        tmp.loc[df[df.target == pert].source.values, :] = df[df.target == pert].importance.values[:, np.newaxis]
        return tmp.values.flatten()    
    pert2neighbor =  {i: get_map(i) for i in list(adata.obs["perturbation"].cat.categories)}    
    adata.uns["pert2neighbor"] = pert2neighbor

    pert2neighbor = np.asarray([val for val in adata.uns["pert2neighbor"].values()])
    keep_idx = pert2neighbor.sum(0) > 0
    keep_idx1 = pert2neighbor.sum(0) > 1
    keep_idx2 = pert2neighbor.sum(0) > 2
    keep_idx3 = pert2neighbor.sum(0) > 3

    name_map = dict(adata.obs[["condition", "condition_name"]].drop_duplicates().values)
    ctrl = np.asarray(adata[adata.obs["condition"].isin(["ctrl"])].X.mean(0)).flatten()

    df_perts_expression = pd.DataFrame(adata.X.A, index=adata.obs_names, columns=adata.var_names)
    df_perts_expression["condition"] = adata.obs["condition"]
    df_perts_expression = df_perts_expression.groupby(["condition"]).mean()
    df_perts_expression = df_perts_expression.reset_index()

    single_perts_condition = []
    single_pert_val = []
    double_perts = []
    for pert in adata.obs["condition"].cat.categories:
        if len(pert.split("+")) == 1:
            continue
        elif "ctrl" in pert:
            single_perts_condition.append(pert)
            p1, p2 = pert.split("+")
            if p2 == "ctrl":
                single_pert_val.append(p1)
            else:
                single_pert_val.append(p2)
        else:
            double_perts.append(pert)
    single_perts_condition.append("ctrl")
    single_pert_val.append("ctrl")

    df_singleperts_expression = pd.DataFrame(df_perts_expression.set_index("condition").loc[single_perts_condition].values, index=single_pert_val)
    df_singleperts_emb = np.asarray([adata.uns["pert2neighbor"][p1][keep_idx] for p1 in df_singleperts_expression.index])
    df_singleperts_emb1 = np.asarray([adata.uns["pert2neighbor"][p1][keep_idx1] for p1 in df_singleperts_expression.index])
    df_singleperts_emb2 = np.asarray([adata.uns["pert2neighbor"][p1][keep_idx2] for p1 in df_singleperts_expression.index])
    df_singleperts_emb3 = np.asarray([adata.uns["pert2neighbor"][p1][keep_idx3] for p1 in df_singleperts_expression.index])

    df_singleperts_condition = pd.Index(single_perts_condition)
    df_single_pert_val = pd.Index(single_pert_val)

    df_doubleperts_expression = df_perts_expression.set_index("condition").loc[double_perts].values
    df_doubleperts_condition = pd.Index(double_perts)

    adata_single = anndata.AnnData(X=df_singleperts_expression.values, var=adata.var.copy(), dtype=df_singleperts_expression.values.dtype)
    adata_single.obs_names = df_singleperts_condition
    adata_single.obs["condition"] = df_singleperts_condition
    adata_single.obs["perts_name"] = df_single_pert_val
    adata_single.obsm["perturbation_neighbors"] = df_singleperts_emb
    adata_single.obsm["perturbation_neighbors1"] = df_singleperts_emb1
    adata_single.obsm["perturbation_neighbors2"] = df_singleperts_emb2
    adata_single.obsm["perturbation_neighbors3"] = df_singleperts_emb3

    for split_seed in range(1,4):
        adata_single.obs[f"split{split_seed}"] = None
        adata_single.obs[f"subgroup{split_seed}"] = "Train/Val"
        for cat in ["train","test","ood"]:
            cat_idx = adata_single.obs["condition"].isin(adata[adata.obs[f"split{split_seed}"] == cat].obs["condition"].cat.categories)
            adata_single.obs.loc[cat_idx ,f"split{split_seed}"] = cat
            if cat == "ood":
                for ood_set in ["combo_seen0", "combo_seen1", "combo_seen2", "unseen_single"]:
                    idx_ood = adata_single.obs["condition"].isin(adata[adata.obs[f"subgroup{split_seed}"] == ood_set].obs["condition"].cat.categories)
                    adata_single.obs.loc[idx_ood ,f"subgroup{split_seed}"] = ood_set

    dirName = '/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/{}/hvg5000/bioLord'.format(DataSet)
    if not os.path.isdir(dirName): os.makedirs(dirName)
    os.chdir(dirName)

    adata_single.write("single_biolord.h5ad")
    adata.write("biolord.h5ad")

def prepareData_chemical(DataSet):
    import chemprop
    import scanpy as sc
    dirName = '/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/{}/hvg5000/bioLord'.format(DataSet)
    if not os.path.isdir(dirName): os.makedirs(dirName)
    os.chdir(dirName)
    adata = sc.read_h5ad('../../filter_hvg5000.h5ad')

    features = {}
    for mol in adata.obs["SMILES"].cat.categories:
        features[
            mol
        ] = chemprop.features.features_generators.rdkit_2d_normalized_features_generator(
            mol
        )
    features_arr = np.asarray([features[mol] for mol in features])
        
    features_df = pd.DataFrame.from_dict(features).T
    features_df = features_df.fillna(0)

    threshold = 0.001
    cols_keep = list(np.where(features_df.std() > threshold)[0])
    features_df = features_df.iloc[:, np.where(features_df.std() > threshold)[0]]
    normalized_df = (features_df - features_df.mean()) / features_df.std()
    features_cells = np.zeros((adata.shape[0], normalized_df.shape[1] + 1))
    for mol, rdkit_2d in normalized_df.iterrows():
        features_cells[adata.obs["SMILES"].isin([mol]), :-1] = rdkit_2d.values

    dose = adata.obs["dose"] / np.max(adata.obs["dose"])
    features_cells[:, -1] = dose
    adata.obsm["rdkit2d"] = features_cells[:, :-1]
    adata.obsm["rdkit2d_dose"] = features_cells
    adata.write_h5ad('filter_hvg5000_biolord.h5ad')


'''
https://github.com/nitzanlab/biolord_reproducibility/blob/main/notebooks/perturbations/adamson/1_perturbations_adamson_preprocessing.ipynb

conda activate gears
'''

DataSet = 'Norman'
isComb = True
