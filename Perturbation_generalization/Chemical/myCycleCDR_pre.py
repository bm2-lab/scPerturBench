import sys, subprocess, os
sys.path.append('/home/project/Pertb_benchmark')
from myUtil1 import *
from scipy import sparse
import anndata as ad
import anndata
import pickle
from collections import defaultdict, OrderedDict

import yaml
import torch
import random
import itertools

import torch.optim as optim
from torch.optim import lr_scheduler
from torch_geometric.loader import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler


sys.path.insert(0, '/home/software/cycleCDR')

from cycleCDR.model import cycleCDR  ### type: ignore
from cycleCDR.utils import Trainer    ### type: ignore
from cycleCDR.dataset import load_dataset_splits    ### type: ignore
from cycleCDR.dataset import load_dataset_splits_for_gat    ### type: ignore

from preprocessing_manuscript.sciplex3.helper import rank_genes_groups_by_cov  ### type: ignore

def split_adata(adata: AnnData, split_size=1000):
    adata_res = []
    one_index = adata.obs.shape[0] // split_size
    if one_index * split_size < adata.obs.shape[0]:
        one_index += 1
    for i in range(one_index):
        if i != one_index - 1:
            adata_res.append(
                adata[adata.obs.iloc[i * split_size : (i + 1) * split_size].index]
            )
        else:
            adata_res.append(adata[adata.obs.iloc[i * split_size :].index])

    return adata_res

def step1(X):
    DataSet, seed = X
    dirName = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/cycleCDR/savedModels{}'.format(DataSet, seed)
    if not os.path.isdir(dirName): os.makedirs(dirName)
    adata = sc.read_h5ad('/home/project/Pertb_benchmark/DataSet2/{}/filter_hvg5000_logNor.h5ad'.format(DataSet))
    adata.uns['log1p'] = {}; adata.uns['log1p']["base"] = None
    sc.pp.highly_variable_genes(adata, n_top_genes=200, subset=False)

    idx = np.where(adata.var.highly_variable)[0]
    pickle.dump(idx.tolist()[:200], open("{}/chemcpa_mse_hvg_idx.pkl".format(dirName), "wb"), )


    adata.X = adata.X.todense()
    rank_genes_groups_by_cov(
    adata,
    groupby="cov_drug_dose_name",
    covariate="cell_type",
    control_group="control_1.0",
    key_added="all_DEGs")

    colunms_name = [i for i in range(50)]
    deg_gene = pd.DataFrame(columns=colunms_name)
    for key, value in adata.uns["all_DEGs"].items():
        idx = np.where(adata.var_names.isin(value))
        idx = idx[0].tolist()

        temp = pd.DataFrame([idx], index=[key], columns=colunms_name)
        deg_gene = pd.concat([deg_gene, temp])
    deg_gene.to_csv("{}/chemcpa_deg_gene.csv".format(dirName))


### chemcpa_sciplex.py
    rng = np.random.default_rng(100)
    set_names = ["valid", "test", "train"]

    adata_cpa = adata.copy()
    split_ood_finetuning = 'split_ood_multi_task{}'.format(seed)

    for set_name in set_names:
        if set_name == "test":
            adata_control = adata_cpa[
                (adata_cpa.obs[split_ood_finetuning].isin(["test"]))
                & (adata_cpa.obs["control"].isin([1]))
            ]
            adata_treat = adata_cpa[
                (adata_cpa.obs[split_ood_finetuning].isin(["ood"]))
                & (adata_cpa.obs.control.isin([0]))
            ]

        elif set_name == "valid":
            adata_control = adata_cpa[
                (adata_cpa.obs[split_ood_finetuning].isin(["test"]))
                & (adata_cpa.obs["control"].isin([1]))
            ]
            adata_treat = adata_cpa[
                (adata_cpa.obs[split_ood_finetuning].isin(["test"]))
                & (adata_cpa.obs.control.isin([0]))
            ]

        elif set_name == "train":
            adata_control = adata_cpa[
                (adata_cpa.obs[split_ood_finetuning].isin(["train"]))
                & (adata_cpa.obs["control"].isin([1]))
            ]
            adata_treat = adata_cpa[
                (adata_cpa.obs[split_ood_finetuning].isin(["train"]))
                & (adata_cpa.obs.control.isin([0]))
            ]

        cell_ids = np.unique(adata_treat.obs.cell_type).tolist()

        for cell_id in cell_ids:
            cell_control = adata_control[adata_control.obs.cell_type == cell_id].copy()
            cell_treat = adata_treat[adata_treat.obs.cell_type == cell_id].copy()

            treat_length = cell_treat.obs.shape[0]

            row_sequence = np.arange(cell_control.X.shape[0])

            temp = rng.choice(
                row_sequence, cell_treat.obs.shape[0] - row_sequence.shape[0], replace=True
            )
            row_sequence = np.concatenate((row_sequence, temp))

            treat_columns = [i for i in range(cell_treat.X.shape[1])]
            treat_columns.extend(["cell_type", "SMILES", "cov_drug_dose_name"])
            control_columns = [i for i in range(cell_treat.X.shape[1])]
            control_columns.extend(["cell_type"])

            split_size = 1000
            cell_control = split_adata(cell_control, split_size)
            cell_treat = split_adata(cell_treat, split_size)

            treat_df = []
            control_df = []

            temp_treat_df = []
            temp_control_df = []

            ## treat_length = min(1500, treat_length)
            for i in tqdm(range(treat_length)):
                index = i - ((i // split_size) * split_size)

                treat_row = cell_treat[i // split_size].X[index, :].tolist()
                smile = cell_treat[i // split_size].obs.SMILES.iloc[index]
                ### cov_drug = cell_treat[i // split_size].obs.cov_drug.iloc[index] ### 脚本原始
                cov_drug = cell_treat[i // split_size].obs.cov_drug_dose_name.iloc[index] ### 改成加上浓度

                treat_row.extend([cell_id, smile, cov_drug])
                treat_temp = pd.DataFrame([treat_row], columns=treat_columns)
                temp_treat_df.append(treat_temp)

                control_index = row_sequence[i]
                control_index_temp = control_index - (
                    (control_index // split_size) * split_size
                )
                control_row = (
                    cell_control[control_index // split_size]
                    .X[control_index_temp, :]
                    .tolist()
                )
                control_row.extend([cell_id])
                control_temp = pd.DataFrame([control_row], columns=control_columns)

                temp_control_df.append(control_temp)

                if len(temp_treat_df) == 1000:
                    temp_treat_df = pd.concat(temp_treat_df, ignore_index=True)
                    temp_control_df = pd.concat(temp_control_df, ignore_index=True)
                    treat_df.append(temp_treat_df)
                    control_df.append(temp_control_df)
                    temp_treat_df = []
                    temp_control_df = []

            if len(temp_treat_df) != 0:
                temp_treat_df = pd.concat(temp_treat_df, ignore_index=True)
                temp_control_df = pd.concat(temp_control_df, ignore_index=True)
                treat_df.append(temp_treat_df)
                control_df.append(temp_control_df)

            treat_df = pd.concat(treat_df, ignore_index=True)
            control_df = pd.concat(control_df, ignore_index=True)

            treat_df.to_csv(
                f"{dirName}/chemcpa_trapnell_treat_{set_name}.csv",
                index=False,
            )
            control_df.to_csv(
                f"{dirName}/chemcpa_trapnell_control_{set_name}.csv",
                index=False,
            )

            print(f"Done {set_name} {cell_id}")
            print(treat_df.shape, control_df.shape)

        print(f"Done {set_name}")


def f_step1():
    mylist = []
    for DataSet in SingleDataSets[1:]:
        for seed in seeds:
            mylist.append([DataSet, seed])
    myPool(step1, mylist, processes = 3)


'''
conda activate cpa
'''

SingleDataSets = ['sciplex3_A549', 'sciplex3_MCF7',  'sciplex3_K562']

seeds  = [1, 2, 3]

if __name__ == '__main__':
    step1('sciplex3_A549', seed=1)
    f_step1()