import sys
import pandas as pd
import pickle
import numpy as np
import anndata
import anndata as ad
import biolord
import torch
from myUtil1 import *

sys.path.append('/NFS2_home/NFS2_home_1/wzt/software/biolord/biolord_reproducibility/utils')
from utils_perturbations import (
    compute_metrics,
    bool2idx,
    repeat_n
)



def getSingle_parameter():
    varying_arg = {
        "seed": 42,
        "use_batch_norm": False,
        "use_layer_norm": False,
        "step_size_lr": 45,
        "attribute_dropout_rate": 0.1,
        "cosine_scheduler": True,
        "scheduler_final_lr": 1e-5,
        "n_latent": 32,
        "n_latent_attribute_ordered": 512,
        "reconstruction_penalty": 1000.0,
        "attribute_nn_width": 64,
        "attribute_nn_depth":6,
        "attribute_nn_lr": 0.001,
        "attribute_nn_wd": 4e-8,
        "latent_lr": 0.0001,
        "latent_wd": 0.001,
        "unknown_attribute_penalty": 10000.0,
        "decoder_width": 64,
        "decoder_depth": 1,
        "decoder_activation": False,
        "attribute_nn_activation": False,
        "unknown_attributes": False,
        "decoder_lr": 0.001,
        "decoder_wd": 0.01,
        "max_epochs": 500,
        "early_stopping_patience": 10,
        "ordered_attributes_key": "perturbation_neighbors",
        "n_latent_attribute_categorical": 16,
        "unknown_attribute_noise_param": .2,
        "autoencoder_lr": 0.0001,
        "autoencoder_wd": 0.001
    }


    module_params = {
        "attribute_nn_width":  varying_arg["attribute_nn_width"],
        "attribute_nn_depth": varying_arg["attribute_nn_depth"],
        "use_batch_norm": varying_arg["use_batch_norm"],
        "use_layer_norm": varying_arg["use_layer_norm"],
        "attribute_dropout_rate":  varying_arg["attribute_dropout_rate"],
        "unknown_attribute_noise_param": varying_arg["unknown_attribute_noise_param"],
        "seed": varying_arg["seed"],
        "n_latent_attribute_ordered": varying_arg["n_latent_attribute_ordered"],
        "n_latent_attribute_categorical": varying_arg["n_latent_attribute_categorical"],
        "reconstruction_penalty": varying_arg["reconstruction_penalty"],
        "unknown_attribute_penalty": varying_arg["unknown_attribute_penalty"],
        "decoder_width": varying_arg["decoder_width"],
        "decoder_depth": varying_arg["decoder_depth"],
        "decoder_activation": varying_arg["decoder_activation"],
        "attribute_nn_activation": varying_arg["attribute_nn_activation"],
        "unknown_attributes": varying_arg["unknown_attributes"],
    }


    trainer_params = {
        "n_epochs_warmup": 0,
        "latent_lr": varying_arg["autoencoder_lr"],
        "latent_wd": varying_arg["autoencoder_wd"],
        "attribute_nn_lr": varying_arg["attribute_nn_lr"],
        "attribute_nn_wd": varying_arg["attribute_nn_wd"],
        "step_size_lr": varying_arg["step_size_lr"],
        "cosine_scheduler": varying_arg["cosine_scheduler"],
        "scheduler_final_lr": varying_arg["scheduler_final_lr"],
        "decoder_lr": varying_arg["decoder_lr"],
        "decoder_wd": varying_arg["decoder_wd"]
    }
    return varying_arg, module_params, trainer_params


def getCombination_parameter():
    varying_arg = {
        "seed": 42,
        "unknown_attribute_noise_param": 0.2,
        "use_batch_norm": False,
        "use_layer_norm": False,
        "step_size_lr": 45,
        "attribute_dropout_rate": 0.0,
        "cosine_scheduler":True,
        "scheduler_final_lr":1e-5,
        "n_latent":32,
        "n_latent_attribute_ordered": 32,
        "reconstruction_penalty": 10000.0,
        "attribute_nn_width": 64,
        "attribute_nn_depth" :2,
        "attribute_nn_lr": 0.001,
        "attribute_nn_wd": 4e-8,
        "latent_lr": 0.01,
        "latent_wd": 0.00001,
        "decoder_width": 32,
        "decoder_depth": 2,
        "decoder_activation": True,
        "attribute_nn_activation": True,
        "unknown_attributes": False,
        "decoder_lr": 0.01,
        "decoder_wd": 0.01,
        "max_epochs":500,
        "early_stopping_patience": 200,
        "ordered_attributes_key": "perturbation_neighbors1",
        "n_latent_attribute_categorical": 16,
        "unknown_attribute_penalty": 10000.0

    }


    module_params = {
        "attribute_nn_width":  varying_arg["attribute_nn_width"],
        "attribute_nn_depth": varying_arg["attribute_nn_depth"],
        "use_batch_norm": varying_arg["use_batch_norm"],
        "use_layer_norm": varying_arg["use_layer_norm"],
        "attribute_dropout_rate":  varying_arg["attribute_dropout_rate"],
        "unknown_attribute_noise_param": varying_arg["unknown_attribute_noise_param"],
        "seed": varying_arg["seed"],
        "n_latent_attribute_ordered": varying_arg["n_latent_attribute_ordered"],
        "n_latent_attribute_categorical": varying_arg["n_latent_attribute_categorical"],
        "reconstruction_penalty": varying_arg["reconstruction_penalty"],
        "unknown_attribute_penalty": varying_arg["unknown_attribute_penalty"],
        "decoder_width": varying_arg["decoder_width"],
        "decoder_depth": varying_arg["decoder_depth"],
        "decoder_activation": varying_arg["decoder_activation"],
        "attribute_nn_activation": varying_arg["attribute_nn_activation"],
        "unknown_attributes": varying_arg["unknown_attributes"],
    }


    trainer_params = {
        "n_epochs_warmup": 0,
        "latent_lr": varying_arg["latent_lr"],
        "latent_wd": varying_arg["latent_wd"],
        "attribute_nn_lr": varying_arg["attribute_nn_lr"],
        "attribute_nn_wd": varying_arg["attribute_nn_wd"],
        "step_size_lr": varying_arg["step_size_lr"],
        "cosine_scheduler": varying_arg["cosine_scheduler"],
        "scheduler_final_lr": varying_arg["scheduler_final_lr"],
        "decoder_lr": varying_arg["decoder_lr"],
        "decoder_wd": varying_arg["decoder_wd"]
    }
    return varying_arg, module_params, trainer_params





def clean_condition(condition):
    return condition.replace('+ctrl', '').replace('ctrl+', '').strip()

def runBioLord_single(DataSet):
    dirName = '/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/{}/hvg5000/bioLord'.format(DataSet)
    if not os.path.isdir(dirName): os.makedirs(dirName)
    os.chdir(dirName)
    adata = sc.read("biolord.h5ad")
    adata_single = sc.read("single_biolord.h5ad")
    np.random.seed(42)

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

    varying_arg, module_params, trainer_params = getSingle_parameter()
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
    df_singleperts_condition = pd.Index(single_perts_condition)
    df_single_pert_val = pd.Index(single_pert_val)

    test_metrics_biolord_delta = {}
    test_metrics_biolord_delta_normalized = {}
    ordered_attributes_key = varying_arg["ordered_attributes_key"]

    biolord.Biolord.setup_anndata(
        adata_single,
        ordered_attributes_keys=[ordered_attributes_key],
        categorical_attributes_keys=None,
        retrieval_attribute_key=None,
    )

    for split_seed in range(1,4):
        test_metrics_biolord_delta[split_seed] = {}
        test_metrics_biolord_delta_normalized[split_seed] = {}

        train_idx = df_singleperts_condition.isin(adata[adata.obs[f"split{split_seed}"] == "train"].obs["condition"].cat.categories)
        train_condition_perts = df_singleperts_condition[train_idx]
        train_perts = df_single_pert_val[train_idx]

        model = biolord.Biolord(
            adata=adata_single,
            n_latent=varying_arg["n_latent"],
            model_name=DataSet,
            module_params=module_params,
            train_classifiers=False,
            split_key=f"split{split_seed}"
        )

        model.train(
            max_epochs=int(varying_arg["max_epochs"]),
            batch_size=32,
            plan_kwargs=trainer_params,
            early_stopping=True,
            early_stopping_patience=int(varying_arg["early_stopping_patience"]),
            check_val_every_n_epoch=5,
            num_workers=1,
            enable_checkpointing=False
        )
        adata_control = adata_single[adata_single.obs["condition"] == "ctrl"].copy()
        dataset_control = model.get_dataset(adata_control)
        dataset_reference = model.get_dataset(adata_single)

        n_obs = adata_control.shape[0]

        predictions_dict_delta = {}
        perts = adata[adata.obs[f"subgroup{split_seed}"] == "unseen_single"].obs["condition"].cat.categories

        for i, pert in enumerate(perts):
            if pert in train_condition_perts:
                idx_ref =  bool2idx(adata_single.obs["condition"] == pert)[0]
                expression_pert = dataset_reference["X"][[idx_ref], :].mean(0).cpu().numpy()
                test_preds_delta = expression_pert
            elif "ctrl" in pert:
                idx_ref =  bool2idx(adata_single.obs["condition"] == pert)[0]
                expression_pert = dataset_reference["X"][[idx_ref], :].mean(0).cpu().numpy()

                dataset_pred = dataset_control.copy()
                dataset_pred[ordered_attributes_key] = repeat_n(dataset_reference[ordered_attributes_key][idx_ref, :], n_obs)
                test_preds, _ = model.module.get_expression(dataset_pred)

                test_preds_delta = test_preds.cpu().numpy()

            predictions_dict_delta[clean_condition(pert)] = test_preds_delta.flatten()
        pred = pd.DataFrame(predictions_dict_delta).T
        pred.columns = adata_control.var_names
        
        dirOut = 'savedModels{}'.format(split_seed)
        if not os.path.isdir(dirOut): os.makedirs(dirOut)
        pred.to_csv('{}/pred.tsv'.format(dirOut),  sep='\t')


def runBioLord_combination(DataSet):
    dirName = '/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/{}/hvg5000/bioLord'.format(DataSet)
    if not os.path.isdir(dirName): os.makedirs(dirName)
    os.chdir(dirName)
    adata = sc.read("biolord.h5ad")
    adata_single = sc.read("single_biolord.h5ad")
    np.random.seed(42)
    varying_arg, module_params, trainer_params = getCombination_parameter()

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
        else:
            double_perts.append(pert)
    single_perts_condition.append("ctrl")
    single_pert_val.append("ctrl")

    df_singleperts_expression = pd.DataFrame(df_perts_expression.set_index("condition").loc[single_perts_condition].values, index=single_pert_val)
    df_singleperts_emb = np.asarray([adata.uns["pert2neighbor"][p1][keep_idx] for p1 in df_singleperts_expression.index])
    df_singleperts_pca = sc.pp.pca(df_singleperts_emb)
    df_singleperts_condition = pd.Index(single_perts_condition)
    df_single_pert_val = pd.Index(single_pert_val)

    df_doubleperts_expression = df_perts_expression.set_index("condition").loc[double_perts].values
    df_doubleperts_condition = pd.Index(double_perts)


    test_metrics_biolord_delta = {}
    test_metrics_biolord_delta_normalized = {}
    
    ordered_attributes_key = varying_arg["ordered_attributes_key"]

    biolord.Biolord.setup_anndata(
        adata_single,
        ordered_attributes_keys=[ordered_attributes_key],
        categorical_attributes_keys=None,
        retrieval_attribute_key=None,
    )

    for split_seed in range(1,4):
        test_metrics_biolord_delta[split_seed] = {}
        test_metrics_biolord_delta_normalized[split_seed] = {}

        train_idx = df_singleperts_condition.isin(adata[adata.obs[f"split{split_seed}"] == "train"].obs["condition"].cat.categories)
        train_condition_perts = df_singleperts_condition[train_idx]
        train_condition_perts_double = df_doubleperts_condition[df_doubleperts_condition.isin(adata[adata.obs[f"split{split_seed}"] == "train"].obs["condition"].cat.categories)]
        train_perts = df_single_pert_val[train_idx]

        model = biolord.Biolord(
            adata=adata_single,
            n_latent=varying_arg["n_latent"],
            model_name=DataSet,
            module_params=module_params,
            train_classifiers=False,
            split_key=f"split{split_seed}"
        )

        model.train(
            max_epochs=int(varying_arg["max_epochs"]),
            batch_size=32,
            plan_kwargs=trainer_params,
            early_stopping=True,
            early_stopping_patience=int(varying_arg["early_stopping_patience"]),
            check_val_every_n_epoch=5,
            num_workers=1,
            enable_checkpointing=False
        )
        adata_control = adata_single[adata_single.obs["condition"] == "ctrl"].copy()
        dataset_control = model.get_dataset(adata_control)

        dataset_reference = model.get_dataset(adata_single)

        n_obs = adata_control.shape[0]
        predictions_dict_delta = {}
        predictions_dict_mean = {}
        for ood_set in ["combo_seen0", "combo_seen1", "combo_seen2", "unseen_single"]:
            perts = adata[adata.obs[f"subgroup{split_seed}"] == ood_set].obs["condition"].cat.categories
            for i, pert in enumerate(perts):
                if pert in train_condition_perts:
                    idx_ref =  bool2idx(adata_single.obs["condition"] == pert)[0]
                    expression_pert = dataset_reference["X"][[idx_ref], :].mean(0).cpu().numpy()
                    test_preds_delta = expression_pert
                elif pert in train_condition_perts_double:
                    expression_pert = df_doubleperts_expression[df_doubleperts_condition.isin([pert]), :]
                    test_preds_delta = df_doubleperts_expression[df_doubleperts_condition.isin([pert]), :]
                elif "ctrl" in pert:
                    idx_ref =  bool2idx(adata_single.obs["condition"] == pert)[0]
                    expression_pert = dataset_reference["X"][[idx_ref], :].mean(0).cpu().numpy()

                    dataset_pred = dataset_control.copy()
                    dataset_pred[ordered_attributes_key] = repeat_n(dataset_reference[ordered_attributes_key][idx_ref, :], n_obs)
                    test_preds, _ = model.module.get_expression(dataset_pred)

                    test_preds_delta = test_preds.cpu().numpy()

                else:
                    expression_pert = df_doubleperts_expression[df_doubleperts_condition.isin([pert]), :]
                    test_preds_add = []
                    for p in pert.split("+"):
                        if p in train_perts:
                            test_predsp = df_singleperts_expression.values[df_single_pert_val.isin([p]), :]
                            test_preds_add.append(test_predsp[0, :])
                        else:
                            idx_ref =  bool2idx(adata_single.obs["perts_name"].isin([p]))[0]

                            dataset_pred = dataset_control.copy()
                            dataset_pred[ordered_attributes_key] = repeat_n(dataset_reference[ordered_attributes_key][idx_ref, :], n_obs)
                            test_preds, _ = model.module.get_expression(dataset_pred)
                            test_preds_add.append(test_preds.cpu().numpy())

                    test_preds_delta = test_preds_add[0] + test_preds_add[1] - ctrl

                predictions_dict_delta[clean_condition(pert)] = test_preds_delta.flatten()

        pred = pd.DataFrame(predictions_dict_delta).T
        pred.columns = adata_control.var_names
        
        dirOut = 'savedModels{}'.format(split_seed)
        if not os.path.isdir(dirOut): os.makedirs(dirOut)
        pred.to_csv('{}/pred.tsv'.format(dirOut),  sep='\t')

def generateExp(cellNum, means, std):
    expression_matrix = np.array([
    np.random.normal(loc=means[i], scale=std[i], size=cellNum) 
    for i in range(len(means))]).T
    return expression_matrix


### 根据预测的生成表达量
def generateH5ad(DataSet, seed = 1):
    dirName = '/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/{}/hvg5000/bioLord'.format(DataSet)
    os.chdir(dirName)
    filein = 'savedModels{}/pred.tsv'.format(seed)
    exp = pd.read_csv(filein, sep='\t', index_col=0)
    filein = '../GEARS/savedModels{}/result.h5ad'.format(seed)
    adata = sc.read_h5ad(filein)
    expGene = np.intersect1d(adata.var_names, exp.columns)
    pertGenes = np.intersect1d(adata.obs['perturbation'].unique(), exp.index)
    adata = adata[:, expGene]; exp = exp.loc[:, expGene]

    control_exp = adata[adata.obs['perturbation'] == 'control'].to_df()
    control_std = list(np.std(control_exp))
    control_std = [i if not np.isnan(i) else 0 for i in control_std]
    for pertGene in pertGenes:
        cellNum = adata[(adata.obs['perturbation'] == pertGene) & (adata.obs['Expcategory']=='imputed')].shape[0]
        means = list(exp.loc[pertGene, ])
        expression_matrix = generateExp(cellNum, means, control_std)
        adata[(adata.obs['perturbation'] == pertGene) & (adata.obs['Expcategory']=='imputed')].X = expression_matrix
    adata.write('savedModels{}/result.h5ad'.format(seed))



def runBioLord_chemical(DataSet, seed):
    dirName = '/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/{}/hvg5000/bioLord'.format(DataSet)
    os.chdir(dirName)
    sys.path.append('/home/wzt/software/biolord/biolord_reproducibility/utils')
    #from utils_perturbations_sciplex3 import compute_prediction, compute_baseline, create_df
    varying_arg = {
    "seed": np.random.choice([0, 1, 2, 3, 4]), "batch_size": np.random.choice([256, 512]),
    "unknown_attribute_noise_param": np.random.choice([0.1, 0.5, 1, 2, 5, 10, 20]),
    "decoder_width": np.random.choice([128, 256, 512, 1024, 2048, 4096]),
    "decoder_depth": np.random.choice([1, 2, 3, 4, 6, 8]),
    "use_batch_norm": np.random.choice([True, False]),
    "use_layer_norm": np.random.choice([True, False]),
    "step_size_lr": np.random.choice([45, 90, 180]),
    "attribute_dropout_rate": np.random.choice([0.05, 0.1, 0.25, 0.5, 0.75]),
    "cosine_scheduler": np.random.choice([True, False]),
    "scheduler_final_lr": np.random.choice([1e-3, 1e-4, 1e-5, 1e-6]),
    "n_latent": np.random.choice([16, 32, 64, 128, 256]),
    "n_latent_attribute_ordered": np.random.choice([128, 256, 512]),
    "n_latent_attribute_categorical": np.random.choice([2, 3, 4, 6, 8]),
    "reconstruction_penalty": np.random.choice([1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]),
    "attribute_nn_width": np.random.choice([32, 64, 128, 256, 512]),
    "attribute_nn_depth": np.random.choice([1, 2, 3, 4, 6, 8]),
    "attribute_nn_lr": np.random.choice([1e-2, 1e-3, 1e-4]),
    "attribute_nn_wd": np.random.choice([1e-8, 4e-8, 1e-7]),
    "latent_lr": np.random.choice([1e-2, 1e-3, 1e-4]),
    "latent_wd": np.random.choice([1e-2, 1e-3, 1e-4]),
    "decoder_lr": np.random.choice([1e-2, 1e-3, 1e-4]),
    "decoder_wd": np.random.choice([1e-2, 1e-3, 1e-4]),
    "unknown_attribute_penalty": np.random.choice([1e-1, 1e0, 2e0, 5e0, 1e1, 2e1, 5e1, 1e2, 2e2])}
    
    np.random.seed(42)
    varying_arg["adata"] = sc.read("filter_hvg5000_biolord.h5ad")
    varying_arg["ordered_attributes_keys"] = ["rdkit2d_dose"]
    varying_arg["categorical_attributes_keys"] = ["cell_type"]
    varying_arg["retrieval_attribute_key"] = None
    varying_arg["split_key"] = "split_ood_multi_task{}".format(seed)
    varying_arg["layer"] = None
    varying_arg["gene_likelihood"] = "normal"
    varying_arg["dataset_name"] = "sciplex3"

    module_params = {
        "decoder_width": varying_arg["decoder_width"],
        "decoder_depth": varying_arg["decoder_depth"],
        "attribute_nn_width":  varying_arg["attribute_nn_width"],
        "attribute_nn_depth": varying_arg["attribute_nn_depth"],
        "use_batch_norm": varying_arg["use_batch_norm"],
        "use_layer_norm": varying_arg["use_layer_norm"],
        "attribute_dropout_rate":  varying_arg["attribute_dropout_rate"],
        "unknown_attribute_noise_param": varying_arg["unknown_attribute_noise_param"],
        "seed": varying_arg["seed"],
        "n_latent_attribute_ordered": varying_arg["n_latent_attribute_ordered"],
        "n_latent_attribute_categorical": varying_arg["n_latent_attribute_categorical"],
        "reconstruction_penalty": varying_arg["reconstruction_penalty"],
        "unknown_attribute_penalty":varying_arg["unknown_attribute_penalty"],
    }
    
    trainer_params = {
        "n_epochs_warmup": 0,
        "latent_lr": varying_arg["latent_lr"],
        "latent_wd": varying_arg["latent_wd"],
        "decoder_lr": varying_arg["decoder_lr"],
        "decoder_wd": varying_arg["decoder_wd"],
        "attribute_nn_lr": varying_arg["attribute_nn_lr"],
        "attribute_nn_wd": varying_arg["attribute_nn_wd"],
        "step_size_lr": varying_arg["step_size_lr"],
        "cosine_scheduler": varying_arg["cosine_scheduler"],
        "scheduler_final_lr": varying_arg["scheduler_final_lr"]
    }

    adata = varying_arg["adata"]
    biolord.Biolord.setup_anndata(
        adata,
        ordered_attributes_keys=varying_arg["ordered_attributes_keys"],
        categorical_attributes_keys=varying_arg["categorical_attributes_keys"],
        retrieval_attribute_key=varying_arg["retrieval_attribute_key"],
    )

    model = biolord.Biolord(
        adata=adata,
        n_latent=varying_arg["n_latent"],
        model_name="sciplex3",
        module_params=module_params,
        train_classifiers=False,
        split_key=varying_arg["split_key"],
    )
    model.train(
        max_epochs=200,
        batch_size= 512,
        plan_kwargs=trainer_params,
        early_stopping=True,
        early_stopping_patience=20,
        check_val_every_n_epoch=20,
        num_workers=1,
    )
    model.module.eval()
    idx_test_control = np.where((adata.obs[varying_arg['split_key']] == "test") & (adata.obs["control"] == 1))[0]
    adata_test_control = adata[idx_test_control].copy()
    idx_ood = np.where((adata.obs[varying_arg['split_key']] == "ood"))[0]
    adata_ood = adata[idx_ood].copy()
    dataset_control = model.get_dataset(adata_test_control)
    dataset_ood = model.get_dataset(adata_ood)
    adata1 = compute_prediction(model, adata_ood, dataset_ood, cell_lines = ["A549", "K562", "MCF7"], dataset_control = dataset_control, use_DEGs=False, verbose=True)
    
    adata_control = adata[adata.obs['perturbation'] == 'control'].copy()
    adata_control.obs['Expcategory'] = 'control'

    adata_treat = adata[adata.obs['split_ood_multi_task{}'.format(seed)] == 'ood'].copy()
    adata_treat.obs['Expcategory'] = 'stimulated'

    adata1.obs['Expcategory'] = 'imputed'
    adata1.var = adata.var
    pred = anndata.concat([adata_control, adata_treat, adata1])
    pred.var = adata.var
    tmp = 'savedModels{}'.format(seed)
    if not os.path.isdir(tmp): os.makedirs(tmp)
    pred.write_h5ad('{}/result.h5ad'.format(tmp))

def compute_prediction(model,  adata_ood,  dataset_ood, cell_lines=None, dataset_control=None, use_DEGs=True, verbose=True):
    pert_categories_index = pd.Index(adata_ood.obs["cov_drug_dose_name"].values, dtype="category")
    allowed_cell_lines = []

    cl_dict = {
        torch.Tensor([0.]): "A549",
        torch.Tensor([1.]): "K562",
        torch.Tensor([2.]): "MCF7",
    }

    if cell_lines is None:
        cell_lines = ["A549", "K562", "MCF7"]

    print(cell_lines)
    result_list = []
    for cell_drug_dose_name, category_count in tqdm(zip(*np.unique(pert_categories_index.values, return_counts=True))):
        if category_count <= 5:
            continue
        # doesn"t make sense to evaluate DMSO (=control) as a perturbation
        if ("dmso" in cell_drug_dose_name.lower() or "control" in cell_drug_dose_name.lower()):
            continue
        layer = 'X'
        bool_category = pert_categories_index.get_loc(cell_drug_dose_name)
        idx_all = bool2idx(bool_category)
        idx = idx_all[0]
        y_true = dataset_ood[layer][idx_all, :].to(model.device)
        
                    
        dataset_comb = {}
        if dataset_control is None:
            n_obs = y_true.size(0).to(model.device)
            for key, val in dataset_ood.items():
                dataset_comb[key] = val[idx_all].to(model.device)
        else:
            n_obs = dataset_control[layer].size(0)
            dataset_comb[layer] = dataset_control[layer].to(model.device)
            dataset_comb["ind_x"] = dataset_control["ind_x"].to(model.device)
            for key in dataset_control:
                if key not in [layer, "ind_x"]:
                    dataset_comb[key] = repeat_n(dataset_ood[key][idx, :], n_obs)

        stop = False
        for tensor, cl in cl_dict.items():
            if (tensor == dataset_ood["cell_type"][idx]).all():
                if cl not in cell_lines:
                    stop = True
        if stop:
            continue
        pred, _ = model.module.get_expression(dataset_comb)
        result = anndata.AnnData(np.array(pred.cpu()))
        result.obs['cov_drug_dose_name'] = cell_drug_dose_name
        result_list.append(result)
    pred_result = ad.concat(result_list)
    return pred_result






'''
https://github.com/nitzanlab/biolord_reproducibility/blob/main/notebooks/perturbations/adamson/1_perturbations_adamson_preprocessing.ipynb
conda activate cpa
'''

DataSet = 'Adamson'
isComb = False
seeds = [1, 2, 3]
torch.cuda.set_device('cuda:1')
