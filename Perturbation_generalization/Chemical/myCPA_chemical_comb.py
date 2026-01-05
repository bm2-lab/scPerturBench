import sys, subprocess, os
sys.path.append('/home/project/Pertb_benchmark')
from myUtil1 import *
import scanpy as sc, pandas as pd, numpy as np, anndata as ad
import anndata
import pickle, tqdm, joblib, torch, cpa
import torch.nn as nn
sc.settings.verbosity = 3




def getModelParameter():
    ae_hparams = {
    "n_latent": 128,
    "recon_loss": "gauss",
    "doser_type": "linear",
    "n_hidden_encoder": 512,
    "n_layers_encoder": 2,
    "n_hidden_decoder": 256,
    "n_layers_decoder": 2,
    "use_batch_norm_encoder": True,
    "use_layer_norm_encoder": False,
    "use_batch_norm_decoder": False,
    "use_layer_norm_decoder": False,
    "dropout_rate_encoder": 0.0,
    "dropout_rate_decoder": 0.2,
    "variational": False,
    "seed": 1117,
}

    trainer_params = {
    "n_epochs_kl_warmup": None,
    "n_epochs_pretrain_ae": 30,
    "n_epochs_adv_warmup": 100,
    "n_epochs_mixup_warmup": 10,
    "mixup_alpha": 0.2,
    "adv_steps": 2,
    "n_hidden_adv": 128,
    "n_layers_adv": 2,
    "use_batch_norm_adv": True,
    "use_layer_norm_adv": False,
    "dropout_rate_adv": 0.2,
    "reg_adv": 50.0,
    "pen_adv": 1.0,
    "lr": 0.0003,
    "wd": 4e-07,
    "adv_lr": 0.0003,
    "adv_wd": 4e-07,
    "adv_loss": "cce",
    "doser_lr": 0.0003,
    "doser_wd": 4e-07,
    "do_clip_grad": False,
    "gradient_clip_value": 5.0,
    "step_size_lr": 45,
}
    return ae_hparams,  trainer_params


def genTrainAndTest(adata, seed):
    filein = '../GEARS/data/train/splits/train_simulation_{}_0.8.pkl'.format(seed)
    with open(filein, 'rb') as fin:
        tmp = pickle.load(fin)
    split_key = {}
    for i in ['test', 'train', 'val']:
        for j in tmp[i]:
            split_key[j] = i
    adata.obs['split'] = adata.obs['condition'].apply(lambda x: split_key.get(x))
    return adata


def clean_condition(condition):
    return condition.replace('+ctrl', '').replace('ctrl+', '').strip()


def processComb(adata):
    for seed in seeds:
        filein = '../GEARS/data/train/splits/train_simulation_{}_0.8.pkl'.format(seed)
        with open(filein, 'rb') as fin:
            splits = pickle.load(fin)
        splits['train'] = [clean_condition(i) for i in splits['train']]
        splits['val'] = [clean_condition(i) for i in splits['val']]
        splits['test'] = [clean_condition(i) for i in splits['test']]
        splits_key = 'split_ood_multi_task{}'.format(seed)
        adata.obs[splits_key] = 'train'
        adata.obs.loc[adata.obs['perturbation'].isin(splits['train']), splits_key] = 'train'
        adata.obs.loc[adata.obs['perturbation'].isin(splits['val']), splits_key] = 'val'
        adata.obs.loc[adata.obs['perturbation'].isin(splits['test']), splits_key] = 'test'
    return adata

def runCPA_comb_chemical(DataSet, seed):
    dirName = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/CPA'.format(DataSet)
    if not os.path.isdir(dirName): os.makedirs(dirName)
    os.chdir(dirName)
    modeldir = 'savedModels{}'.format(seed)
    if not os.path.isdir(modeldir): os.makedirs(modeldir)
    adata = sc.read_h5ad('../../filter_hvg5000.h5ad')
    adata = processComb(adata)
    adata.write_h5ad('{}/cpa.h5ad'.format(modeldir))
    adata.write_h5ad('../../filter_hvg5000.h5ad')
    cpa.CPA.setup_anndata(adata,
                        perturbation_key='condition_ID',
                        dosage_key='log_dose',
                        control_group='control',
                        batch_key=None,
                        smiles_key='smiles_rdkit',
                        is_count_data=False,
                        categorical_covariate_keys=['cell_type'],
                        #deg_uns_key='rank_genes_groups_cov',
                        #deg_uns_cat_key='cov_drug_dose',
                        max_comb_len=2,
                        )

    ae_hparams = {'n_latent': 64,
    'recon_loss': 'gauss',
    'doser_type': 'linear',
    'n_hidden_encoder': 256,
    'n_layers_encoder': 3,
    'n_hidden_decoder': 512,
    'n_layers_decoder': 2,
    'use_batch_norm_encoder': True,
    'use_layer_norm_encoder': False,
    'use_batch_norm_decoder': True,
    'use_layer_norm_decoder': False,
    'dropout_rate_encoder': 0.25,
    'dropout_rate_decoder': 0.25,
    'variational': False,
    'seed': 6478}

    trainer_params = {'n_epochs_kl_warmup': None,
    'n_epochs_pretrain_ae': 50,
    'n_epochs_adv_warmup': 100,
    'n_epochs_mixup_warmup': 10,
    'mixup_alpha': 0.1,
    'adv_steps': None,
    'n_hidden_adv': 128,
    'n_layers_adv': 3,
    'use_batch_norm_adv': False,
    'use_layer_norm_adv': False,
    'dropout_rate_adv': 0.2,
    'reg_adv': 10.0,
    'pen_adv': 0.1,
    'lr': 0.0005,
    'wd': 4e-07,
    'adv_lr': 0.0003,
    'adv_wd': 4e-07,
    'adv_loss': 'cce',
    'doser_lr': 0.0003,
    'doser_wd': 4e-07,
    'do_clip_grad': False,
    'gradient_clip_value': 1.0,
    'step_size_lr': 10}

    model = cpa.CPA(adata=adata,
                split_key='split_ood_multi_task{}'.format(seed),
                train_split='train',
                valid_split='valid',
                test_split='test',
                use_rdkit_embeddings=True,
                **ae_hparams,
               )

    model.train(max_epochs= 500,
            use_gpu=True,
            batch_size=1024,
            plan_kwargs=trainer_params,
            early_stopping_patience=10,
            check_val_every_n_epoch=10,
            save_path= modeldir,
           )

def getPredict_chemical(DataSet, seed, ood = 'ood'):
    dirName = '/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/{}/hvg5000/CPA'.format(DataSet)
    os.chdir(dirName)
    split_key = 'split_ood_multi_task{}'.format(seed)
    modeldir = 'savedModels{}'.format(seed)
    adata = sc.read_h5ad('{}/cpa.h5ad'.format(modeldir))
    model = cpa.CPA.load(modeldir, adata = adata, use_gpu = True)
    model.predict(adata, batch_size=2048)
    adata_pred = adata[adata.obs[split_key] ==  ood].copy()
    adata_pred.X = adata_pred.obsm['CPA_pred']
    adata_pred.obs['Expcategory'] = 'imputed'

    adata_ctrl = adata[adata.obs['perturbation'] == 'control']
    adata_ctrl.obs['Expcategory'] = 'control'

    adata_stimulated = adata[adata.obs[split_key] == ood].copy()
    adata_stimulated.obs['Expcategory'] = 'stimulated'

    adata_fi = ad.concat([adata_ctrl, adata_pred, adata_stimulated])
    adata_fi.write('savedModels{}/result.h5ad'.format(seed))



DataSet = 'sciplex3_comb'
seeds = [1, 2, 3]
max_epoch = 150
torch.cuda.set_device('cuda:1')



if __name__ == '__main__':
    print ('hello, world')
    for seed in tqdm(seeds):
        runCPA_comb_chemical(DataSet, seed)
        getPredict_chemical(DataSet, seed, ood = 'test')
    