import sys, subprocess, os
sys.path.append('/home/project/Pertb_benchmark')
from myUtil1 import *
import scanpy as sc, pandas as pd, numpy as np, anndata as ad
import anndata
import pickle, tqdm, joblib, torch, cpa
import torch.nn as nn
sc.settings.verbosity = 3

def runCPA_chemical(DataSet, seed):
    dirName = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/CPA'.format(DataSet)
    if not os.path.isdir(dirName): os.makedirs(dirName)
    os.chdir(dirName)
    modeldir = 'savedModels{}'.format(seed)
    if not os.path.isdir(modeldir): os.makedirs(modeldir)
    adata = sc.read_h5ad('../../filter_hvg5000.h5ad')
    cmd = 'ln -sf ../../../filter_hvg5000.h5ad  {}/cpa.h5ad'.format(modeldir)
    subprocess.call(cmd, shell=True)

    cpa.CPA.setup_anndata(adata,
                        perturbation_key='condition',
                        dosage_key='dose',
                        control_group='control',
                        batch_key=None,
                        smiles_key='SMILES',
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
                test_split='ood',
                use_rdkit_embeddings=True,
                **ae_hparams,
               )

    model.train(max_epochs= max_epoch,
            use_gpu=True,
            batch_size=1024,
            plan_kwargs=trainer_params,
            early_stopping_patience=10,
            check_val_every_n_epoch=10,
            save_path= modeldir,
           )

def clean_condition(condition):
    return condition.replace('+ctrl', '').replace('ctrl+', '').strip()


def getPredict_chemical(DataSet, seed, ood = 'ood'):
    dirName = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/CPA'.format(DataSet)
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



chemicalDataSets = ['sciplex3_MCF7', 'sciplex3_A549', 'sciplex3_K562']
DataSet = 'sciplex3_A549'
seeds = [1, 2, 3]
max_epoch = 150
torch.cuda.set_device('cuda:0')



if __name__ == '__main__':
    print ('hello, world')
    for seed in tqdm(seeds):
        runCPA_chemical(DataSet, seed)
        getPredict_chemical(DataSet, seed)    