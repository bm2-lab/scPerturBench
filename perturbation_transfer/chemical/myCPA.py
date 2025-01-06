import sys, subprocess, os
sys.path.append('/home/wzt/project/Pertb_benchmark')
from myUtil1 import *
from scipy import sparse
import anndata as ad
import anndata
from sklearn.metrics import mean_squared_error
import cpa
from gears import PertData
import pickle
import torch.nn as nn
import joblib
import torch
from collections import defaultdict, OrderedDict
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


#### 使用和gears一样的训练 验证和测试集
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

def runCPA_genetic(DataSet, seed):
    os.chdir('/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/{}/hvg5000/GEARS'.format(DataSet))
    pert_data = PertData('./data') # specific saved folder   下载 gene2go_all.pkl
    pert_data.load(data_path = './data/train') # load the processed data, the path is saved folder + dataset_name

    dirName = '/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/{}/hvg5000/CPA'.format(DataSet)
    if not os.path.isdir(dirName): os.makedirs(dirName)
    os.chdir(dirName)

    modeldir_pt = 'savedModels{}/model.pt'.format(seed)
    if os.path.isfile(modeldir_pt): pass     #### 避免重复跑
    adata = pert_data.adata
    cpa.CPA.setup_anndata(adata,
                    perturbation_key='perturbation',
                    control_group='control',
                    dosage_key = None,
                    batch_key = None,
                    is_count_data= False,
                    max_comb_len=2,
                    )
    adata = genTrainAndTest(adata, seed)
    modeldir = 'savedModels{}'.format(seed)
    if not os.path.isdir(modeldir): os.makedirs(modeldir)
    adata.write_h5ad('{}/cpa.h5ad'.format(modeldir))
    data_ge = joblib.load("/home/wzt/project/GW_PerturbSeq/geneEmbedding/scGPT.pkl") ### 改变embedding以预测unseen的结果
    
    gene_list = list(data_ge.keys())
    gene_embeddings = np.array(list(data_ge.values()))
    gene_embeddings = np.concatenate((gene_embeddings, np.random.rand(1, gene_embeddings.shape[1])), 0)
    embeddings = anndata.AnnData(X=gene_embeddings)
    embeddings.obs.index = gene_list+['control']
    perturb_genes = list(cpa.CPA.pert_encoder.keys())
    perturb_genes.remove('<PAD>')
    embeddings = embeddings[perturb_genes,:]

    GENE_embeddings = nn.Embedding(len(perturb_genes)+1, embeddings.shape[1], padding_idx = 0)
    pad_X = np.zeros(shape=(1, embeddings.shape[1]))
    X = np.concatenate((pad_X, embeddings.X), 0)
    GENE_embeddings.weight.data.copy_(torch.tensor(X))
    GENE_embeddings.weight.requires_grad = False

    if not os.path.isdir(modeldir): os.makedirs(modeldir)
    ae_hparams,  trainer_params = getModelParameter()
    model = cpa.CPA(adata=adata,
                use_rdkit_embeddings=True, 
                gene_embeddings = GENE_embeddings,
                split_key='split',
                train_split='train',
                valid_split='val',
                test_split='test',
                **ae_hparams,
               )

    model.train(max_epochs=max_epoch,
            use_gpu= 1,
            batch_size=1024,
            plan_kwargs=trainer_params,
            early_stopping_patience=5,
            check_val_every_n_epoch=5,
            save_path= modeldir,
           )



def runCPA_chemical(DataSet, seed):
    dirName = '/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/{}/hvg5000/CPA'.format(DataSet)
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
    dirName = '/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/{}/hvg5000/CPA'.format(DataSet)
    if not os.path.isdir(dirName): os.makedirs(dirName)
    os.chdir(dirName)
    modeldir = 'savedModels{}'.format(seed)
    if not os.path.isdir(modeldir): os.makedirs(modeldir)
    adata = sc.read_h5ad('../../filter_hvg5000.h5ad')
    adata = processComb(adata)  #### 根据GEARS标注分割 train, test, val数据
    adata.write_h5ad('{}/cpa.h5ad'.format(modeldir))
    adata.write_h5ad('../../filter_hvg5000.h5ad')  #### 覆盖原有数据, 方便linearModel mean chemical使用
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
                use_rdkit_embeddings=True,  ### 重要参数
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




### 使用预测的值进行差异基因计算
def getPredict(DataSet, seed):
    dirName = '/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/{}/hvg5000/CPA'.format(DataSet)
    os.chdir(dirName)

    modeldir = 'savedModels{}'.format(seed)
    adata = sc.read_h5ad('{}/cpa.h5ad'.format(modeldir))
    model = cpa.CPA.load(modeldir, adata = adata, use_gpu = True)
    model.predict(adata, batch_size=2048)
    adata_pred = adata[adata.obs['split'] == 'test'].copy()
    adata_pred.X = adata_pred.obsm['CPA_pred']
    adata_pred.obs['Expcategory'] = 'imputed'

    adata_ctrl = adata[adata.obs['perturbation'] == 'control']
    adata_ctrl.obs['Expcategory'] = 'control'

    adata_stimulated = adata[adata.obs['split'] == 'test'].copy()
    adata_stimulated.obs['Expcategory'] = 'stimulated'

    adata_fi = ad.concat([adata_ctrl, adata_pred, adata_stimulated])
    adata_fi.write('savedModels{}/result.h5ad'.format(seed))

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



SinglePertDataSets = ['Adamson', "Frangieh", "TianActivation", "TianInhibition", "Replogle_exp7", "Replogle_exp8", "Papalexi", "Replogle_RPE1essential", "Replogle_K562essential"]
CombPertDataSets = ['Norman', 'Schmidt', "Replogle_exp6", "Replogle_exp10"]
chemicalDataSets = ['sciplex3_MCF7', 'sciplex3_A549', 'sciplex3_K562', 'sciplex3_comb']

DataSet = 'sciplex3_comb'   ### sciplex3_comb
isGenetic = False
seeds = [1, 2, 3]
max_epoch = 500
torch.cuda.set_device('cuda:1')

    