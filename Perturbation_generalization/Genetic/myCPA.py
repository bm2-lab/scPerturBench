import sys, subprocess, os
sys.path.append('/home/project/Pertb_benchmark')
from myUtil1 import *
import scanpy as sc, pandas as pd, numpy as np, anndata as ad
import anndata
from gears import PertData
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

def runCPA_genetic(DataSet, seed):
    dirName = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/GEARS'.format(DataSet)
    os.chdir(dirName)

    pert_data = PertData('./data')
    pert_data.load(data_path = './data/train')

    dirName = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/CPA'.format(DataSet)
    if not os.path.isdir(dirName): os.makedirs(dirName)
    os.chdir(dirName)

    modeldir_pt = 'savedModels{}/model.pt'.format(seed)
    if os.path.isfile(modeldir_pt): pass
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
    data_ge = joblib.load("/home/project/GW_PerturbSeq/geneEmbedding/scGPT.pkl")
    
    gene_list = list(data_ge.keys())
    gene_embeddings = np.array(list(data_ge.values()))
    gene_embeddings = np.concatenate((gene_embeddings, np.random.rand(1, gene_embeddings.shape[1])), 0)
    embeddings = anndata.AnnData(X=gene_embeddings)
    embeddings.obs.index = gene_list+['control']
    mean_embedding = embeddings.X.mean(axis=0, keepdims=True)
    perturb_genes = list(cpa.CPA.pert_encoder.keys())
    perturb_genes.remove('<PAD>')
    perturb_X = []
    valid_genes = []
    for g in perturb_genes:
        if g in embeddings.obs.index:
            perturb_X.append(embeddings[g].X[0])
        else:
            perturb_X.append(mean_embedding[0])
        valid_genes.append(g)
    perturb_X = np.stack(perturb_X, axis=0)
    embeddings = anndata.AnnData(X=perturb_X)
    embeddings.obs.index = valid_genes

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



def clean_condition(condition):
    return condition.replace('+ctrl', '').replace('ctrl+', '').strip()

def getPredict(DataSet, seed):
    dirName = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/CPA'.format(DataSet)
    os.chdir(dirName)

    modeldir = 'savedModels{}'.format(seed)
    adata = sc.read_h5ad('{}/cpa.h5ad'.format(modeldir))
    model = cpa.CPA.load(modeldir, adata = adata, use_gpu = True)


    a = adata[adata.obs['split'] == 'test'].shape[0]
    tmp = adata[adata.obs['perturbation'] == 'control']
    tmp = tmp.to_df().sample(n=a, random_state=42, replace=True)
    adata[adata.obs['split'] == 'test'].X = tmp.values

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


SinglePertDataSets = ['Adamson', "Frangieh", "TianActivation", "TianInhibition", "Replogle_exp7", "Replogle_exp8", "Papalexi", "Replogle_RPE1essential", "Replogle_K562essential"]
CombPertDataSets = ['Norman', 'Wessels', 'Schmidt', "Replogle_exp6", "Replogle_exp10"]

DataSet = 'Papalexi'
seeds = [1, 2, 3]
max_epoch = 500
torch.cuda.set_device('cuda:0')

if __name__ == '__main__':
    print ('hello, world')
    for seed in tqdm(seeds):
        runCPA_genetic(DataSet, seed)
        #getPredict(DataSet, seed)

    