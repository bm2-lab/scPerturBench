import os, sys
sys.path.append('/home/wzt/project/Pertb_benchmark')
from myUtil import *
import scarches as sca
from scipy.sparse import issparse
import anndata as ad
import torch
from nvitop import Device
import warnings
warnings.filterwarnings('ignore')


early_stopping_kwargs = {
    "early_stopping_metric": "val_unweighted_loss",
    "threshold": 0,
    "patience": 100,
    "reduce_lr": False
}

def get_reconstruction(model, x, encoder_labels=None, decoder_labels = None):

    x_ = torch.log(1 + x)
    if model.model.recon_loss == 'mse':
        x_ = x
    
    z_mean, z_log_var = model.model.encoder(x_, encoder_labels)
    latent = model.model.sampling(z_mean, z_log_var)
    output = model.model.decoder(latent, decoder_labels)
    return output[0]

def perturbation_prediction(model, adata, source_cond, target_cond):
    device = next(model.model.parameters()).device

    source_adata = adata[adata.obs["condition2"] == source_cond]

    from scarches.dataset.trvae._utils import label_encoder

    encoder_labels = label_encoder(source_adata, model.model.condition_encoder, "condition2")
    decoder_labels = np.zeros_like(encoder_labels) + model.model.condition_encoder[target_cond]

    x = adata.X

    latents = []
    indices = torch.arange(x.shape[0])
    subsampled_indices = indices.split(512)
    for batch in subsampled_indices:
            x_batch = x[batch, :]
            if issparse(x_batch):
                    x_batch = x_batch.toarray()
            x_batch = torch.tensor(x_batch, device=device)
            encoder_labels = torch.tensor(encoder_labels, device=device)
            decoder_labels = torch.tensor(decoder_labels, device=device)
            latent = get_reconstruction(model, x_batch, encoder_labels[batch], decoder_labels[batch])
            latents += [latent.cpu().detach()]

    return np.array(torch.cat(latents))

def inSample(X):
    hvg, DataLayer = X
    basePath = '/home/wzt/project/Pertb_benchmark/DataSet/{}/inSample/hvg{}/trVAE/'.format(DataSet, hvg)
    if not os.path.isdir(basePath): os.makedirs(basePath)
    os.chdir(basePath)

    path = '/home/wzt/project/Pertb_benchmark/DataSet/{}/filter_hvg{}_{}.h5ad'.format(DataSet, hvg, DataLayer)
    adata = sc.read_h5ad(path)
    adata.X = adata.X.astype(np.float32)
    outSamples = list(adata.obs['condition1'].unique())
    perturbations = list(adata.obs['condition2'].unique())  
    perturbations = [i for i in perturbations if i != 'control']

    cell_type_key = 'condition1'; condition_key = 'condition2'
    conditions = list(adata.obs['condition2'].unique())  
    train_adata = adata[adata.obs['iid_test'] == 'train']  
    
    trvae = sca.models.TRVAE(adata=train_adata, condition_key=condition_key,
            conditions=conditions, hidden_layer_sizes=[128, 128],
            recon_loss="mse", dr_rate=0.2,
            use_bn=True
            )
    
    trvae.train(n_epochs=trvae_epochs, alpha_epoch_anneal=200,
            early_stopping_kwargs=early_stopping_kwargs,
            batch_size=512, clip_value=100, 
        )
    

    for specific_celltype in outSamples:
        outSample = specific_celltype
        adata_source = adata[(adata.obs[cell_type_key] ==  specific_celltype) & (adata.obs['iid_test'] == 'test') & (adata.obs[condition_key] == 'control')] ### control
        for perturbation in perturbations:
            corrected_data = perturbation_prediction(trvae, adata_source, "control", perturbation) 

        ###control:  control_adata
            adata_source.obs['perturbation'] = 'control'

            imputed = adata_source.copy()
            imputed.X = corrected_data
            imputed.obs['perturbation'] = 'imputed'

            treat_adata = adata[(adata.obs[cell_type_key] ==  specific_celltype) & (adata.obs['iid_test'] == 'test') & (adata.obs[condition_key] == perturbation)] ### control
            treat_adata.obs['perturbation'] = 'stimulated'

            result = ad.concat([adata_source, treat_adata, imputed]) 
            if not os.path.isdir(outSample): os.makedirs(outSample)
            result.write_h5ad('{}/{}_imputed.h5ad'.format(outSample, perturbation))
                
