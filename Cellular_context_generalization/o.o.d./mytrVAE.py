import os, sys
sys.path.append('/home/project/Pertb_benchmark')
from myUtil import *
import scarches as sca
import gdown
from scipy.sparse import issparse
import anndata as ad
import torch
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
 
def Kang_OutSample(DataSet, outSample):
    basePath = f'/home/project/Pertb_benchmark/DataSet/{DataSet}/outSample/hvg5000/trVAE/'
    tmp = '{}/{}'.format(basePath, outSample)
    if not os.path.isdir(tmp): os.makedirs(tmp)
    os.chdir(tmp)

    path = f'/home/project/Pertb_benchmark/DataSet/{DataSet}/filter_hvg5000_logNor.h5ad'
    adata = sc.read_h5ad(path)
    adata.X = adata.X.astype(np.float32)
    perturbations = list(adata.obs['condition2'].unique())
    perturbations = [i for i in perturbations if i != 'control']

    cell_type_key = 'condition1'; condition_key = 'condition2'
    conditions = list(adata.obs['condition2'].unique())
    target_conditions = [i for i in conditions if i != 'control']
    specific_celltype = outSample
    train_adata = adata[~((adata.obs[cell_type_key] == specific_celltype) & (adata.obs[condition_key].isin(target_conditions)))]
    adata_source = train_adata[train_adata.obs[cell_type_key] == specific_celltype]
    try:
        trvae = sca.models.TRVAE( adata=train_adata, condition_key=condition_key,
                conditions=conditions, hidden_layer_sizes=[128, 128],
                recon_loss="mse", dr_rate=0.2,
                use_bn=True
                )
        
        trvae.train(n_epochs=trvae_epochs, alpha_epoch_anneal=200,
                early_stopping_kwargs=early_stopping_kwargs,
                batch_size=512, clip_value=100,   
            )
        
        for perturbation in perturbations:
            corrected_data = perturbation_prediction(trvae, adata_source, "control", perturbation)

        ###control:  control_adata
            adata_source.obs['perturbation'] = 'control'
        #### impute的h5ad
            imputed = adata_source[adata_source.obs[condition_key] == "control"].copy()
            imputed.X = corrected_data
            imputed.obs['perturbation'] = 'imputed'

        #### 获得treat的
            treat_adata = adata[(adata.obs['condition1'] == outSample) & (adata.obs['condition2']==perturbation)]
            treat_adata.obs['perturbation'] = 'stimulated'
            result = ad.concat([adata_source, treat_adata, imputed])
            result.write_h5ad('{}_imputed.h5ad'.format(perturbation))
    except Exception as e:
        print (X, e, '***********************************************\n')
        for perturbation in perturbations:
        ###control:  control_adata
            adata_source.obs['perturbation'] = 'control'
        #### impute的h5ad
            imputed = adata_source.copy()
            imputed.obs['perturbation'] = 'imputed'
        #### 获得treat的
            treat_adata = adata[(adata.obs['condition1'] == outSample) & (adata.obs['condition2']==perturbation)]
            treat_adata.obs['perturbation'] = 'stimulated'
            result = ad.concat([adata_source, treat_adata, imputed])
            result.write_h5ad('{}_imputed.h5ad'.format(perturbation))

         

def KangMain():
    mylist = []
    filein_tmp = '/home/project/Pertb_benchmark/DataSet/{}/filter_hvg5000_logNor.h5ad'.format(DataSet)
    adata_tmp = sc.read_h5ad(filein_tmp)
    outSamples = list(adata_tmp.obs['condition1'].unique())
    for outSample in outSamples:
        Kang_OutSample(DataSet, outSample)

trvae_epochs = 1000
DataSet = 'kangCrossCell'
torch.cuda.set_device('cuda:1')

### conda activate cpa


if __name__ == '__main__':
    KangMain()