import os, sys
sys.path.append('/home//software/inVAE')
sys.path.append('/home//project/Pertb_benchmark')
from myUtil import *
from INVAE import INVAE, main #type: ignore
from inVAE_model import * #type: ignore
from inVAE_utils import * #type: ignore
from inVAE_data import get_adata #type: ignore
import yaml
import shutil
import anndata as ad
import torch

import warnings
warnings.filterwarnings('ignore')

def Kang_OutSample(outSample):
    basePath = f'/home//project/Pertb_benchmark/DataSet/{DataSet}/outSample/hvg5000/inVAE/'
    tmp = '{}/{}'.format(basePath, outSample)
    if not os.path.isdir(tmp): os.makedirs(tmp)
    os.chdir(tmp)
    
    path = f'/home//project/Pertb_benchmark/DataSet/{DataSet}/filter_hvg5000_logNor.h5ad'
    cond_t_key = 'condition2'
    p_cond = 'control'
    cell_t_key = 'condition1'
    p_cell =  outSample
    tmp = sc.read_h5ad(path)
    perturbations = list(tmp.obs['condition2'].unique())
    perturbations = [i for i in perturbations if i != 'control']
    for perturbation in perturbations:
        ncell = tmp[(tmp.obs['condition1'] == outSample) & (tmp.obs['condition2'] == perturbation)]
        if ncell.shape[0] == 0: continue
        t_cond = perturbation
        try:
            adata, pred_cell_N, targ_cond_N, pred_cond_N = get_adata(anndata_path = path, pred_cell = p_cell, targ_condition = p_cond, pred_condition=t_cond ,cell_type_key=cell_t_key, condition_type_key=cond_t_key)
            adata = adata[adata[:,-2]==pred_cell_N]
            fea_n = adata.shape[1] - 2
            adata = adata[:,:fea_n]
            real_mean_ = np.mean(adata.numpy(), axis=0)

            model = main(ann_path=path, pred_cell=p_cell, targ_condition=t_cond, pred_condition=p_cond,epoch_N=epoch_N, bat_N=bat_N,cell_type_key=cell_t_key, condition_type_key=cond_t_key,real_mean_=real_mean_)

            adata, pred_cell_N, targ_cond_N, pred_cond_N = get_adata(anndata_path = path, pred_cell = p_cell, targ_condition = t_cond, pred_condition=p_cond,cell_type_key=cell_t_key, condition_type_key=cond_t_key)
            adata = adata[adata[:,-2]==pred_cell_N]
            z_params_o = model.encoder.forward(adata.cuda()).view(adata.size(0), (60), 2)
            z_params_o = torch.clamp(z_params_o, min=-30, max=30)
            z_params = Normal().sample(params=z_params_o).cuda() #type: ignore
            z_com = z_params[:,:(30)]
            z_spe = z_params[:,(30):(60)]
            impute_D = adata[:,-1].reshape([-1,1]).cuda() 
            impute_D[:] = targ_cond_N
            impute_C = adata[:,-2].reshape([-1,1]).cuda()
            inp_spe = torch.cat((z_spe,impute_D),1)
            inp_con = torch.cat((z_com,impute_C),1)
            z_proj_N = model.p_layer.forward(inp_spe)
            x_params_spe_N = model.decoder_S.forward(z_proj_N)
            x_params_spe_N = x_params_spe_N.view(len(inp_con), fea_n)
            x_params_com_N = model.decoder_C.forward(inp_con).view(len(inp_con), fea_n)
            x_params_N = x_params_spe_N + x_params_com_N 
            outArray = x_params_N.detach().cpu().numpy()
        ###control:  control_adata
            adata = sc.read_h5ad(path)
            adata_control = adata[(adata.obs["condition1"] == outSample) & (adata.obs["condition2"] == "control")]
            adata_control.obs['perturbation'] = 'control'

            imputed = ad.AnnData(outArray, obs=pd.DataFrame(index=adata_control.obs_names), var=adata.var)
            imputed.obs['perturbation'] = 'imputed'

            treat_adata = adata[(adata.obs['condition1'] == outSample) & (adata.obs['condition2']== perturbation)]
            treat_adata.obs['perturbation'] = 'stimulated'
            result = ad.concat([adata_control, treat_adata, imputed])
            result.write_h5ad('{}_imputed.h5ad'.format(perturbation))
        except Exception as e: 
            adata = sc.read_h5ad(path)
            ###control:  control_adata
            adata_control = adata[(adata.obs["condition1"] == outSample) & (adata.obs["condition2"] == "control")]
            adata_control.obs['perturbation'] = 'control'
            imputed = adata_control.copy()
            imputed.obs['perturbation'] = 'imputed'
            treat_adata = adata[(adata.obs['condition1'] == outSample) & (adata.obs['condition2']== perturbation)]
            treat_adata.obs['perturbation'] = 'stimulated'

            result = ad.concat([adata_control, treat_adata, imputed])
            result.write_h5ad('{}_imputed.h5ad'.format(perturbation))
            print ('**********************************************************\n')
            print (e, perturbation, p_cell)


def KangMain():
    mylist = []
    filein_tmp = '/home//project/Pertb_benchmark/DataSet/{}/filter_hvg5000_logNor.h5ad'.format(DataSet)
    adata_tmp = sc.read_h5ad(filein_tmp)
    outSamples = list(adata_tmp.obs['condition1'].unique())
    for outSample in outSamples:
        Kang_OutSample(outSample)

DataSet = 'kangCrossCell'
torch.cuda.set_device('cuda:0')

epoch_N= 5
bat_N = 1024

## conda activate  cpa

if __name__ == '__main__':
    KangMain()