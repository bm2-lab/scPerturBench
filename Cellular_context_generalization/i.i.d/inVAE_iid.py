import os, sys
sys.path.append('/home/wzt/software/inVAE')
sys.path.append('/home/wzt/project/Pertb_benchmark')
from myUtil import *
from INVAE import main #type: ignore
from inVAE_model import * #type: ignore
from inVAE_utils import * #type: ignore
from inVAE_data import get_adata #type: ignore
from nvitop import Device
import anndata as ad
import torch

import warnings
warnings.filterwarnings('ignore')


def inSample(X):
    outSample, hvg, DataLayer, redo = X
    outSample_test = outSample + '_test'  
    basePath = '/home/wzt/project/Pertb_benchmark/DataSet/{}/inSample/hvg{}/inVAE/'.format(DataSet, hvg)
    tmp = '{}/{}'.format(basePath, outSample)
    if not os.path.isdir(tmp): os.makedirs(tmp)
    os.chdir(tmp)
    
    path = '/home/wzt/project/Pertb_benchmark/DataSet/{}/filter_hvg{}_{}.h5ad'.format(DataSet, hvg, DataLayer)
    myAdata = sc.read_h5ad(path)
    myAdata.obs['condition3'] = myAdata.obs['condition1'].astype(str) + '_' + myAdata.obs['iid_test'].astype(str) ##
 
    perturbations = list(myAdata.obs['condition2'].unique())  #
    perturbations = [i for i in perturbations if i != 'control']

    for perturbation in perturbations:
        if not redo and os.path.isfile('{}_imputed.h5ad'.format(perturbation)): continue
        try:
            adata, pred_cell_N, targ_cond_N, pred_cond_N = get_adata(anndata_path = myAdata, 
                                                                     pred_cell = outSample_test, 
                                                                     targ_condition = "control", 
                                                                     pred_condition= perturbation,
                                                                     cell_type_key="condition3", 
                                                                     condition_type_key="condition2")
            adata = adata[adata[:,-2]==pred_cell_N] ####
            fea_n = adata.shape[1] - 2
            adata = adata[:,:fea_n]
            real_mean_ = np.mean(adata.numpy(), axis=0)   ##

            model = main(ann_path=myAdata, pred_cell=outSample_test, targ_condition=perturbation, pred_condition="control", cell_type_key="condition3", condition_type_key="condition2", real_mean_=real_mean_, epoch_N=epoch_N, bat_N=bat_N)
            adata, pred_cell_N, targ_cond_N, pred_cond_N = get_adata(anndata_path = myAdata, pred_cell = outSample_test, targ_condition = perturbation, pred_condition= "control", cell_type_key="condition3", condition_type_key="condition2")
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
            adata = myAdata.copy()
            adata_control = adata[(adata.obs["condition3"] == outSample_test) & (adata.obs["condition2"] == "control")]
            adata_control.obs['perturbation'] = 'control'

            imputed = ad.AnnData(outArray, obs=pd.DataFrame(index=adata_control.obs_names), var=adata.var)
            imputed.obs['perturbation'] = 'imputed'

            treat_adata = adata[(adata.obs['condition3'] == outSample_test) & (adata.obs['condition2']== perturbation)]
            treat_adata.obs['perturbation'] = 'stimulated'
            result = ad.concat([adata_control, treat_adata, imputed])
            result.write_h5ad('{}_imputed.h5ad'.format(perturbation))
        except Exception as e:  ###
            adata = myAdata.copy()
            ###control:  control_adata
            adata_control = adata[(adata.obs["condition3"] == outSample_test) & (adata.obs["condition2"] == "control")]
            adata_control.obs['perturbation'] = 'control'

            imputed = adata_control.copy()
            imputed.obs['perturbation'] = 'imputed'
            treat_adata = adata[(adata.obs['condition3'] == outSample_test) & (adata.obs['condition2']== perturbation)]
            treat_adata.obs['perturbation'] = 'stimulated'

            result = ad.concat([adata_control, treat_adata, imputed])
            result.write_h5ad('{}_imputed.h5ad'.format(perturbation))
            print ('**********************************************************\n')
            print (e, perturbation, outSample)
    


multiprocessing.set_start_method('spawn', force=True)
epoch_N= 100   #
bat_N = 1024   




DataSet = 'sciplex3'
torch.cuda.set_device('cuda:0')


