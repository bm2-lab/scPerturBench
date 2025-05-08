import os, sys
sys.path.append('/home/software/scPreGAN')
sys.path.append('/home/project/Pertb_benchmark')
from myUtil import *
from scPreGANUtil_iid import * # type: ignore
from scPreGAN import * # type: ignore
import anndata as ad
import torch
import warnings
warnings.filterwarnings('ignore')
from nvitop import Device

'''
conda activate cpa   ## 模改35行   /home/software/scPreGAN/scPreGANUtil_iid.py
'''


### 对kang数据进行out sample test 
def Kang_inSample(DataSet):
    basePath = f'/home/project/Pertb_benchmark/DataSet/{DataSet}/inSample/hvg5000/scPreGAN/'
    if not os.path.isdir(basePath): os.makedirs(basePath)
    os.chdir(basePath)
    path = f'/home/project/Pertb_benchmark/DataSet/{DataSet}/filter_hvg5000_logNor.h5ad'
    adata = sc.read_h5ad(path)
    outSamples = list(adata.obs['condition1'].unique())
    perturbations = list(adata.obs['condition2'].unique())
    perturbations = [i for i in perturbations if i != 'control']
    for perturbation in perturbations:
        train_data = load_anndata(path=path, # type: ignore
                    condition_key='condition2',
                    condition= {'case': perturbation, 'control': 'control'},
                    cell_type_key='condition1',
                    out_sample_prediction = False,
                    prediction_type=None
                    )
        minSample = min(train_data[0].shape[0], train_data[2].shape[0])
        batch_size = 64 if minSample > 64 else int(minSample / 2)
        model = Model(n_features=adata.shape[1], n_classes=len(adata.obs['condition1'].unique()), use_cuda=True, epoch=20000)  #type: ignore    epoch=20000
        # training
        model.train(train_data=train_data, niter=model.epoch, batch_size=batch_size)
        # predicting
        control_adata = adata[(adata.obs['iid_test'] == 'test') & (adata.obs['condition2']=='control')]
        control_adata.obs['perturbation'] = 'control'  ### control
        pred_perturbed_adata = model.predict(control_adata=control_adata,
                        cell_type_key='condition1',
                        condition_key="condition2")
        pred_perturbed_adata.obs['perturbation'] = 'imputed'
        pred_perturbed_adata.obs['condition2'] = perturbation
        pred_perturbed_adata.obs_names = control_adata.obs_names
        treat_adata = adata[(adata.obs['iid_test'] == 'test') & (adata.obs['condition2']==perturbation)]
        treat_adata.obs['perturbation'] = 'stimulated'
        result = ad.concat([control_adata, treat_adata, pred_perturbed_adata])
        
        for outSample in outSamples:
            if not os.path.isdir(outSample): os.makedirs(outSample)
            tmp = result[result.obs['condition1'] == outSample]
            tmp.write_h5ad('{}/{}_imputed.h5ad'.format(outSample, perturbation))


DataSet = 'kangCrossCell'
torch.cuda.set_device('cuda:1')
if __name__ == '__main__':
    Kang_inSample(DataSet)