import os, sys
sys.path.append('/home/wzt/software/scPreGAN')
sys.path.append('/home/wzt/project/Pertb_benchmark')
from myUtil import *
from scPreGANUtil_iid import * # type
from scPreGAN import * # type: ignore
import anndata as ad
import torch
import warnings
warnings.filterwarnings('ignore')
from nvitop import Device




def inSample(X):
    try:
        hvg, DataLayer = X
        basePath = '/home/wzt/project/Pertb_benchmark/DataSet/{}/inSample/hvg{}/scPreGAN/'.format(DataSet, hvg)
        if not os.path.isdir(basePath): os.makedirs(basePath)
        os.chdir(basePath)

        path = '/home/wzt/project/Pertb_benchmark/DataSet/{}/filter_hvg{}_{}.h5ad'.format(DataSet, hvg, DataLayer)
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
            
        ###control:  control_adata
        #### impute的h5ad 
            pred_perturbed_adata.obs['perturbation'] = 'imputed'
            pred_perturbed_adata.obs['condition2'] = perturbation
            pred_perturbed_adata.obs_names = control_adata.obs_names

        #### treat data
            treat_adata = adata[(adata.obs['iid_test'] == 'test') & (adata.obs['condition2']==perturbation)]
            treat_adata.obs['perturbation'] = 'stimulated'
            result = ad.concat([control_adata, treat_adata, pred_perturbed_adata])
            
            for outSample in outSamples:
                if not os.path.isdir(outSample): os.makedirs(outSample)
                tmp = result[result.obs['condition1'] == outSample]
                tmp.write_h5ad('{}/{}_imputed.h5ad'.format(outSample, perturbation))
        
        pid= os.getpid()
        gpu_memory = pd.DataFrame(1, index=['gpu_memory'], columns=['gpu_memory'], dtype=str)
        devices = Device.all()
        for device in devices:
            processes = device.processes()
            if pid in processes.keys():
                p=processes[pid]
                gpu_memory['gpu_memory'] = p.gpu_memory_human()
        gpu_memory.to_csv('gpu_memory.csv', sep='\t', index=False)
    except Exception as e: 
        print ('myError********************************************************\n')
        print (e)
        print (X, perturbation)

