import os, sys
sys.path.append('/home/wzt/software/scPreGAN')
sys.path.append('/home/wzt/project/Pertb_benchmark')
from myUtil import *
from scPreGANUtil import * # type: ignore
from scPreGAN import * # type: ignore
import yaml
import shutil
import anndata as ad
import torch
import warnings
warnings.filterwarnings('ignore')
from nvitop import Device



def OutSample(X):
    try:
        outSample, hvg, DataLayer, redo = X
        basePath = '/home/wzt/project/Pertb_benchmark/DataSet/{}/outSample/hvg{}/scPreGAN/'.format(DataSet, hvg)
        tmp = '{}/{}'.format(basePath, outSample)
        if not os.path.isdir(tmp): os.makedirs(tmp)
        os.chdir(tmp)

        path = '/home/wzt/project/Pertb_benchmark/DataSet/{}/filter_hvg{}_{}.h5ad'.format(DataSet, hvg, DataLayer)
        adata = sc.read_h5ad(path)
        perturbations = list(adata.obs['condition2'].unique())  ### 
        perturbations = [i for i in perturbations if i != 'control']
        for perturbation in perturbations:
            if not redo and os.path.isfile('{}_imputed.h5ad'.format(perturbation)): continue
            ncell = adata[(adata.obs['condition1'] == outSample) & (adata.obs['condition2'] == perturbation)]
            if ncell.shape[0] == 0: continue    ####
            train_data = load_anndata(path=path,
                        condition_key='condition2',
                        condition= {'case': perturbation, 'control': 'control'},
                        cell_type_key='condition1',
                        out_sample_prediction= True,
                        prediction_type=outSample
                        )
            minSample = min(train_data[0].shape[0], train_data[2].shape[0])
            batch_size = 64 if minSample > 64 else int(minSample / 2) ##
            model = Model(n_features=adata.shape[1], n_classes=len(adata.obs['condition1'].unique()), use_cuda=True, epoch=20000)  #type: ignore
            # training
            model.train(train_data=train_data, niter=model.epoch, batch_size=batch_size)
            # predicting
            control_adata = adata[(adata.obs['condition1'] == outSample) & (adata.obs['condition2']=='control')]
            control_adata.obs['perturbation'] = 'control'  ### control
            pred_perturbed_adata = model.predict(control_adata=control_adata,
                            cell_type_key='condition1',
                            condition_key="condition2")
            
        ###control:  control_adata
 
            pred_perturbed_adata.obs['perturbation'] = 'imputed'
            pred_perturbed_adata.obs_names = control_adata.obs_names


            treat_adata = adata[(adata.obs['condition1'] == outSample) & (adata.obs['condition2']==perturbation)]
            treat_adata.obs['perturbation'] = 'stimulated'

            result = ad.concat([control_adata, treat_adata, pred_perturbed_adata])
            result.write_h5ad('{}_imputed.h5ad'.format(perturbation))
        
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







redo = False
DataSet = 'TCDDJointFactor'
torch.cuda.set_device('cuda:1')
