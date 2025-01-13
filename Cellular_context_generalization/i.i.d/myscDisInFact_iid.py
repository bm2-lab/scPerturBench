import os, sys
sys.path.append('/home/wzt/project/Pertb_benchmark')
from myUtil import *
import yaml
import shutil
import anndata as ad
import torch
from scDisInFact import scdisinfact, create_scdisinfact_dataset
import warnings
warnings.filterwarnings('ignore')
from nvitop import Device

multiprocessing.set_start_method('spawn', force=True)


def getPrediction(adata, model, counts_input, meta_input, outSample, perturbation, meta_input1, Batch, attrList1):
    attrList2 = [i for i in attrList1 if i not in ['condition1', 'condition2']]
    meta_input2 = meta_input1[meta_input1['batch'] == Batch]
    tmp = meta_input2[attrList2].values.tolist()
    unique_conditions= list(set(tuple(sublist) for sublist in tmp))
    counts_predict_list = []
    for condition in unique_conditions:
        try:
            counts_predict = model.predict_counts(input_counts = counts_input, meta_cells = meta_input, condition_keys = attrList1, 
                                        batch_key = "batch", predict_conds = [outSample, perturbation] + list(condition), predict_batch=Batch)

            #### get imputed h5ad
            imputed = ad.AnnData(counts_predict, obs=meta_input, var=adata.var)
            imputed.obs['perturbation'] = 'imputed'

            for attrList_tmp1, attrList_tmp2 in zip(attrList2 + ['batch'], list(condition) + [Batch]):
                imputed.obs[attrList_tmp1] = attrList_tmp2
            counts_predict_list.append(imputed)
        except:
            continue
    if len(counts_predict_list) >= 1:
        imputed = ad.concat(counts_predict_list)
        return imputed
    else:
        return []


def inSample_batch(X):
    hvg, DataLayer  = X
    basePath = '/home/wzt/project/Pertb_benchmark/DataSet/{}/inSample/hvg{}/scDisInFact/'.format(DataSet, hvg)
    if not os.path.isdir(basePath): os.makedirs(basePath)
    os.chdir(basePath)

    path = '/home/wzt/project/Pertb_benchmark/DataSet/{}/filter_hvg{}_{}.h5ad'.format(DataSet, hvg, DataLayer)
    adata = sc.read_h5ad(path)
    outSamples = list(adata.obs['condition1'].unique())
    perturbations = list(adata.obs['condition2'].unique())  ###
    perturbations = [i for i in perturbations if i != 'control']
    counts = adata.to_df().values
    attrList = [i for i in attrList_all if i in adata.obs.columns]  ### 

    meta_cells = adata.obs[attrList].copy()
    for attr in attrList:
        meta_cells[attr] = meta_cells[attr].astype(str)

    if 'batch' not in attrList:
        meta_cells['batch'] = 1  ##

    attrList1 = [i for i in attrList if i != 'batch'] ###
    test_idx = ((adata.obs["iid_test"] == 'test')).values
    train_idx = ~test_idx
    data_dict = create_scdisinfact_dataset(counts[train_idx,:], meta_cells.loc[train_idx,:], condition_key = attrList1, batch_key = "batch")
    
    if 'batch' in attrList: Ks = [8] + [4] * (len(attrList) - 1)
    else: Ks = [8] + [4] * (len(attrList))
    
    model = scdisinfact(data_dict = data_dict, device = device, Ks = Ks)
    model.train()
    if DataLayer == 'counts':
        model.train_model(nepochs = nepochs, recon_loss = "NB")   #### MSE
    else:
        model.train_model(nepochs = nepochs, recon_loss = "MSE")   #### MSE
    #torch.save(model.state_dict(),  f"model.pth")
    #model.load_state_dict(torch.load(f"model.pth", map_location = device))
    _ = model.eval()

    for outSample in outSamples:
        input_idx = ((meta_cells["condition1"] == outSample) & (meta_cells["condition2"] == "control") & (adata.obs["iid_test"] == "train")).values
        counts_input = counts[input_idx,:]  ### control 
        meta_input = meta_cells.loc[input_idx,:]  ### control
        for perturbation in perturbations:
            input_idx = ((meta_cells["condition1"] == outSample) & (meta_cells["condition2"] == perturbation) & (adata.obs["iid_test"] == "test")).values
            meta_input1 = meta_cells.loc[input_idx,:]  ####
            Batchs = meta_input1['batch'].unique()
            counts_predict_list = []
            for Batch in Batchs:
                imputed = getPrediction(adata, model, counts_input, meta_input, outSample, perturbation, meta_input1, Batch, attrList1)
                if imputed:
                    counts_predict_list.append(imputed)

            if len(counts_predict_list) >=1:
                imputed = ad.concat(counts_predict_list)

                control_adata = ad.AnnData(counts_input, obs=meta_input, var=adata.var)
                control_adata.obs['perturbation'] = 'control'

                input_idx = ((meta_cells["condition1"] == outSample) & (meta_cells["condition2"] == perturbation) & (adata.obs["iid_test"] == "test")).values
                counts_input1 = counts[input_idx,:]
                meta_input1 = meta_cells.loc[input_idx,:]
                treat_adata = ad.AnnData(counts_input1, obs=meta_input1, var=adata.var)
                treat_adata.obs['perturbation'] = 'stimulated'

                result = ad.concat([control_adata, treat_adata, imputed])
                if not os.path.isdir(outSample): os.makedirs(outSample)
                result.write_h5ad('{}/{}_imputed.h5ad'.format(outSample, perturbation))
    
    pid= os.getpid()
    gpu_memory = pd.DataFrame(1, index=['gpu_memory'], columns=['gpu_memory'], dtype=str)
    devices = Device.all()
    for device_ in devices:
        processes = device_.processes()
        if pid in processes.keys():
            p=processes[pid]
            gpu_memory['gpu_memory'] = p.gpu_memory_human()
    gpu_memory.to_csv('gpu_memory.csv', sep='\t', index=False)
    torch.cuda.empty_cache()


