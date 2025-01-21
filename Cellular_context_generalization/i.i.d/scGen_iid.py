import os, sys
sys.path.append('/home/wzt/project/Pertb_benchmark')
from myUtil import *
from nvitop import Device
import yaml
import scanpy as sc
import scgen
import anndata as ad
import torch
print('p')
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

### kang data out sample test 
def inSample(X):
    outSample, hvg, DataLayer, redo = X   #hvg,datalayer,outsample celltype
    basePath = '/home/wzt/project/Pertb_benchmark/DataSet/{}/inSample/hvg{}/scGen/'.format(DataSet, hvg) #### 0000
    outSample_test = outSample + '_test' ### 11111
    print(outSample)
    tmp = '{}/{}'.format(basePath, outSample)
    if not os.path.isdir(tmp): os.makedirs(tmp)
    os.chdir(tmp)
    print(tmp)
    #if not redo and os.path.isfile('imputed.h5ad'): return
    #kang_preData(hvg, DataLayer)
    path = '/home/wzt/project/Pertb_benchmark/DataSet/{}/filter_hvg{}_{}.h5ad'.format(DataSet, hvg, DataLayer)
    adata0 = sc.read_h5ad(path)
    adata0.obs['condition_iid'] = adata0.obs['condition1'].astype(str) + '_' + adata0.obs['iid_test'].astype(str)  ### 22222
    
    
    perturbations = list(adata0.obs['condition2'].unique())  ### perturbations
    filtered_per = list(filter(lambda x: x != 'control', perturbations))
    
    for perturbation in filtered_per:
        adata = adata0[(adata0.obs['condition2'] == perturbation) | (adata0.obs['condition2'] == 'control')]
            
        adata_train = adata[~((adata.obs["condition_iid"] == outSample_test) & (adata.obs["condition2"] == perturbation))].copy()
    
        print(perturbation)
        print(outSample)
        
        scgen.SCGEN.setup_anndata(adata_train, batch_key="condition2", labels_key="condition_iid")
        model = scgen.SCGEN(adata_train)
        print('model start')
        

        model.train(max_epochs=200, batch_size=128, early_stopping=True, early_stopping_patience=25)
        print('model done')
    
        pred, delta = model.predict(ctrl_key='control', stim_key=perturbation, celltype_to_predict=outSample_test)
    
        pred.obs['condition2'] = 'imputed'
    
        ctrl_adata = adata[((adata.obs['condition_iid'] == outSample_test) & (adata.obs['condition2'] == 'control'))]
        stim_adata = adata[((adata.obs['condition_iid'] == outSample_test) & (adata.obs['condition2'] == perturbation))]
    
        eval_adata = ctrl_adata.concatenate(stim_adata, pred) 
        eval_adata.obs=eval_adata.obs[['condition2']]
        eval_adata.obs = eval_adata.obs.rename(columns={'condition2':'perturbation'})
        eval_adata.obs['perturbation'] = eval_adata.obs['perturbation'].replace([perturbation], 'stimulated')
        del eval_adata.var
    
        eval_adata.write_h5ad('{}_imputed.h5ad'.format(perturbation))
        #sc.tl.pca(eval_adata)
        #sc.pl.pca(eval_adata, color=["perturbation"], frameon=False, save = "_{}.pdf".format(perturbation), show=False)
    
    pid= os.getpid()
    print(pid)
    gpu_memory = pd.DataFrame(1, index=['gpu_memory'], columns=['gpu_memory'], dtype=str)
    devices = Device.all()
    for device in devices:
        processes = device.processes()
        if pid in processes.keys():
            p=processes[pid]
            gpu_memory['gpu_memory'] = p.gpu_memory_human()
    gpu_memory.to_csv('gpu_memory.csv', sep='\t', index=False)
            




def Main():
    mylist = []
    filein_tmp = '/home/wzt/project/Pertb_benchmark/DataSet/{}/filter_hvg5000_logNor.h5ad'.format(DataSet)
    adata_tmp = sc.read_h5ad(filein_tmp)
    outSamples = list(adata_tmp.obs['condition1'].unique())
    #outSamples = ['endothelial', 'hPSC']
    for hvg in [5000]:
        for outSample in outSamples:
            mylist.append([outSample, hvg, 'logNor', False])
    myPool(inSample, mylist, processes=1)


DataSet = 'Afriat'
print(DataSet)


if __name__ == '__main__':
    Main() 