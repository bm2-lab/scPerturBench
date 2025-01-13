import yaml
import os, sys
sys.path.append('/home/wzt/project/Pertb_benchmark')
from myUtil import *
from pathlib import Path
from cellot.utils.evaluate import load_conditions # type: ignore
from cellot.utils import load_config # type: ignore
import shutil
import torch

from scipy import sparse
import anndata as ad
from nvitop import Device
import warnings
warnings.filterwarnings('ignore')

def writeYaml(basePath, outSample, hvg, DataLayer):
    filein = '/home/wzt/software/cellot/configs/tasks/task.yaml'
    with open(filein, 'r',encoding="utf-8") as fin:
        config = yaml.safe_load(fin)
        config['data']['path'] = '{}/../../../filter_hvg{}_{}.h5ad'.format(basePath, hvg, DataLayer)
        config['data']['source'] = 'control'
        config['data']['condition'] = 'condition2'
        config['datasplit']['holdout'] = outSample
        config['datasplit']['key'] = "condition1"
        config['datasplit']['groupby'] = ['condition1', 'condition2']
    with open('task.yaml', 'w') as fout:
        yaml.safe_dump(config, fout)


def Kang_OutSample(X):
    outSample, hvg, DataLayer, redo = X
    basePath = '/home/wzt/project/Pertb_benchmark/DataSet/{}/outSample/hvg{}/cellot/'.format(DataSet, hvg)
    tmp = '{}/{}'.format(basePath, outSample)
    if redo and os.path.isdir(tmp): shutil.rmtree(tmp)
    if not os.path.isdir(tmp): os.makedirs(tmp)
    os.chdir(tmp)

    filein = '/home/wzt/project/Pertb_benchmark/DataSet/{}/filter_hvg{}_{}.h5ad'.format(DataSet, hvg, DataLayer)
    adata  = sc.read_h5ad(filein)
    perturbations = list(adata.obs['condition2'].unique())  #
    perturbations = [i for i in perturbations if i != 'control']
    for perturbation in tqdm(perturbations):
        ncell = adata[(adata.obs['condition1'] == outSample) & (adata.obs['condition2'] == perturbation)]
        if ncell.shape[0] == 0: continue    ##
        if not redo and os.path.isfile('{}_imputed.h5ad'.format(perturbation)): continue
        writeYaml(basePath, outSample, hvg, DataLayer)
        cmd = "python /home/wzt/software/cellot/scripts/train.py --outdir model-scgen-{}"\
        " --config  task.yaml --config /home/wzt/software/cellot/configs/models/scgen.yaml"\
        " --config.data.target {}   ".format(perturbation, perturbation)
        subprocess.call(cmd, shell=True)

        cmd = "python /home/wzt/software/cellot/scripts/train.py --outdir model-cellot-{}"\
        " --config  task.yaml --config  /home/wzt/software/cellot/configs/models/cellot.yaml"\
        " --config.data.target {}  --config.data.ae_emb.path  model-scgen-{} ".format(perturbation, perturbation, perturbation)
        subprocess.call(cmd, shell=True)

        setting = 'ood'
        where = 'data_space'
        embedding = 'ae'
        outdir_cellot = '{}/{}/model-cellot-{}'.format(basePath, outSample, perturbation)
        outdir_scgen = '{}/{}/model-scgen-{}'.format(basePath, outSample, perturbation)
        expdir_cellot = Path(outdir_cellot)
        expdir_scgen = Path(outdir_scgen)
        config = load_config(expdir_cellot / 'config.yaml')
        config.data.ae_emb.path = 'model-scgen-{}'.format(perturbation)

        control, treateddf, imputed = load_conditions(
                expdir_cellot, expdir_scgen, where, setting, embedding=embedding)
        adata_control = ad.AnnData(X=control.values, obs=pd.DataFrame(index=control.index), var=pd.DataFrame(index=control.columns))
        adata_control.obs['perturbation'] = 'control'
        adata_treat = ad.AnnData(X=treateddf.values, obs=pd.DataFrame(index=treateddf.index), var=pd.DataFrame(index=treateddf.columns))
        adata_treat.obs['perturbation'] = 'stimulated'
        imputed.obs['perturbation'] = 'imputed'
        result = ad.concat([adata_control, adata_treat, imputed])
        result.write_h5ad('{}/../{}_imputed.h5ad'.format(outdir_cellot, perturbation))

    pid= os.getpid()
    gpu_memory = pd.DataFrame(1, index=['gpu_memory'], columns=['gpu_memory'], dtype=str)
    devices = Device.all()
    for device in devices:
        processes = device.processes()
        if pid in processes.keys():
            p=processes[pid]
            gpu_memory['gpu_memory'] = p.gpu_memory_human()
    gpu_memory.to_csv('gpu_memory.csv', sep='\t', index=False)






DataSet = 'TCDDJointFactor'

