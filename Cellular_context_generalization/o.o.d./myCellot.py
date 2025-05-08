import yaml
import os, sys
sys.path.append('/home/project/Pertb_benchmark')
sys.path.append('/home/software/cellot')
from myUtil import *
from pathlib import Path
from cellot.utils.evaluate import load_conditions # type: ignore
from cellot.utils import load_config # type: ignore
import shutil
import torch
from scipy import sparse
import anndata as ad
import warnings
warnings.filterwarnings('ignore')


def writeYaml(basePath, outSample):
    filein = '/home/software/cellot/configs/tasks/task.yaml'
    with open(filein, 'r',encoding="utf-8") as fin:
        config = yaml.safe_load(fin)
        config['data']['path'] = '{}/../../../filter_hvg5000_logNor.h5ad'.format(basePath)
        config['data']['source'] = 'control'
        config['data']['condition'] = 'condition2'
        config['datasplit']['holdout'] = outSample
        config['datasplit']['key'] = "condition1"
        config['datasplit']['groupby'] = ['condition1', 'condition2']
    with open('task.yaml', 'w') as fout:
        yaml.safe_dump(config, fout)


def Kang_OutSample(DataSet, outSample):
    basePath = f'/home/project/Pertb_benchmark/DataSet/{DataSet}/outSample/hvg5000/cellot/'
    tmp = '{}/{}'.format(basePath, outSample)
    if not os.path.isdir(tmp): os.makedirs(tmp)
    os.chdir(tmp)

    filein = f'/home//project/Pertb_benchmark/DataSet/{DataSet}/filter_hvg5000_logNor.h5ad'
    adata  = sc.read_h5ad(filein)
    perturbations = list(adata.obs['condition2'].unique())
    perturbations = [i for i in perturbations if i != 'control']
    for perturbation in tqdm(perturbations):
        ncell = adata[(adata.obs['condition1'] == outSample) & (adata.obs['condition2'] == perturbation)]
        if ncell.shape[0] == 0: continue
        writeYaml(basePath, outSample)
        cmd = "python /home//software/cellot/scripts/train.py --outdir model-scgen-{}"\
        " --config  task.yaml --config /home//software/cellot/configs/models/scgen.yaml"\
        " --config.data.target {}   ".format(perturbation, perturbation)
        subprocess.call(cmd, shell=True)

        cmd = "python /home//software/cellot/scripts/train.py --outdir model-cellot-{}"\
        " --config  task.yaml --config  /home//software/cellot/configs/models/cellot.yaml"\
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


def KangMain():
    mylist = []
    filein_tmp = '/home/project/Pertb_benchmark/DataSet/{}/filter_hvg5000_logNor.h5ad'.format(DataSet)
    adata_tmp = sc.read_h5ad(filein_tmp)
    outSamples = list(adata_tmp.obs['condition1'].unique())
    for outSample in outSamples:
        Kang_OutSample(DataSet, outSample)

DataSet = 'kangCrossCell'

## conda activate  cellot


if __name__ == '__main__':
    KangMain()

