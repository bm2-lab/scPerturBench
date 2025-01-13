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

def writeYaml(basePath, hvg, DataLayer):
    filein = '/home/wzt/software/cellot/configs/tasks/inSample_task.yaml'
    with open(filein, 'r',encoding="utf-8") as fin:
        config = yaml.safe_load(fin)
        config['data']['path'] = '{}/../../../filter_hvg{}_{}.h5ad'.format(basePath, hvg, DataLayer)
        config['data']['source'] = 'control'
        config['data']['condition'] = 'condition2'

        config['datasplit']['groupby'] = ['condition1', 'condition2']
        config['datasplit']['test_size'] = 0.5 
    with open('inSample_task.yaml', 'w') as fout:
        yaml.safe_dump(config, fout)


def inSample(hvg, DataLayer, redo):
    basePath = '/home/wzt/project/Pertb_benchmark/DataSet/{}/inSample/hvg{}/cellot/'.format(DataSet, hvg)
    if not os.path.isdir(basePath): os.makedirs(basePath)
    os.chdir(basePath)

    filein = '/home/wzt/project/Pertb_benchmark/DataSet/{}/filter_hvg{}_{}.h5ad'.format(DataSet, hvg, DataLayer)
    adata = sc.read_h5ad(filein)
    perturbations = list(adata.obs['condition2'].unique())  #
    outSamples = list(adata.obs['condition1'].unique())
    
    if redo and os.path.basename(os.getcwd()) == 'cellot':  ##
        cmd = 'rm -r ./*'; subprocess.call(cmd, shell=True)

    for perturbation in tqdm(perturbations):
        if perturbation == 'control': continue
        writeYaml(basePath, hvg, DataLayer)
        cmd = "python /home/wzt/software/cellot/scripts/train.py --outdir model-scgen-{}"\
        " --config  inSample_task.yaml --config /home/wzt/software/cellot/configs/models/scgen.yaml"\
        " --config.data.target {}   ".format(perturbation, perturbation)
        subprocess.call(cmd, shell=True)

        cmd = "python /home/wzt/software/cellot/scripts/train.py --outdir model-cellot-{}"\
        " --config  inSample_task.yaml --config  /home/wzt/software/cellot/configs/models/cellot.yaml"\
        " --config.data.target {}  --config.data.ae_emb.path  model-scgen-{} ".format(perturbation, perturbation, perturbation)
        subprocess.call(cmd, shell=True)

        setting = 'iid'
        where = 'data_space'
        embedding = 'ae'
        outdir_cellot = '{}/model-cellot-{}'.format(basePath, perturbation)
        outdir_scgen = '{}/model-scgen-{}'.format(basePath, perturbation)
        expdir_cellot = Path(outdir_cellot)
        expdir_scgen = Path(outdir_scgen)
        config = load_config(expdir_cellot / 'config.yaml')
        config.data.ae_emb.path = 'model-scgen-{}'.format(perturbation)

        ### 进行预测
        control, treateddf, imputed = load_conditions(
                expdir_cellot, expdir_scgen, where, setting, embedding=embedding)
        adata_control = ad.AnnData(X=control.values, obs=adata.obs.loc[control.index, ], var=pd.DataFrame(index=control.columns))
        adata_control.obs['perturbation'] = 'control'
        adata_treat = ad.AnnData(X=treateddf.values, obs=adata.obs.loc[treateddf.index, ], var=pd.DataFrame(index=treateddf.columns))
        adata_treat.obs['perturbation'] = 'stimulated'
        imputed.obs['perturbation'] = 'imputed'
        result = ad.concat([adata_control, adata_treat, imputed])

        for outSample in outSamples:
            if not os.path.isdir(outSample): os.makedirs(outSample)
            tmp = result[result.obs['condition1'] == outSample]
            tmp.write_h5ad('{}/{}_imputed.h5ad'.format(outSample, perturbation))
