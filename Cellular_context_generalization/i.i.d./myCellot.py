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

def writeYaml(basePath):
    filein = '/home/software/cellot/configs/tasks/inSample_task.yaml'
    with open(filein, 'r',encoding="utf-8") as fin:
        config = yaml.safe_load(fin)
        config['data']['path'] = f'{basePath}/../../../filter_hvg5000_logNor.h5ad'
        config['data']['source'] = 'control'
        config['data']['condition'] = 'condition2'

        config['datasplit']['groupby'] = ['condition1', 'condition2']
        config['datasplit']['test_size'] = 0.5 
    with open('inSample_task.yaml', 'w') as fout:
        yaml.safe_dump(config, fout)


def Kang_inSample(DataSet):
    basePath = f'/home/project/Pertb_benchmark/DataSet/{DataSet}/inSample/hvg5000/cellot/'
    if not os.path.isdir(basePath): os.makedirs(basePath)
    os.chdir(basePath)

    filein = f'/home/project/Pertb_benchmark/DataSet/{DataSet}/filter_hvg5000_logNor.h5ad'
    adata = sc.read_h5ad(filein)
    perturbations = list(adata.obs['condition2'].unique())
    outSamples = list(adata.obs['condition1'].unique())    
    for perturbation in tqdm(perturbations):
        if perturbation == 'control': continue
        writeYaml(basePath)
        cmd = "python /home/software/cellot/scripts/train.py --outdir model-scgen-{}"\
        " --config  inSample_task.yaml --config /home/software/cellot/configs/models/scgen.yaml"\
        " --config.data.target {}   ".format(perturbation, perturbation)
        subprocess.call(cmd, shell=True)

        cmd = "python /home/software/cellot/scripts/train.py --outdir model-cellot-{}"\
        " --config  inSample_task.yaml --config  /home/software/cellot/configs/models/cellot.yaml"\
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


DataSet = 'kangCrossCell'
if __name__ == '__main__':
    Kang_inSample(DataSet)