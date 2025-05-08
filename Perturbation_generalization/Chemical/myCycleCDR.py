import sys, subprocess, os
sys.path.append('/home/project/Pertb_benchmark')
from myUtil1 import *
from scipy import sparse
import anndata as ad
import anndata
import pickle
from collections import defaultdict, OrderedDict

import yaml
import torch
import random
import itertools

import torch.optim as optim
from torch.optim import lr_scheduler
from torch_geometric.loader import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler


sys.path.insert(0, '/home/software/cycleCDR')

from cycleCDR.model import cycleCDR  ### type: ignore
from cycleCDR.utils import Trainer    ### type: ignore
from cycleCDR.dataset import load_dataset_splits    ### type: ignore
from cycleCDR.dataset import load_dataset_splits_for_gat    ### type: ignore

def parse_config():
    config_path = '/home/software/cycleCDR/configs/train_sciplex3_gat_row_optuna_fixed.yaml'

    with open(config_path, 'rb') as f:
        data = yaml.safe_load_all(f)
        data = list(data)[0]

    return data


def get_lr_lambda(config):
    num_epoch = config["num_epoch"]
    lr = config["lr"]
    lambda_lr_A = config["lambda_lr_A"]
    lambda_lr_B = config["lambda_lr_B"]

    return lambda epoch: lr * (
        (num_epoch * lambda_lr_A - epoch) / (num_epoch * lambda_lr_B)
    )


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def getPredResults(trainer, adata, seed):
    results = trainer.test_res['pred_treats_dict']
    trainMean_adata = sc.read_h5ad('../trainMean/savedModels{}/result.h5ad'.format(seed))
    mylist = []
    for pred in results:
        tmp = results[pred]
        tmp = [np.array(i.detach()) for i in tmp]
        df = pd.DataFrame({"cov_drug_dose_name": [pred] * len(tmp), "Expcategory":["imputed"] * len(tmp)})
        tmp_adata = ad.AnnData(pd.DataFrame(tmp), var = adata.var, obs=df)
        mylist.append(tmp_adata)
    adata_results = ad.concat(mylist)  #### 最后的结果
    NoImplemented = trainMean_adata[trainMean_adata.obs['Expcategory'] != 'imputed']
    adata_results = ad.concat([adata_results, NoImplemented])
    adata_results.write('savedModels{}/result.h5ad'.format(seed))

def trainModel(DataSet, seed):
    dirName = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/cycleCDR'.format(DataSet)
    os.chdir(dirName)
    adata = sc.read_h5ad('../../filter_hvg5000_logNor.h5ad')
    config = parse_config()
    
    config['append_layer_width'] = adata.shape[1]
    config['lambda_disc_A'] = 1
    config['lambda_disc_B'] = 1
    config['NumFeatures'] = adata.shape[1]

    config['sciplex3_treat_train'] = 'savedModels{}/chemcpa_trapnell_treat_train.csv'.format(seed)
    config['sciplex3_treat_valid'] = 'savedModels{}/chemcpa_trapnell_treat_valid.csv'.format(seed)
    config['sciplex3_treat_test'] = 'savedModels{}/chemcpa_trapnell_treat_test.csv'.format(seed)
    config['sciplex3_control_train'] = 'savedModels{}/chemcpa_trapnell_control_train.csv'.format(seed)
    config['sciplex3_control_valid'] = 'savedModels{}/chemcpa_trapnell_control_valid.csv'.format(seed)
    config['sciplex3_control_test'] = 'savedModels{}/chemcpa_trapnell_control_test.csv'.format(seed)
    config['sciplex3_de_gene'] = 'savedModels{}/chemcpa_deg_gene.csv'.format(seed)
    config['sciplex3_mse_de_gene'] = 'savedModels{}/chemcpa_mse_hvg_idx.pkl'.format(seed)

    config['sciplex3_drug'] = '/home/software/cycleCDR/datasets/preprocess/sciplex3/rdkit2D_embedding.parquet'
    config['sciplex3_row_drug'] = '/home/software/cycleCDR/datasets/preprocess/sciplex3/rdkit2D_embedding.parquet'
    config['sciplex3_processed_drug'] = '/home/software/cycleCDR/datasets/preprocess/sciplex3/processed_drug_smiles.pt'


    set_seed(config["seed"])
    model = cycleCDR(config).cuda()  ### 模型

    optimizer_D = optim.Adam(
        itertools.chain(
            model.discriminator_A.parameters(), model.discriminator_B.parameters()
        ),
        lr=config["d_lr"],
        weight_decay=float(config["d_weight_decay"]),
    )
    # lr 由 scheduler 调整, 初始 lr 也是在 scheduler 中设置, 这里的 lr 是一个无用参数
    optimizer_G = optim.Adam(
        itertools.chain(
            model.encoderG_A.parameters(),
            model.encoderG_B.parameters(),
            model.decoderG_A.parameters(),
            model.decoderG_B.parameters(),
        ),
        lr=config["g_lr"],
        weight_decay=float(config["g_weight_decay"]),
    )

    optimizer_DRUG = optim.Adam(
        model.drug_encoder.parameters(),
        lr=config["drug_lr"],
        weight_decay=float(config["drug_weight_decay"]),
    )

    scheduler_G = lr_scheduler.StepLR(
        optimizer_G, step_size=config["g_step_size"], gamma=config["g_gamma"]
    )

    scheduler_D = lr_scheduler.StepLR(
        optimizer_D, step_size=config["d_step_size"], gamma=config["d_gamma"]
    )

    scheduler_DRUG = lr_scheduler.StepLR(
        optimizer_DRUG, step_size=config["drug_step_size"], gamma=config["drug_gamma"]
    )

    datasets = load_dataset_splits_for_gat(config)

    train_dataloader = DataLoader(
        dataset=datasets["train"],
        batch_size=config["batch_size"],
        shuffle=False)
    valid_dataloader = DataLoader(
        dataset=datasets["valid"],
        batch_size=config["batch_size"],
        shuffle=False)
    test_dataloader = DataLoader(
        dataset=datasets["test"],
        batch_size=config["batch_size"],
        shuffle=False)

    trainer = Trainer(
        model,
        config["num_epoch"],
        optimizer_G,
        optimizer_D,
        scheduler_G,
        scheduler_D,
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        optimizer_DRUG=optimizer_DRUG,
        scheduler_DRUG=scheduler_DRUG,
        dataset_name=config["dataset"],
        is_mse=config["is_mse"],
        is_gan=config["is_gan"],
        config=config,
    )

    trainer.fit()
    getPredResults(trainer, adata, seed)


torch.cuda.set_device('cuda:1')

SingleDataSets = ['sciplex3_A549', 'sciplex3_MCF7',  'sciplex3_K562']

seeds  = [1, 2, 3]


if __name__ == '__main__':
    trainModel('sciplex3_K562', seed = 1)
    trainModel('sciplex3_K562', seed = 2)
    trainModel('sciplex3_MCF7', seed = 3)
    
