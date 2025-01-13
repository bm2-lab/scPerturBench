import sys, re
sys.path.append('/home/wzt/project/Pertb_benchmark')
sys.path.append('/home/wzt/software/AttentionPert')
from myUtil import *
import torch
import pickle
from tqdm import tqdm
from collections import OrderedDict
from itertools import chain
from attnpert import PertData #type: ignore
from attnpert.attnpert import ATTNPERT_RECORD_TRAIN #type: ignore
from attnpert.model import *  #type: ignore
from attnpert.utils import print_sys  #type: ignore
from attnpert.inference import evaluate  #type: ignore

def remove_duplicates_and_preserve_order(input_list):
    # 使用OrderedDict来创建一个有序的字典，其中重复的元素将被覆盖
    # 字典的键将是唯一的元素，值将保持原始顺序
    deduplicated_dict = OrderedDict.fromkeys(input_list)
    # 从有序字典中提取唯一元素，以维护原始顺序
    deduplicated_list = list(deduplicated_dict.keys())
    return deduplicated_list

def runAttentionPert(DataSet, seed = 1, isPredict = False):
    dirName = '/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/{}/hvg5000/AttentionPert'.format(DataSet)
    if not os.path.isdir(dirName): os.makedirs(dirName)
    os.chdir(dirName)
    modeloutPT = 'savedModels{}/model.pt'.format(seed)
    if os.path.isfile(modeloutPT) and not isPredict: return
    if not isPredict:
        if not os.path.isdir('data'): os.makedirs('data')
        if not os.path.isdir('data/train'): os.makedirs('data/train')
        if not os.path.isdir('data/train/splits'): os.makedirs('data/train/splits')
        cmd = 'cp /home/wzt/software/AttentionPert/gene2vec_dim_200_iter_9_w2v.txt data'; subprocess.call(cmd, shell=True)
        cmd = 'cp /home/wzt/software/AttentionPert/gene2go_all.pkl data'; subprocess.call(cmd, shell=True)
        cmd = 'cp ../GEARS/data/train/perturb_processed.h5ad  data/train/'; subprocess.call(cmd, shell=True)
        cmd = 'python /home/wzt/software/AttentionPert/gene2vec_example.py'; subprocess.call(cmd, shell=True)

    MODEL_CLASS = PL_PW_non_add_Model  #type: ignore
    EPOCHS = 20;  BATCH_SIZE = 128; VALID_EVERY = 1
    gene2vec_file_path = 'data/train/gene2vec.npy'
    record_pred = True; ACT = 'softmax'; beta  = 5e-2
    default_setting = {"gene2vec_args": {"gene2vec_file": gene2vec_file_path}, 
                        "pert_local_min_weight": 0.75, 
                        "pert_local_conv_K": 1,
                        "pert_weight_heads": 2,
                        "pert_weight_head_dim": 64,
                        "pert_weight_act": ACT,
                        "non_add_beta": beta,
                        'record_pred': record_pred}

    pert_data = PertData('./data')
    pert_data.load(data_path = './data/train')
    cmd = 'cp ../GEARS/data/train/splits/train_simulation_{}_0.8.pkl   data/train/splits/'.format(seed)  ### 使用GEARS的分割以保持一致
    subprocess.call(cmd, shell=True)
    cmd = 'cp ../GEARS/data/train/splits/train_simulation_{}_0.8_subgroup.pkl   data/train/splits/'.format(seed)
    subprocess.call(cmd, shell=True)
    pert_data.prepare_split(split = 'simulation', seed = seed, train_gene_set_size=.8) # get data split with seed
    pert_data.get_dataloader(batch_size = BATCH_SIZE, test_batch_size = BATCH_SIZE)
    
    wandb = False
    exp_name = f'seed{seed}'
    print_sys("EXPERIMENT: " + exp_name)
    attnpert_model = ATTNPERT_RECORD_TRAIN(pert_data, device = device,  
                    weight_bias_track=wandb,
                    proj_name = 'attnpert',
                    exp_name = exp_name)
    attnpert_model.model_initialize(hidden_size = 64, 
                                model_class=MODEL_CLASS,
                                exp_name = exp_name, 
                                **default_setting)

    if isPredict: return pert_data, attnpert_model
    print_sys(attnpert_model.config)
    attnpert_model.train(epochs = EPOCHS, valid_every= VALID_EVERY)  ### 训练模型
    if not os.path.isdir('savedModels{}'.format(seed)): os.makedirs('savedModels{}'.format(seed))
    torch.save(attnpert_model.best_model.state_dict(), modeloutPT)
    

def predict(DataSet, seed = 1, isPredict = True):
    dirName = '/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/{}/hvg5000/AttentionPert'.format(DataSet)
    os.chdir(dirName)
    filein = 'savedModels{}/model.pt'.format(seed)
    fileout = 'savedModels{}/result.h5ad'.format(seed)
    if os.path.isfile(fileout): return  ### 已经预测了，不需要重新跑
    pert_data, attnpert_model = runAttentionPert(DataSet, seed, isPredict)
    model = attnpert_model.best_model
    adata = pert_data.adata
    model.load_state_dict(torch.load(filein))
    test_res = evaluate(pert_data.dataloader['test_loader'], model, False, device, real_num_genes = None)
    pert_cats = remove_duplicates_and_preserve_order(test_res['pert_cat'])
    adata_list = [adata[adata.obs['condition'] == pert_cat].copy()  for pert_cat in pert_cats ]
    adata1 = ad.concat(adata_list)
    adata1.X = test_res['pred']   ### 预测值
    adata1.obs['Expcategory'] = 'imputed'

    adata_ctrl = attnpert_model.ctrl_adata
    adata_ctrl.obs['Expcategory'] = 'control'

    adata_list = [adata[adata.obs['condition'] == pert_cat].copy()  for pert_cat in pert_cats]
    adata2 = ad.concat(adata_list)
    adata2.obs['Expcategory'] = 'stimulated'
    adata_fi = ad.concat([adata_ctrl, adata1, adata2])
    adata_fi.write_h5ad(fileout)


### conda activate gears
seeds = [1, 2, 3]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DataSet = 'TianActivation'

SinglePertDataSets = ['Adamson', "Frangieh", "TianActivation", "TianInhibition", "Replogle_exp7", "Replogle_exp8", "Papalexi", "Replogle_RPE1essential", "Replogle_K562essential"]
CombPertDataSets = ['Norman', 'Wessels', 'Schmidt', "Replogle_exp6", "Replogle_exp10"]


