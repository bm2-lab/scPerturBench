import sys, os
sys.path.append('/home/project/Pertb_benchmark')
from myUtil import *
from collections import OrderedDict
from itertools import chain
import pickle
import torch.nn as nn
import numpy as np
from sklearn.linear_model import Ridge
from tqdm import tqdm
import anndata as ad
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


def clean_condition(condition):
    return condition.replace('+ctrl', '').replace('ctrl+', '').strip()



def getData(tmp_train, adata_train, embeddings):
    ytr = np.vstack([
        adata_train[adata_train.obs['cov_drug_dose_name'] == drug].X.mean(axis=0) 
        for drug in tmp_train
    ])

    Xtr = np.vstack([embeddings[drug] for drug in tmp_train])
    return Xtr, np.array(ytr)


# 1. 定义单隐藏层MLP模型
class OneHiddenLayerMLP(nn.Module):
    def __init__(self, input_dim=5000, hidden_dim=256, output_dim=5000):
        super(OneHiddenLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 输入层 → 隐藏层
        self.fc2 = nn.Linear(hidden_dim, output_dim) # 隐藏层 → 输出层
        self.relu = nn.ReLU()  # 激活函数

    def forward(self, x):
        x = self.fc2(self.fc1(x))
        x = self.relu(x)
        return x


def trainModel(Xtr, ytr, batch_size=128, hidden_dim = 1024):
    train_size = int(0.8 * Xtr.shape[0])
    val_size = Xtr.shape[0] - train_size
    
    patience = 20      # 容忍的验证损失不下降的epoch数
    min_delta = 0.0001  # 认为显著下降的最小变化
    best_val_loss = np.inf
    counter = 0        # 记录未改善的epoch数

    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    ytr = torch.tensor(ytr, dtype=torch.float32)
    dataset = TensorDataset(Xtr, ytr)

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


    # 4. 初始化模型、损失函数和优化器
    model = OneHiddenLayerMLP(input_dim=Xtr.shape[1], output_dim=ytr.shape[1], hidden_dim=hidden_dim).cuda()
    criterion = nn.MSELoss()              # MSE损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs.cuda())
            loss = criterion(outputs, targets.cuda())
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

    ### 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs.cuda())
                val_loss += criterion(outputs, targets.cuda()).item()
        avg_val_loss = val_loss / len(val_loader)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')  # 保存最佳模型
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping triggered at epoch {epoch+1}!')
                break
    
    model.load_state_dict(torch.load('best_model.pth'))
    print("Training completed with best validation loss:", best_val_loss)
    return model



def doLinearModel_single(DataSet, seed = 1):
    dirName = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/baseMLP'.format(DataSet)
    if not os.path.isdir(dirName): os.makedirs(dirName)
    os.chdir(dirName)
    dirOut = 'savedModels{}'.format(seed)
    fileout = '{}/pred.tsv'.format(dirOut)
    if os.path.isfile(fileout): return
    dataset_path = "/home/project/Pertb_benchmark/DataSet2/{}/filter_hvg5000_logNor.h5ad".format(DataSet)
    embedding_path =  "/home/project/Pertb_benchmark/DataSet2/{}/chemicalEmbedding.pkl".format(DataSet)
    with open(embedding_path, "rb") as fp:
        embeddings = pickle.load(fp)
    adata = sc.read_h5ad(dataset_path)
    adata = adata[adata.obs['condition'] != 'control']

    tmp = 'split_ood_multi_task{}'.format(seed)
    adata_train = adata[adata.obs[tmp] == 'train']
    adata_test = adata[adata.obs[tmp] == 'ood']   ### 单个扰动用 ood

    tmp_train = list(adata_train.obs['cov_drug_dose_name'].unique())  #### 训练的药物
    tmp_test = list(adata_test.obs['cov_drug_dose_name'].unique())   #### 测试的药物
    Xtr, ytr = getData(tmp_train, adata_train, embeddings)
    Xte, yte = getData(tmp_test, adata_test, embeddings)

    model = trainModel(Xtr, ytr)
    ypred = model(torch.tensor(Xte, dtype=torch.float).cuda())
    ypred = ypred.cpu().detach().numpy()

    result = pd.DataFrame(ypred, columns=adata.var_names, index= tmp_test)

    if not os.path.isdir(dirOut): os.makedirs(dirOut)
    result.to_csv(fileout, sep='\t')




def doLinearModel_comb(DataSet, seed = 1):
    dirName = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/baseMLP'.format(DataSet)
    if not os.path.isdir(dirName): os.makedirs(dirName)
    os.chdir(dirName)
    dirOut = 'savedModels{}'.format(seed)
    fileout = '{}/pred.tsv'.format(dirOut)
    if os.path.isfile(fileout): return
    dataset_path = "/home/project/Pertb_benchmark/DataSet2/{}/filter_hvg5000_logNor.h5ad".format(DataSet)
    embedding_path =  "/home/project/Pertb_benchmark/DataSet2/{}/chemicalEmbedding.pkl".format(DataSet)
    with open(embedding_path, "rb") as fp:
        embeddings = pickle.load(fp)
    adata = sc.read_h5ad(dataset_path)
    adata = adata[adata.obs['condition'] != 'control']

    tmp = 'split_ood_multi_task{}'.format(seed)
    adata_train = adata[adata.obs[tmp] == 'train']
    adata_test = adata[adata.obs[tmp] == 'test']   ### 组合扰动用test作为ood

    tmp_train = list(adata_train.obs['cov_drug_dose_name'].unique())  #### 训练的药物
    tmp_test = list(adata_test.obs['cov_drug_dose_name'].unique())   #### 测试的药物
    Xtr, ytr = getData(tmp_train, adata_train, embeddings)
    Xte, yte = getData(tmp_test, adata_test, embeddings)

    model = trainModel(Xtr, ytr)
    ypred = model(torch.tensor(Xte, dtype=torch.float).cuda())
    ypred = ypred.cpu().detach().numpy()
    
    result = pd.DataFrame(ypred, columns=adata.var_names, index= tmp_test)

    if not os.path.isdir(dirOut): os.makedirs(dirOut)
    result.to_csv(fileout,  sep='\t')


def generateExp(cellNum, means, std):
    expression_matrix = np.array([
    np.random.normal(loc=means[i], scale=std[i], size=cellNum) 
    for i in range(len(means))]).T
    return expression_matrix


def generateH5ad(DataSet, seed = 1, senario='trainMean'):
    import anndata
    dirName = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/{}'.format(DataSet, senario)
    os.chdir(dirName)
    filein = 'savedModels{}/pred.tsv'.format(seed)
    exp = pd.read_csv(filein, sep='\t', index_col=0)
    filein = '../../filter_hvg5000.h5ad'
    adata = sc.read_h5ad(filein)
    expGene = np.intersect1d(adata.var_names, exp.columns)
    pertGenes = np.intersect1d(adata.obs['cov_drug_dose_name'].unique(), exp.index)
    adata = adata[:, expGene]; exp = exp.loc[:, expGene]

    control_exp = adata[adata.obs['perturbation'] == 'control'].to_df()
    control_std = list(np.std(control_exp))
    control_std = [i if not np.isnan(i) else 0 for i in control_std]

    pred_list = []
    for pertGene in tqdm(pertGenes):
        cellNum = adata[adata.obs['cov_drug_dose_name'] == pertGene].shape[0]
        means = list(exp.loc[pertGene, ])
        expression_matrix = generateExp(cellNum, means, control_std)
        pred = anndata.AnnData(expression_matrix, var=adata.var)
        pred.obs['cov_drug_dose_name'] = pertGene
        pred_list.append(pred)
    pred_results = anndata.concat(pred_list)
    pred_results.obs['Expcategory'] = 'imputed'

    control_adata = adata[adata.obs['perturbation'] == 'control'].copy()
    control_adata.obs['Expcategory'] = 'control'

    stimulated_adata = adata[adata.obs['cov_drug_dose_name'].isin(pertGenes)]
    stimulated_adata.obs['Expcategory'] = 'stimulated'
    pred_fi = anndata.concat([pred_results, control_adata, stimulated_adata])
    pred_fi.write('savedModels{}/result.h5ad'.format(seed))



### conda activate cpa
SinglePertDataSets = ['sciplex3_MCF7', "sciplex3_A549", "sciplex3_K562"]
CombPertDataSets = ['sciplex3_comb']
num_epochs = 200


if __name__ == '__main__':
    print ('hello, world')

#### single
    for DataSet in tqdm(['sciplex3_A549']):
        seeds = [1, 2, 3]
        print (DataSet)
        for seed in tqdm(seeds):
            doLinearModel_single(DataSet, seed)
            #generateH5ad(DataSet, seed, senario='baseMLP')


### comb
    # for DataSet in tqdm(CombPertDataSets):
    #     seeds = [1, 2, 3]
    #     print (DataSet)
    #     for seed in tqdm(seeds):
    #         doLinearModel_comb(DataSet, seed)
    #         generateH5ad(DataSet, seed, senario='baseMLP')

