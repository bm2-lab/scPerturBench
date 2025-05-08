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


def getcondition(DataSet, seed = 1):
    filein = '/home//project/Pertb_benchmark/DataSet2/{}/hvg5000/GEARS/data/train/splits/train_simulation_{}_0.8.pkl'.format(DataSet, seed)
    with open(filein, 'rb') as fin:
        mydict = pickle.load(fin)
    train_conditions = mydict['train']
    test_conditions = mydict['test']
    return train_conditions, test_conditions

def clean_condition(condition):
    return condition.replace('+ctrl', '').replace('ctrl+', '').strip()


def getData(tmp_train, adata_train, embeddings):
    ytr = np.vstack([
        adata_train[adata_train.obs['perturbation'] == drug].X.mean(axis=0) 
        for drug in tmp_train
    ])
    tmp = np.vstack([embeddings[drug] for drug in embeddings]).mean(axis=0)
    Xtr = np.vstack([embeddings.get(drug, tmp) for drug in tmp_train])
    return Xtr, np.array(ytr)



def getData_comb(train_conditions, adata_train, embeddings):
    ytr = np.vstack([
        adata_train[adata_train.obs['perturbation'] == drug].X.mean(axis=0) 
        for drug in train_conditions
    ])


    tmp_mean = np.vstack([embeddings[drug] for drug in embeddings]).mean(axis=0)  ### 所有值的平均

    tmp = []
    for drug in train_conditions:
        if '+' in drug:
            a, b = drug.split('+')
            tmp_embedding = (np.array(embeddings.get(a, tmp_mean)) + np.array(embeddings.get(b, tmp_mean))) / 2  #### 求平均
        else:
            tmp_embedding = np.array(embeddings.get(drug, tmp_mean))
        tmp.append(tmp_embedding)
    Xtr = np.vstack(tmp)
    return Xtr, np.array(ytr)





class OneHiddenLayerMLP(nn.Module):
    def __init__(self, input_dim=5000, hidden_dim=256, output_dim=5000):
        super(OneHiddenLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 输入层 → 隐藏层
        self.fc2 = nn.Linear(hidden_dim, output_dim) # 隐藏层 → 输出层
        self.relu = nn.ReLU()

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

    dataset_path = "/home/project/Pertb_benchmark/DataSet2/{}/filter_hvg5000_logNor.h5ad".format(DataSet)
    embedding_path =  "/home/project/GW_PerturbSeq/geneEmbedding/scGPT.pkl"
    
    with open(embedding_path, "rb") as fp:
        embeddings = pickle.load(fp)
    adata = sc.read_h5ad(dataset_path)
    adata = adata[adata.obs['perturbation'] != 'control']
    train_conditions, test_conditions = getcondition(DataSet, seed)
    train_conditions = [clean_condition(i) for i in train_conditions]
    train_conditions = [i for i in train_conditions if i != 'ctrl']
    test_conditions = [clean_condition(i) for i in test_conditions]
    test_conditions = [i for i in test_conditions if i != 'ctrl']

    adata_train = adata[adata.obs["perturbation"].isin(train_conditions)]
    adata_test = adata[adata.obs["perturbation"].isin(test_conditions)]

    Xtr, ytr = getData(train_conditions, adata_train, embeddings)
    Xte, yte = getData(test_conditions, adata_test, embeddings)

    model = trainModel(Xtr, ytr)
    ypred = model(torch.tensor(Xte, dtype=torch.float).cuda())
    ypred = ypred.cpu().detach().numpy()

    result = pd.DataFrame(ypred, columns=adata.var_names, index= test_conditions)
    if not os.path.isdir(dirOut): os.makedirs(dirOut)
    result.to_csv(fileout,  sep='\t')




def doLinearModel_comb(DataSet, seed = 1):
    dirName = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/baseMLP'.format(DataSet)
    if not os.path.isdir(dirName): os.makedirs(dirName)
    os.chdir(dirName)
    dirOut = 'savedModels{}'.format(seed)
    fileout = '{}/pred.tsv'.format(dirOut)

    dataset_path = "/home/project/Pertb_benchmark/DataSet2/{}/filter_hvg5000_logNor.h5ad".format(DataSet)
    embedding_path =  "/home/project/GW_PerturbSeq/geneEmbedding/scGPT.pkl"

    with open(embedding_path, "rb") as fp:
        embeddings = pickle.load(fp)
    adata = sc.read_h5ad(dataset_path)
    adata = adata[adata.obs['perturbation'] != 'control']
    train_conditions, test_conditions = getcondition(DataSet, seed)
    train_conditions = [clean_condition(i) for i in train_conditions]
    train_conditions = [i for i in train_conditions if i != 'ctrl']
    test_conditions = [clean_condition(i) for i in test_conditions]
    test_conditions = [i for i in test_conditions if i != 'ctrl']

    adata_train = adata[adata.obs["perturbation"].isin(train_conditions)]
    adata_test = adata[adata.obs["perturbation"].isin(test_conditions)]

    Xtr, ytr = getData_comb(train_conditions, adata_train, embeddings)
    Xte, yte = getData_comb(test_conditions, adata_test, embeddings)

    model = trainModel(Xtr, ytr)
    ypred = model(torch.tensor(Xte, dtype=torch.float).cuda())
    ypred = ypred.cpu().detach().numpy()

    result = pd.DataFrame(ypred, columns=adata.var_names, index= test_conditions)
    if not os.path.isdir(dirOut): os.makedirs(dirOut)
    result.to_csv(fileout,  sep='\t')

def generateExp(cellNum, means, std):
    expression_matrix = np.array([
    np.random.normal(loc=means[i], scale=std[i], size=cellNum) 
    for i in range(len(means))]).T
    return expression_matrix

def generateH5ad(DataSet, seed = 1):
    dirName = '/home//project/Pertb_benchmark/DataSet2/{}/hvg5000/baseMLP'.format(DataSet)
    os.chdir(dirName)
    fileout = 'savedModels{}/result.h5ad'.format(seed)

    filein = 'savedModels{}/pred.tsv'.format(seed)
    exp = pd.read_csv(filein, sep='\t', index_col=0)
    filein = '../GEARS/savedModels{}/result.h5ad'.format(seed)
    adata = sc.read_h5ad(filein)
    expGene = np.intersect1d(adata.var_names, exp.columns)
    pertGenes = np.intersect1d(adata.obs['perturbation'].unique(), exp.index)
    adata = adata[:, expGene]; exp = exp.loc[:, expGene]

    control_exp = adata[adata.obs['perturbation'] == 'control'].to_df()
    control_std = list(np.std(control_exp))
    control_std = [i if not np.isnan(i) else 0 for i in control_std]
    for pertGene in pertGenes:
        cellNum = adata[(adata.obs['perturbation'] == pertGene) & (adata.obs['Expcategory']=='imputed')].shape[0]
        means = list(exp.loc[pertGene, ])
        expression_matrix = generateExp(cellNum, means, control_std)
        adata[(adata.obs['perturbation'] == pertGene) & (adata.obs['Expcategory']=='imputed')].X = expression_matrix
    adata.write(fileout)

### conda activate cpa
SinglePertDataSets = ['Adamson', "Frangieh", "TianActivation", "TianInhibition", "Replogle_exp7", "Replogle_exp8", "Papalexi", "Replogle_RPE1essential", "Replogle_K562essential"]
CombPertDataSets = ['Norman', 'Wessels', 'Schmidt', "Replogle_exp6", "Replogle_exp10"]

SinglePertDataSets = ["Papalexi"]
num_epochs = 200


if __name__ == '__main__':
    print ('hello, world')

#### single
    for DataSet in tqdm(SinglePertDataSets):
        seeds = [1, 2, 3]
        print (DataSet)
        for seed in tqdm(seeds):
            doLinearModel_single(DataSet, seed)
            #generateH5ad(DataSet, seed)


# ### comb
#     for DataSet in tqdm(CombPertDataSets):
#         seeds = [1, 2, 3]
#         print (DataSet)
#         for seed in tqdm(seeds):
#             doLinearModel_comb(DataSet, seed)
#             #generateH5ad(DataSet, seed)

