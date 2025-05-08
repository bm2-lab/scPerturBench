import os, sys
sys.path.append('/home/project/Pertb_benchmark')
from myUtil import *
import anndata as ad
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

def generatePairedSample(adata, outSample, perturbation):
    cellTypes = list(adata.obs['condition1'].unique())
    cellTypes = [i for i in cellTypes if i != outSample]
    annList_Pertb = []; annList_control = []
    for celltype in cellTypes:
        perturb_cells = adata[ (adata.obs['condition1'] == celltype) & (adata.obs['condition2'] == perturbation)]
        control_cells = adata[ (adata.obs['condition1'] == celltype) & (adata.obs['condition2'] ==  'control')]
        Nums = min(perturb_cells.shape[0], control_cells.shape[0])
        perturb_cells = perturb_cells[:Nums]
        control_cells = control_cells[:Nums]
        annList_Pertb.append(perturb_cells)
        annList_control.append(control_cells)
    annList_Pertb = ad.concat(annList_Pertb)
    annList_control = ad.concat(annList_control)
    return annList_Pertb, annList_control


# 1. 定义单隐藏层MLP模型
class OneHiddenLayerMLP(nn.Module):
    def __init__(self, input_dim=5000, hidden_dim=1024, output_dim=5000):
        super(OneHiddenLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 输入层 → 隐藏层
        self.fc2 = nn.Linear(hidden_dim, output_dim) # 隐藏层 → 输出层
        self.relu = nn.ReLU()  # 激活函数

    def forward(self, x):
        x = self.fc2(self.fc1(x))
        x = self.relu(x)
        return x


def trainModel(Xtr, ytr, perturbation, batch_size=128):
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


    # 4.
    model = OneHiddenLayerMLP(input_dim=Xtr.shape[1], output_dim=ytr.shape[1], hidden_dim=1024).cuda()
    criterion = nn.MSELoss()              # MSE loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:

            optimizer.zero_grad()
            outputs = model(inputs.cuda())
            loss = criterion(outputs, targets.cuda())

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)


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
            torch.save(model.state_dict(), '{}_best_model.pth'.format(perturbation))  # save best model
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping triggered at epoch {epoch+1}!')
                break
    
    model.load_state_dict(torch.load('{}_best_model.pth'.format(perturbation)))
    print("Training completed with best validation loss:", best_val_loss)
    return model


def Kang_OutSample(DataSet, outSample):
    basePath = f'/home//project/Pertb_benchmark/DataSet/{DataSet}/outSample/hvg5000/baseMLP/'
    tmp = '{}/{}'.format(basePath, outSample)
    if not os.path.isdir(tmp): os.makedirs(tmp)
    os.chdir(tmp)
    
    path = f'/home//project/Pertb_benchmark/DataSet/{DataSet}/filter_hvg5000_logNor.h5ad'
    tmp = sc.read_h5ad(path)
    perturbations = list(tmp.obs['condition2'].unique())
    perturbations = [i for i in perturbations if i != 'control']
    for perturbation in perturbations:
        adata = tmp[tmp.obs['condition2'].isin([perturbation, 'control'])]
        Xtr_anndata, ytr_anndata = generatePairedSample(adata, outSample, perturbation)
        Xtr = Xtr_anndata.X;  ytr  = ytr_anndata.X
        Xte_anndata = adata[ (adata.obs['condition2'].isin(['control'])) & (adata.obs['condition1'].isin([outSample]))]
        yte_anndata = adata[ (adata.obs['condition2'].isin([perturbation])) & (adata.obs['condition1'].isin([outSample]))]
        Xte = Xte_anndata.X
        model = trainModel(Xtr, ytr, perturbation, batch_size=128)
        ypred = model(torch.tensor(Xte, dtype=torch.float).cuda())
        ypred = ypred.cpu().detach().numpy()

        Xte_anndata.obs['perturbation'] = 'control'

        imputed = Xte_anndata.copy()
        imputed.X = ypred
        imputed.obs['perturbation'] = 'imputed'

        yte_anndata.obs['perturbation'] = 'stimulated'

        result = ad.concat([Xte_anndata, yte_anndata, imputed])
        result.write_h5ad('{}_imputed.h5ad'.format(perturbation))

 
def KangMain(DataSet):
    filein_tmp = '/home//project/Pertb_benchmark/DataSet/{}/filter_hvg5000_logNor.h5ad'.format(DataSet)
    adata_tmp = sc.read_h5ad(filein_tmp)
    outSamples = list(adata_tmp.obs['condition1'].unique())
    for outSample in outSamples:
        Kang_OutSample(DataSet, outSample)


DataSets = ["kangCrossCell", "kangCrossPatient",  "Parekh", "Haber", "crossPatient", "KaggleCrossPatient",
            "KaggleCrossCell", "crossSpecies", "McFarland", "Afriat", "TCDD", "sciplex3"]
DataSets = ["kangCrossCell"]

torch.cuda.set_device('cuda:0')

## conda activate  cpa

if __name__ == '__main__':
    for DataSet in tqdm(DataSets):
        KangMain(DataSet)