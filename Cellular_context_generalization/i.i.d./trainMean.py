import os, sys
sys.path.append('/home/project/Pertb_benchmark')
from myUtil import *
import anndata as ad
import torch
import warnings
warnings.filterwarnings('ignore')

def generateExp(cellNum, means, std):
    expression_matrix = np.array([
    np.random.normal(loc=means[i], scale=std[i], size=cellNum)
    for i in range(len(means))]).T
    return expression_matrix


def Kang_OutSample(DataSet, outSample):
    basePath = f'/home/project/Pertb_benchmark/DataSet/{DataSet}/inSample/hvg5000/trainMean/'
    tmp = '{}/{}'.format(basePath, outSample)
    if not os.path.isdir(tmp): os.makedirs(tmp)
    os.chdir(tmp)

    path = f'/home/project/Pertb_benchmark/DataSet/{DataSet}/filter_hvg5000_logNor.h5ad'
    adata = sc.read_h5ad(path)
    perturbations = list(adata.obs['condition2'].unique())
    perturbations = [i for i in perturbations if i != 'control']
    adata_test = adata[adata.obs['iid_test'] == 'test']
    adata_train = adata[adata.obs['iid_test'] == 'train']
    for perturbation in perturbations:
        adata_train_p = adata_train[adata_train.obs["condition2"] == perturbation]
        exp = adata_train_p.to_df()
        train_mean = list(exp.mean(axis=0))
        control_adata = adata[(adata.obs['condition1'] == outSample) & (adata.obs['condition2']=='control')]
        control_adata.obs['perturbation'] = 'control'
        control_exp = control_adata.to_df()
        control_std = list(np.std(control_exp))
        control_std = [i if not np.isnan(i) else 0 for i in control_std]
        cellNum = control_adata.shape[0]
        expression_matrix = generateExp(cellNum, train_mean, control_std)
        pred_perturbed_adata = control_adata.copy()
        pred_perturbed_adata.X = expression_matrix
        pred_perturbed_adata.obs['perturbation'] = 'imputed'
        treat_adata = adata_test[(adata_test.obs['condition1'] == outSample) & (adata_test.obs['condition2']==perturbation)]
        treat_adata.obs['perturbation'] = 'stimulated'
        result = ad.concat([control_adata, treat_adata, pred_perturbed_adata])
        result.write_h5ad('{}_imputed.h5ad'.format(perturbation))



def KangMain(DataSet):
    filein_tmp = '/home/project/Pertb_benchmark/DataSet/{}/filter_hvg5000_logNor.h5ad'.format(DataSet)
    adata_tmp = sc.read_h5ad(filein_tmp)
    outSamples = list(adata_tmp.obs['condition1'].unique())
    for outSample in tqdm(outSamples):
        Kang_OutSample(DataSet, outSample)

DataSets = ["kangCrossCell", "kangCrossPatient", "Parekh",  'Salmonella', "Haber", "crossPatient", "KaggleCrossPatient",
            "KaggleCrossCell", "crossSpecies", "McFarland", "Afriat", "TCDD", "sciplex3"]

DataSets = ['kangCrossCell']
if __name__ == '__main__':
    for DataSet in DataSets:
        print (DataSet)
        KangMain(DataSet)