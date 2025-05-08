import sys
sys.path.append('/home/project/Pertb_benchmark')
from myUtil import *
from myUtil1 import *
import pickle
import scanpy as sc
from tqdm import tqdm
import warnings, json
from pathlib import Path
warnings.filterwarnings('ignore')

def clean_condition(condition):
    return condition.replace('+ctrl', '').replace('ctrl+', '').strip()

def predExp_single(DataSet, seed, senario='trainMean', ood = 'ood'):
    dirName = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/{}'.format(DataSet, senario)
    if not os.path.isdir(dirName): os.makedirs(dirName)
    os.chdir(dirName)
    adata = sc.read_h5ad('../../filter_hvg5000.h5ad')
    splits_key = 'split_ood_multi_task{}'.format(seed)
    test_perts = list(adata.obs.loc[adata.obs[splits_key] == ood]['cov_drug_dose_name'].unique())
    if senario == 'trainMean':
        adata_train = adata[adata.obs[splits_key] == 'train']
    else:
        adata_train = adata[adata.obs['condition'] == 'control']
    exp = adata_train.to_df()
    train_mean = exp.mean(axis=0).to_frame().T
    pred = pd.concat([train_mean] * len(test_perts))
    pred.index = test_perts
    dirOut = 'savedModels{}'.format(seed)
    if not os.path.isdir(dirOut): os.makedirs(dirOut)
    pred.to_csv('{}/pred.tsv'.format(dirOut),  sep='\t')

def getConditionExp_comb(adata_train_single, test_perts_combo, adata_control):
    adata_train_single_pert = list(adata_train_single.obs['cov_drug_dose_name'].unique())  ### 训练数据集的单个扰动
    exp_list = []
    mean_groupby = adata_train_single.to_df().groupby(adata_train_single.obs['cov_drug_dose_name']).mean()
    mean_all = mean_groupby.mean(axis=0)
    control_mean = adata_control.to_df().mean()
    for comb in test_perts_combo:
        cov, geneAgeneB, doseAdoseB = comb.split('_')
        geneA, geneB = geneAgeneB.split('+')
        doseA, doseB = doseAdoseB.split('+')
        geneA = '_'.join([cov, geneA, doseA])
        geneB = '_'.join([cov, geneB, doseB])

        if geneA in adata_train_single_pert and geneB not in adata_train_single_pert:
            exp = mean_groupby.loc[geneA, ] + mean_all - control_mean
        elif geneA not in adata_train_single_pert and geneB in adata_train_single_pert:
            exp = mean_groupby.loc[geneB, ] + mean_all - control_mean
        elif geneA not in adata_train_single_pert and geneB not in adata_train_single_pert:
            exp = mean_all + mean_all - control_mean
        elif geneA in adata_train_single_pert and geneB in adata_train_single_pert:
            exp = mean_groupby.loc[geneA, ] + mean_groupby.loc[geneB, ] - control_mean
        exp_list.append(exp)
    exp_all = pd.concat(exp_list, axis=1).T
    exp_all.index = test_perts_combo
    return exp_all

def getConditionExp_single(adata_train_single, test_perts_single):
    if len(test_perts_single) == 0: return  ### 没有元素返回空值
    exp = adata_train_single.to_df()
    train_mean = exp.mean(axis=0).to_frame().T
    pred = pd.concat([train_mean] * len(test_perts_single))
    pred.index = test_perts_single
    return pred


def predExp_combination(DataSet, seed, senario='trainMean'):
    if senario == 'controlMean':
        predExp_single(DataSet, seed, senario, ood = 'test')  ### 和single一样, 直接用control的平均值
    else:
        dirName = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/{}'.format(DataSet, senario)
        if not os.path.isdir(dirName): os.makedirs(dirName)
        os.chdir(dirName)
        adata  =sc.read_h5ad('../../filter_hvg5000.h5ad')
        splits_key = 'split_ood_multi_task{}'.format(seed)

        train_perts = list(adata.obs.loc[adata.obs[splits_key] == "train"]['cov_drug_dose_name'].unique())
        train_perts = [i for i in train_perts if i != 'A549_control_1.0']  #### ctrl
        train_perts_single = [i for i in train_perts if '+' not in i]
        adata_train_single = adata[adata.obs['cov_drug_dose_name'].isin(train_perts_single)]

        test_perts = list(adata.obs.loc[adata.obs[splits_key] == "test"]['cov_drug_dose_name'].unique())
        test_perts = [i for i in test_perts if i != 'A549_control_1.0']  #### ctrl
        test_perts_combo =  [i for i in test_perts if '+' in i]
        test_perts_single = [i for i in test_perts if '+' not in i]

        adata_control = adata[adata.obs['cov_drug_dose_name'].isin(['A549_control_1.0'])]
        pred_comb = getConditionExp_comb(adata_train_single, test_perts_combo, adata_control)
        pred_single = getConditionExp_single(adata_train_single, test_perts_single)
        pred = pd.concat([pred_comb, pred_single])

        dirOut = 'savedModels{}'.format(seed)
        if not os.path.isdir(dirOut): os.makedirs(dirOut)
        pred.to_csv('{}/pred.tsv'.format(dirOut),  sep='\t')


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




SinglePertDataSets = ['sciplex3_MCF7', 'sciplex3_K562', 'sciplex3_A549']
CombPertDataSets = ["sciplex3_comb"]

'''
conda activate cpa
'''

if __name__ == '__main__':
    print ('hello, world')
    for DataSet in tqdm(SinglePertDataSets):
        for seed in tqdm([1, 2, 3]):
            predExp_single(DataSet, seed = seed, senario='trainMean')
            generateH5ad(DataSet, seed, senario='trainMean')

    for DataSet in tqdm(CombPertDataSets):
        for seed in tqdm([4, 5, 6]):
            predExp_combination(DataSet, seed = seed, senario='trainMean')
            generateH5ad(DataSet, seed, senario='trainMean')


            
            