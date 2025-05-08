import sys
sys.path.append('/home/project/Pertb_benchmark')
from myUtil import *
from myUtil1 import *
import pickle
import scanpy as sc
from tqdm import tqdm
import warnings, json
from gears import PertData
from pathlib import Path
warnings.filterwarnings('ignore')


def clean_condition(condition):
    return condition.replace('+ctrl', '').replace('ctrl+', '').strip()

def predExp_single(DataSet, seed=1, senario='trainMean'):
    dirName = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/GEARS'.format(DataSet)
    os.chdir(dirName)
    pert_data = PertData('./data') # specific saved folder
    pert_data.load(data_path = './data/train') # load the processed data, the path is saved folder + dataset_name
    pert_data.prepare_split(split = 'simulation', seed = seed, train_gene_set_size=.8) # get 

    dirName = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/{}'.format(DataSet, senario)
    if not os.path.isdir(dirName): os.makedirs(dirName)
    os.chdir(dirName)

    adata = pert_data.adata
    train_perts = pert_data.set2conditions['train']
    train_perts = [i for i in train_perts if i != 'ctrl']
    test_perts = pert_data.set2conditions['test']
    test_perts = [clean_condition(i) for i in test_perts]
    if senario == 'trainMean':
        adata_train = adata[adata.obs['condition'].isin(train_perts)]
    else:
        adata_train = adata[adata.obs['perturbation'].isin(['control'])]
    exp = adata_train.to_df()
    train_mean = exp.mean(axis=0).to_frame().T
    pred = pd.concat([train_mean] * len(test_perts))
    pred.index = test_perts

    dirOut = 'savedModels{}'.format(seed)
    if not os.path.isdir(dirOut): os.makedirs(dirOut)
    pred.to_csv('{}/pred.tsv'.format(dirOut),  sep='\t')


def getConditionExp_comb(adata_train_single, test_perts_combo, adata_control):
    adata_train_single_pert = list(adata_train_single.obs['perturbation'].unique())  ### 训练数据集单个扰动的列
    exp_list = []
    mean_groupby = adata_train_single.to_df().groupby(adata_train_single.obs['perturbation']).mean()
    mean_all = mean_groupby.mean(axis=0)
    control_mean = adata_control.to_df().mean()
    for comb in test_perts_combo:
        geneA, geneB = comb.split('+')
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
    exp = adata_train_single.to_df()
    train_mean = exp.mean(axis=0).to_frame().T
    pred = pd.concat([train_mean] * len(test_perts_single))
    pred.index = test_perts_single
    return pred


def predExp_combination(DataSet, seed=1, senario='trainMean'):
    if senario == 'controlMean':
        predExp_single(DataSet, seed, senario)
    else:
        dirName = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/GEARS'.format(DataSet)
        os.chdir(dirName)
        pert_data = PertData('./data') # specific saved folder   download gene2go_all.pkl
        pert_data.load(data_path = './data/train') # load the processed data, the path is saved folder + dataset_name
        pert_data.prepare_split(split = 'simulation', seed = seed, train_gene_set_size=.8) # get 

        dirName = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/{}'.format(DataSet, senario)
        if not os.path.isdir(dirName): os.makedirs(dirName)
        os.chdir(dirName)

        adata = pert_data.adata
        train_perts = pert_data.set2conditions['train']
        train_perts = [clean_condition(i) for i in train_perts]
        train_perts = [i for i in train_perts if i != 'ctrl']
        train_perts_single = [i for i in train_perts if '+' not in i]
        adata_train_single = adata[adata.obs['perturbation'].isin(train_perts_single)]

        test_perts_combo = pert_data.subgroup['test_subgroup']['combo_seen0'] + \
        pert_data.subgroup['test_subgroup']['combo_seen1'] + pert_data.subgroup['test_subgroup']['combo_seen2']
        
        test_perts_single = pert_data.subgroup['test_subgroup']['unseen_single']
        test_perts_single = [clean_condition(i) for i in test_perts_single]

        adata_control = adata[adata.obs['perturbation'].isin(['control'])]
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
    dirName = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/{}'.format(DataSet, senario)
    os.chdir(dirName)
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
    for pertGene in tqdm(pertGenes):
        cellNum = adata[(adata.obs['perturbation'] == pertGene) & (adata.obs['Expcategory']=='imputed')].shape[0]
        means = list(exp.loc[pertGene, ])
        expression_matrix = generateExp(cellNum, means, control_std)
        adata[(adata.obs['perturbation'] == pertGene) & (adata.obs['Expcategory']=='imputed')].X = expression_matrix
    adata.write('savedModels{}/result.h5ad'.format(seed))


seeds = [1, 2, 3]
SinglePertDataSets = ['Adamson', "Frangieh", "TianActivation", "TianInhibition", "Replogle_exp7", "Replogle_exp8", "Papalexi", "Replogle_RPE1essential", "Replogle_K562essential"]
CombPertDataSets = ['Norman', 'Wessels', 'Schmidt', "Replogle_exp6", "Replogle_exp10"]
SinglePertDataSets = ["Papalexi"]

### conda activate gears

if __name__ == '__main__':
    print ('hello, world')

    #### single
    for DataSet in tqdm(SinglePertDataSets):
        for seed in tqdm(seeds):
            predExp_single(DataSet, seed = seed, senario='controlMean')
            predExp_single(DataSet, seed = seed, senario='trainMean')
            #generateH5ad(DataSet, seed, senario='controlMean')
            #generateH5ad(DataSet, seed, senario='trainMean')
            

    # for DataSet in tqdm(CombPertDataSets):
    #     for seed in tqdm(seeds):
    #         predExp_combination(DataSet, seed = seed, senario='controlMean')
    #         predExp_combination(DataSet, seed = seed, senario='trainMean')

    #         generateH5ad(DataSet, seed, senario='controlMean')
    #         generateH5ad(DataSet, seed, senario='trainMean')