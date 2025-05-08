import sys
sys.path.append('/home/project/Pertb_benchmark')
from myUtil import *
import pickle
import scanpy as sc
from tqdm import tqdm
import warnings, json
from gears import PertData
from pathlib import Path
warnings.filterwarnings('ignore')

def normalize_condition_names(obs):
  import pandas as pd
  ser = pd.Series(["+".join(sorted(x.split("+"))) for x in obs['condition']], dtype = "category", index = obs.index)
  obs['condition'] = ser
  return obs


def preData1(DataSet, seed, isComb=False):
    dirName = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/linearModel'.format(DataSet)
    if not os.path.isdir(dirName): os.makedirs(dirName)
    os.chdir(dirName)

    if os.path.isdir('data'):
        cmd = 'rm -r data'; subprocess.call(cmd, shell=True)
    cmd = 'cp -r ../GEARS/data/  .'
    subprocess.call(cmd, shell=True)

    if isComb:
        pert_data = PertData('./data') # specific saved folder   download gene2go_all.pkl
        pert_data.load(data_path = './data/train') # load the processed data, the path is saved folder + dataset_name
        pert_data.prepare_split(split = 'simulation', seed = seed, train_gene_set_size=.8) # get data split with seed
        norman_adata = pert_data.adata
        new_obs = normalize_condition_names(norman_adata.obs.copy())
        if not norman_adata.obs.equals(new_obs):
            norman_adata.obs = new_obs
            # Override the perturb_processed.h5ad
            norman_adata.write_h5ad("data/train/perturb_processed.h5ad")
            # Delete the data_pyg folder because it has the problematic references to the 
            data_pyg_folder = Path("data/train/data_pyg")
            if data_pyg_folder.exists():
                (data_pyg_folder / "cell_graphs.pkl").unlink(missing_ok = True)
                data_pyg_folder.rmdir()
                pert_data = PertData('./data') # specific saved folder
                pert_data.load(data_path = './data/train') # load the processed data,
        conds = norman_adata.obs['condition'].cat.remove_unused_categories().cat.categories.tolist()
        single_pert = [x for x in conds if 'ctrl' in x]
        double_pert = np.setdiff1d(conds, single_pert).tolist()
        double_training = np.random.choice(double_pert, size=len(double_pert) // 2, replace=False).tolist()
        double_test = np.setdiff1d(double_pert, double_training).tolist()
        double_test = double_test[0:(len(double_test)//2)]
        double_holdout = np.setdiff1d(double_pert, double_training + double_test).tolist()
        set2conditions = {   #### 组合扰动50% is  double_training, 25% is  double_test, 25% is double_holdout
            "train": single_pert + double_training,
            "test": double_test,
            "val": double_holdout
        }
    else:
        pert_data = PertData('./data')
        pert_data.load(data_path = './data/train') # load the
        pert_data.prepare_split(split = 'simulation', seed = seed, train_gene_set_size=.8)
        set2conditions = pert_data.set2conditions

    outfile = 'data/train/splits/set2conditions_{}.tsv'.format(seed)
    with open(outfile, "w") as outfile:
        json.dump(set2conditions, outfile)


def runLinearModel(DataSet, seed):
    dirName = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/linearModel'.format(DataSet)
    if not os.path.isdir(dirName): os.makedirs(dirName)
    os.chdir(dirName)
    dirOut = 'savedModels{}'.format(seed)
    if not os.path.isdir(dirOut): os.makedirs(dirOut)
    cmd = '/usr/local/bin/Rscript   /home/project/Pertb_benchmark/manuscipt2/linearModel.r  {}  {}'.format(seed, dirName)
    print (cmd)
    subprocess.call(cmd, shell=True)

def generateExp(cellNum, means, std):
    expression_matrix = np.array([
    np.random.normal(loc=means[i], scale=std[i], size=cellNum) 
    for i in range(len(means))]).T
    return expression_matrix


### 根据预测的生成表达量
def generateH5ad(DataSet, seed = 1):
    dirName = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/linearModel'.format(DataSet)
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
    for pertGene in pertGenes:
        cellNum = adata[(adata.obs['perturbation'] == pertGene) & (adata.obs['Expcategory']=='imputed')].shape[0]
        means = list(exp.loc[pertGene, ])
        expression_matrix = generateExp(cellNum, means, control_std)
        adata[(adata.obs['perturbation'] == pertGene) & (adata.obs['Expcategory']=='imputed')].X = expression_matrix
    adata.write('savedModels{}/result.h5ad'.format(seed))


def predComb(DataSet, seed):
    dirName = '/home/project/Pertb_benchmark/DataSet2/{}/hvg5000/linearModel'.format(DataSet)
    os.chdir(dirName)
    filein1 = 'savedModels{}/pred.tsv'.format(seed)
    filein2 = '../trainMean/savedModels{}/pred.tsv'.format(seed)
    dat1 = pd.read_csv(filein1, sep='\t', index_col=0).T
    dat2 = pd.read_csv(filein2, sep='\t', index_col=0)
    expGene = list(np.intersect1d(dat1.columns, dat2.columns))
    single_perts = [i for i in dat1.index if '+' not in i and i != 'ctrl']
    comb_perts = [i for i in dat2.index if '+' in i]
    dat1 = dat1.loc[single_perts, expGene]; dat2 = dat2.loc[comb_perts, expGene]
    pred = pd.concat([dat1, dat2])
    dirOut = 'savedModels{}'.format(seed)
    pred.to_csv('{}/pred.tsv'.format(dirOut),  sep='\t')

'''
conda activate linear_perturbation_prediction
'''

SinglePertDataSets = ['Adamson', "Frangieh", "TianActivation", "TianInhibition", "Replogle_exp7", "Replogle_exp8", "Papalexi", "Replogle_RPE1essential", "Replogle_K562essential"]
CombPertDataSets = ['Norman', 'Wessels', 'Schmidt', "Replogle_exp6", "Replogle_exp10"]
seeds = [1, 2, 3]
SinglePertDataSets = ["Papalexi"]
CombPertDataSets = ['Norman']

if __name__ == '__main__':
    print ('hello, world')
    for DataSet in CombPertDataSets[1:2]:
        print (DataSet)
        isComb = True
        for seed in tqdm(seeds):
            preData1(DataSet, seed, isComb=isComb)
            runLinearModel(DataSet, seed)
            predComb(DataSet, seed)
            generateH5ad(DataSet, seed)