import os, sys, warnings
sys.path.append('/home/project/Pertb_benchmark')
from myUtil import *
import pickle
import pertpy as pt  # type: ignore
from itertools import chain
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def checkNan(adata, condition_column = 'perturbation', control_tag = 'control'):
    adata1 = adata.copy()
    if sparse.issparse(adata.X):
        adata1.X = adata.X.toarray()
    nan_rows = np.where(np.isnan(adata1.X).any(axis=1))[0]
    if len(nan_rows) >= 1:
        a = adata1[adata1.obs[condition_column] == control_tag].X.mean(axis=0)
        a = a.reshape([1, -1])
        b = np.tile(a, [len(nan_rows), 1])
        adata1[nan_rows].X = b
    return adata1


def getDEG(DataSet, perturb, numDEG):
    import pickle
    filein = '/home/project/Pertb_benchmark/DataSet2/{}/DEG_hvg5000.pkl'.format(DataSet)
    with open(filein, 'rb') as fin:
        mydict = pickle.load(fin)
        DegList = list(mydict[perturb].index[:numDEG])
    return DegList

def getDEG_cutoff(DataSet, perturb):
    filein = f'/home/project/Pertb_benchmark/DataSet2/{DataSet}/DEG_hvg5000.pkl'
    with open(filein, 'rb') as fin:
        mydict = pickle.load(fin)
        if perturb in mydict:
            tmp = mydict[perturb]
            tmp_filter = tmp[(tmp['pvals_adj'] <= 0.01) & ((tmp['foldchanges'] >=2) | ((tmp['foldchanges'] <= 0.5)))]
            DegList = list(tmp_filter.index)
            if len(DegList) == 0:
                return None
            else:
                return DegList
        else:
            return None

def f_subSample(adata, n_samples = 2000):
    def subSample(adata, n_samples):
        if adata.shape[0] <= n_samples:
            return adata
        else:
            sampled_indices = np.random.choice(adata.n_obs, n_samples, replace=False)
            adata_sampled = adata[sampled_indices, :]
            return adata_sampled
    np.random.seed(42)
    adata_imputed = adata[adata.obs['Expcategory'] ==  "imputed"]
    adata_imputed = subSample(adata_imputed, n_samples)

    adata_treat = adata[adata.obs['Expcategory'] ==  "stimulated"]
    adata_treat = subSample(adata_treat, n_samples)

    adata_control = adata[adata.obs['Expcategory'] ==  'control']
    adata_control = subSample(adata_control, n_samples)

    adata1 = ad.concat([adata_imputed, adata_control, adata_treat])
    return adata1

class SuppressOutput:
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._stdout
        sys.stderr = self._stderr

def calculateDelta(adata):
    adata_control = adata[adata.obs['Expcategory'] == 'control'].copy()
    adata_imputed = adata[adata.obs['Expcategory'] == 'imputed'].copy()
    adata_stimulated = adata[adata.obs['Expcategory'] == 'stimulated'].copy()
    control_mean = adata_control.X.mean(axis=0)
    adata_imputed.X = adata_imputed.X - control_mean
    adata_stimulated.X = adata_stimulated.X - control_mean
    adata_delta = ad.concat([adata_control, adata_imputed, adata_stimulated])
    return adata_delta

def calPerfor(X):
    try:
        adata, DataSet, method, numDEG, seed, perturb, condition_column, control_tag, metric = X
        mylist = []
        a = adata[(adata.obs[condition_column] == perturb) & (adata.obs['Expcategory'] == "stimulated")].shape[0]
        b = adata[(adata.obs[condition_column] == perturb) & (adata.obs['Expcategory'] == "imputed")].shape[0]
        if a ==0 or b == 0: return
        adata = adata[adata.obs[condition_column].isin([control_tag, perturb])]
        DegList = getDEG(DataSet, perturb, numDEG)
        DegList = [i for i in DegList if i in adata.var_names]
        adata = adata[:, DegList].copy()
        adata = checkNan(adata, condition_column, control_tag)
        if metric == 'pearson_distance':
            adata = calculateDelta(adata)
        adata.layers['X'] = adata.X
        if doSubSample: adata_subSample = f_subSample(adata, 2000)
        try:
            with SuppressOutput():
                Distance = pt.tools.Distance(metric=metric,  layer_key='X')
                if doSubSample and metric in ['edistance', 'wasserstein', 'mean_var_distribution', 'sym_kldiv']:
                    pairwise_df = Distance.onesided_distances(adata_subSample, groupby="Expcategory", selected_group='imputed', groups=["stimulated"])  ###
                else:
                    pairwise_df = Distance.onesided_distances(adata, groupby="Expcategory", selected_group='imputed', groups=["stimulated"])  ### 已经转换好了，不需要1-pairwise_df进行转换
                perf = round(pairwise_df['stimulated'], 4)
                if metric == 'sym_kldiv':
                    perf = np.log2(perf + 1) 
        except Exception as e:
            print (e)
            print (X, metric); perf = np.nan
        mylist.append(perf)
        dat = pd.DataFrame({'performance': mylist, "metric": metric})
        dat['DataSet'] = DataSet; dat['method'] = method;  dat['perturb'] = perturb; dat['DEG'] = numDEG
        dat['Ncontrol'] = adata[adata.obs['Expcategory'] ==  "control"].shape[0]
        dat['Nimputed'] = adata[adata.obs['Expcategory'] ==  "imputed"].shape[0]
        dat['Nstimulated'] = adata[adata.obs['Expcategory'] ==  "stimulated"].shape[0]
        dat['seed'] = seed
        return dat
    except Exception as e:
        print (e); print (X)


def f_calPerfor(X):
    DataSet, method, seed, condition_column, control_tag = X
    mylist_parameter1 = []
    results_list = []
    os.chdir('/home/project/Pertb_benchmark/DataSet2/{}'.format(DataSet))
    filein = 'hvg5000/{}/savedModels{}/result.h5ad'.format(method, seed)
    if not os.path.isfile(filein): return
    adata = sc.read_h5ad(filein)
    perturbations = adata.obs[condition_column].unique()
    perturbations = [i for i in perturbations if i not in control_list]
    for metric in metrics:
        if metric == 'wasserstein':
            numDEG_list = [100]
        else: 
            numDEG_list = [100, 5000]
        for perturb in perturbations:
            for numDEG in numDEG_list:
                mylist_parameter1.append([adata, DataSet, method, numDEG, seed, perturb, condition_column, control_tag, metric])
    for i in tqdm(mylist_parameter1):
        results_list.append(calPerfor(i))
    results = pd.concat(results_list)
    return results


def ff_calPerfor(DataSet, condition_column = 'perturbation', control_tag = 'control'):
    mylist_parameter = []
    fileout = f'/home/project/Pertb_benchmark/DataSet2/{DataSet}/performance.tsv'
    for seed in seeds:
        for method in methods:
            mylist_parameter.append([DataSet, method, seed, condition_column, control_tag])
    results = myPool(f_calPerfor, mylist_parameter, processes=20)
    results = pd.concat(results)
    results.to_csv(fileout, sep='\t', index=False)



### conda activate pertpyV7
### export OPENBLAS_NUM_THREADS=20, export JAX_PLATFORMS=cpu

doSubSample = True
control_list = ['control', 'MCF7_control_1.0', 'A549_control_1.0', 'K562_control_1.0']
seeds = [1, 2, 3]
CellLineDataSets = ['A549', "K562", 'MCF7']
metrics = ['mse', 'pearson_distance', 'edistance', 'sym_kldiv', 'wasserstein']

isComb = True
if isComb:
    methods = ['CPA', 'trainMean', 'controlMean', 'baseReg', 'baseMLP',  'PRnet']
else:
    methods = ['CPA',  'trainMean', 'controlMean', 'baseReg', 'baseMLP', 'PRnet', 'cycleCDR','chemCPA', 'bioLord']




if __name__ == '__main__':
    print ('hello, world')
    ff_calPerfor('sciplex3_A549')
