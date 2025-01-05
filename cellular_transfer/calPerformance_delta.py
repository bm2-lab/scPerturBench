import os, sys, warnings
sys.path.append('/home/wzt/project/Pertb_benchmark')
from myUtil import *
import pickle
import scanpy
import pertpy as pt  # type: ignore
from itertools import chain
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def checkNan(adata):
    if sparse.issparse(adata.X):
        adata.X = adata.X.toarray()
    nan_rows = np.where(np.isnan(adata.X).any(axis=1))[0]
    a = adata[adata.obs['perturbation'] == 'control'].X.mean(axis=0)
    a = a.reshape([1, -1])
    b = np.tile(a, [len(nan_rows), 1])
    adata[nan_rows].X = b
    return adata

def getDEG(DataSet, hvg, outSample, perturb, DEG):
    filein = '/home/wzt/project/Pertb_benchmark/DataSet/{}/DEG_hvg{}.pkl'.format(DataSet, hvg)
    with open(filein, 'rb') as fin:
        mydict = pickle.load(fin)
        if perturb in mydict[outSample]:
            DegList = list(mydict[outSample][perturb].index[:DEG])
            return DegList
        else:
            return None


def deleteCondition(adata, DataSet):
    if DataSet in ["haber", "crossPatient", "crossPatientKaggle", "crossPatientKaggleCell"]:
        condition_list = list(adata.obs[adata.obs['perturbation'] == 'stimulated']['batch'].unique())
        adata = adata[adata.obs['batch'].isin(condition_list)]
    elif DataSet in ["crossSpecies", "McFarland", "Afriat"]:
        condition_list = list(adata.obs[adata.obs['perturbation'] == 'stimulated']['condition3'].unique())
        adata = adata[adata.obs['condition3'].isin(condition_list)]
    elif DataSet in ["TCDD", "sciplex3", 'sciplex3_10000']:
        condition_list = list(adata.obs[adata.obs['perturbation'] == 'stimulated']['dose'].unique()) ### gai,yuan tmp
        adata = adata[adata.obs['dose'].isin(condition_list)]
    else:
        adata = adata[adata.obs['perturbation'] != 'control']
    return adata


def f_subSample(adata, n_samples = 2000):
    def subSample(adata, n_samples):
        if adata.shape[0] <= n_samples:
            return adata
        else:
            sampled_indices = np.random.choice(adata.n_obs, n_samples, replace=False)
            adata_sampled = adata[sampled_indices, :]
            return adata_sampled
    np.random.seed(1116)
    adata_imputed = adata[adata.obs['perturbation'] ==  "imputed"]
    adata_imputed = subSample(adata_imputed, n_samples)

    adata_treat = adata[adata.obs['perturbation'] ==  "stimulated"]
    adata_treat = subSample(adata_treat, n_samples)

    adata_control = adata[adata.obs['perturbation'] ==  "control"]
    adata_control = subSample(adata_control, n_samples)

    adata1 = ad.concat([adata_control, adata_imputed, adata_treat])
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

def calPerfor(X):
    DataSet, method, hvg, outSample, perturb, DEG = X
    mylist = []
    os.chdir('/home/wzt/project/Pertb_benchmark/DataSet/{}'.format(DataSet))
    filein = 'outSample/hvg5000/{}/{}/{}_imputed.h5ad'.format(method, outSample, perturb)
    if not os.path.isfile(filein): return
    adata = sc.read_h5ad(filein)
    adata = checkNan(adata)
    adata.layers['X'] = adata.X

    ada_p = adata[(adata.obs['perturbation'] == 'imputed')]
    ada_c = adata[(adata.obs['perturbation'] == 'control')]
    ada_t = adata[(adata.obs['perturbation'] == 'stimulated')]
    c_mean = np.mean(ada_c.X, axis=0)
    ada_p.X = ada_p.X - c_mean
    ada_t.X = ada_t.X - c_mean
    adata = ad.concat([ada_p, ada_c, ada_t]) 
    adata = checkNan(adata)
    adata.layers['X'] = adata.X

    DegList = getDEG(DataSet, hvg, outSample, perturb, DEG)
    if DegList is None: return   ####
    adata = adata[:, DegList].copy()
    if method in ['bioLord', 'biolord', 'scDisInFact']:  ##
        try:
            adata_control = adata[adata.obs['perturbation'] == 'control']
            adata = deleteCondition(adata, DataSet)
            adata = ad.concat([adata_control, adata])
        except:
            print (X)
            return
    if doSubSample: adata_subSample = f_subSample(adata, n_samples=2000)
    for metric in chain(metrics):
        if metric == 'wasserstein' and DEG == 5000: continue
        try:
            with SuppressOutput():
                Distance = pt.tools.Distance(metric=metric,  layer_key='X')
                if doSubSample and metric in ['edistance', 'wasserstein', 'mmd']:
                    pairwise_df = Distance.onesided_distances(adata_subSample, groupby="perturbation", selected_group='imputed', groups=["stimulated"])  ###
                elif DataSet == 'crossPatient':
                    pairwise_df = Distance.onesided_distances(adata_subSample, groupby="perturbation", selected_group='imputed', groups=["stimulated"])  #
                else:
                    pairwise_df = Distance.onesided_distances(adata, groupby="perturbation", selected_group='imputed', groups=["stimulated"])  #
                perf = round(pairwise_df['stimulated'], 4)
        except Exception as e:
            print (e)
            print (X, metric); perf = np.nan
        mylist.append(perf)
    
    if DEG == 5000:
        metrics_tmp = [i for i in metrics if i != 'wasserstein']
        dat = pd.DataFrame({'performance': mylist, "metric": metrics_tmp})
    else:
        dat = pd.DataFrame({'performance': mylist, "metric": metrics})
    dat['DataSet'] = DataSet; dat['method'] = method; dat['hvg'] = hvg; dat['outSample'] = outSample; dat['perturb'] = perturb; dat['DEG'] = DEG
    dat['Ncontrol'] = adata[adata.obs['perturbation'] ==  "control"].shape[0]
    dat['Nimputed'] = adata[adata.obs['perturbation'] ==  "imputed"].shape[0]
    dat['Nstimulated'] = adata[adata.obs['perturbation'] ==  "stimulated"].shape[0]
    return dat


def f_calPerfor(DataSet):
    mylist = []
    filein_tmp = '/home/wzt/project/Pertb_benchmark/DataSet/{}/filter_hvg500_logNor.h5ad'.format(DataSet)
    adata_tmp = sc.read_h5ad(filein_tmp)
    outSamples = list(adata_tmp.obs['condition1'].unique())
    perturbs = list(adata_tmp.obs['condition2'].unique())
    perturbs = [i for i in perturbs if i != 'control']
    for hvg in [5000]:
        for DEG in [100, 5000]:
            for method in methods:
                for outSample in outSamples:
                    for perturb in perturbs:
                        mylist.append([DataSet, method, hvg, outSample, perturb, DEG])
    results = myPool(calPerfor, mylist, processes=2)
    results = pd.concat(results)
    fileout = '/home/wzt/project/Pertb_benchmark/DataSet/{}/{}_performance.tsv'.format(DataSet, senario)
    results.to_csv(fileout, sep='\t', index=False)

class0 = ['pearson_distance', 'mse', 'edistance']
#classI = ["mse", "mmd"]
classI = ["mse", "mmd", "euclidean", "edistance",  "mean_absolute_error"]
classII = ["pearson_distance", "r2_distance", "cosine_distance"]
classIII = ['spearman_distance', 'wasserstein']  
#### conda activate pertpyV7
doSubSample = True
senario = 'outSample'   ### outSample, inSample, holdoutExperiment
metrics = class0

methods= ['cellot', 'trVAE', 'scPreGAN', 'inVAE', 'SCREEN', 'scDisInFact',  'scGen', 'bioLord', 'scPRAM', 'scVIDR', 'trainMean','bioLord_optimized']

cmd = 'export OPENBLAS_NUM_THREADS=5'; subprocess.call(cmd, shell=True)
cmd = 'export JAX_PLATFORMS=cpu'; subprocess.call(cmd, shell=True)

### export OPENBLAS_NUM_THREADS=20, export JAX_PLATFORMS=cpu
