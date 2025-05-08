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

def checkNan(adata):
    if sparse.issparse(adata.X):
        adata.X = adata.X.toarray()
    nan_rows = np.where(np.isnan(adata.X).any(axis=1))[0]
    a = adata[adata.obs['perturbation'] == 'control'].X.mean(axis=0)
    a = a.reshape([1, -1])
    b = np.tile(a, [len(nan_rows), 1])
    adata[nan_rows].X = b
    return adata

def calculateDelta(adata):
    adata_control = adata[adata.obs['perturbation'] == 'control'].copy()
    adata_imputed = adata[adata.obs['perturbation'] == 'imputed'].copy()
    adata_stimulated = adata[adata.obs['perturbation'] == 'stimulated'].copy()
    control_mean = adata_control.X.mean(axis=0)
    adata_imputed.X = adata_imputed.X - control_mean
    adata_stimulated.X = adata_stimulated.X - control_mean
    adata_delta = ad.concat([adata_control, adata_imputed, adata_stimulated])
    return adata_delta


def getDEG(DataSet, hvg, outSample, perturb, DEG):
    filein = '/home/project/Pertb_benchmark/DataSet/{}/DEG_hvg{}.pkl'.format(DataSet, hvg)
    with open(filein, 'rb') as fin:
        mydict = pickle.load(fin)
        if perturb in mydict[outSample]:
            DegList = list(mydict[outSample][perturb].index[:DEG])
            return DegList
        else:
            return None

def getDEG_cutoff(DataSet, hvg, outSample, perturb):
    filein = '/home/project/Pertb_benchmark/DataSet/{}/DEG_hvg{}.pkl'.format(DataSet, hvg)
    with open(filein, 'rb') as fin:
        mydict = pickle.load(fin)
        if perturb in mydict[outSample]:
            tmp = mydict[outSample][perturb]
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
    DataSet, method, hvg, outSample, perturb, DEG, metric = X
    myPerformance = []
    os.chdir('/home/project/Pertb_benchmark/DataSet/{}'.format(DataSet))
    filein = '{}/hvg{}/{}/{}/{}_imputed.h5ad'.format(senario, hvg, method, outSample, perturb)
    if not os.path.isfile(filein): return
    adata = sc.read_h5ad(filein)
    adata = checkNan(adata)
    DegList = getDEG(DataSet, hvg, outSample, perturb, DEG)
    if DegList is None: return
    adata = adata[:, DegList].copy()   
    if metric == 'pearson_distance':
        adata = calculateDelta(adata)
    adata.layers['X'] = adata.X
    if doSubSample: adata_subSample = f_subSample(adata, n_samples=2000)
    if metric == 'wasserstein' and DEG == 5000: return
    try:
        with SuppressOutput():
            Distance = pt.tools.Distance(metric=metric,  layer_key='X')
            if doSubSample and metric in ['edistance', 'wasserstein', 'mean_var_distribution', 'sym_kldiv']:
                pairwise_df = Distance.onesided_distances(adata_subSample, groupby="perturbation", selected_group='imputed', groups=["stimulated"])
            else:
                pairwise_df = Distance.onesided_distances(adata, groupby="perturbation", selected_group='imputed', groups=["stimulated"])
            perf = round(pairwise_df['stimulated'], 4)
            if metric == 'sym_kldiv':
                perf = np.log2(perf + 1)
    except Exception as e:
        print (e)
        print (X, metric); perf = np.nan
    myPerformance.append(perf)
    dat = pd.DataFrame({'performance': myPerformance, "metric": metric})
    dat['DataSet'] = DataSet; dat['method'] = method; dat['hvg'] = hvg; dat['outSample'] = outSample; dat['perturb'] = perturb; dat['DEG'] = DEG
    dat['Ncontrol'] = adata[adata.obs['perturbation'] ==  "control"].shape[0]
    dat['Nimputed'] = adata[adata.obs['perturbation'] ==  "imputed"].shape[0]
    dat['Nstimulated'] = adata[adata.obs['perturbation'] ==  "stimulated"].shape[0]
    return dat


def f_calPerfor(DataSet):
    mylist = []
    filein_tmp = f'/home/project/Pertb_benchmark/DataSet/{DataSet}/filter_hvg5000_logNor.h5ad'
    adata_tmp = sc.read_h5ad(filein_tmp)
    outSamples = list(adata_tmp.obs['condition1'].unique())
    perturbs = list(adata_tmp.obs['condition2'].unique())
    perturbs = [i for i in perturbs if i != 'control']
    for DEG in [100, 5000]:
        for method in methods:
            for outSample in outSamples:
                for perturb in perturbs:
                    for metric in metrics:
                        mylist.append([DataSet, method, "5000", outSample, perturb, DEG, metric])
    results = myPool(calPerfor, mylist, processes=10)
    results = pd.concat(results)
    fileout = f'/home/project/Pertb_benchmark/DataSet/{DataSet}/performance.tsv'
    results.to_csv(fileout, sep='\t', index=False)

DataSets = ["kangCrossCell", "kangCrossPatient",  "Parekh", "Haber", "crossPatient", "KaggleCrossPatient",
            "KaggleCrossCell", "crossSpecies", "McFarland", "Afriat", "TCDD", "sciplex3"]



#### conda activate pertpyV7
doSubSample = True 
metrics = ['mse', 'pearson_distance', 'edistance', 'sym_kldiv', 'wasserstein']
methods= ['cellot', 'trVAE', 'scPreGAN', 'inVAE', 'SCREEN', 'scDisInFact',  'scGen', 'bioLord',  
            'scPRAM', 'scVIDR', 'trainMean', 'controlMean', 'baseMLP', 'baseReg']

### export OPENBLAS_NUM_THREADS=20
### export JAX_PLATFORMS=cpu
if __name__ == '__main__':
    print ('hello, world')
    f_calPerfor('kangCrossCell')
    
