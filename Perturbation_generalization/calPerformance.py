import os, sys, warnings
sys.path.append('/home/wzt/project/Pertb_benchmark')
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
    filein = '/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/{}/DEG_hvg5000.pkl'.format(DataSet)
    with open(filein, 'rb') as fin:
        mydict = pickle.load(fin)
        DegList = list(mydict[perturb].index[:numDEG])
    return DegList

def f_subSample(adata, n_samples = 2000, control_tag='control'):
    def subSample(adata, n_samples):
        if adata.shape[0] <= n_samples:
            return adata
        else:
            sampled_indices = np.random.choice(adata.n_obs, n_samples, replace=False)
            adata_sampled = adata[sampled_indices, :]
            return adata_sampled
    np.random.seed(42)  ### 设置种子，使得结果具有可重复性
    adata_imputed = adata[adata.obs['Expcategory'] ==  "imputed"]
    adata_imputed = subSample(adata_imputed, n_samples)

    adata_treat = adata[adata.obs['Expcategory'] ==  "stimulated"]
    adata_treat = subSample(adata_treat, n_samples)

    adata_control = adata[adata.obs['Expcategory'] ==  control_tag]
    adata_control = subSample(adata_control, n_samples)

    adata1 = ad.concat([adata_imputed, adata_control, adata_treat])
    return adata1

class SuppressOutput:
    def __enter__(self):
        # 保存当前的标准输出
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        # 将标准输出和标准错误输出重定向到/dev/null
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_value, traceback):
        # 恢复标准输出
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
        adata, DataSet, method, numDEG, seed, perturb, condition_column, control_tag = X
        mylist = []
        a = adata[(adata.obs[condition_column] == perturb) & (adata.obs['Expcategory'] == "stimulated")].shape[0]
        b = adata[(adata.obs[condition_column] == perturb) & (adata.obs['Expcategory'] == "imputed")].shape[0]
        if a ==0 or b == 0: return
        adata = adata[adata.obs[condition_column].isin([control_tag, perturb])]  ### 保留该扰动信息
        DegList = getDEG(DataSet, perturb, numDEG)
        DegList = [i for i in DegList if i in adata.var_names]  #### scFoundation存在少量差异
        adata = adata[:, DegList]
        adata = checkNan(adata, condition_column, control_tag)
        adata = calculateDelta(adata)
        adata.layers['X'] = adata.X
        if doSubSample: adata_subSample = f_subSample(adata, 2000, control_tag)
        for metric in chain(metrics):
            if metric == 'wasserstein' and numDEG == 5000: continue  #### 推土机距离5000维度会报错
            try:
                with SuppressOutput():
                    Distance = pt.tools.Distance(metric=metric,  layer_key='X')
                    if doSubSample and metric in ['edistance', 'wasserstein', 'mmd']:
                        pairwise_df = Distance.onesided_distances(adata_subSample, groupby="Expcategory", selected_group='imputed', groups=["stimulated"])  ###
                    else:
                        pairwise_df = Distance.onesided_distances(adata, groupby="Expcategory", selected_group='imputed', groups=["stimulated"])  ### 已经转换好了，不需要1-pairwise_df进行转换
                    perf = round(pairwise_df['stimulated'], 4)
            except Exception as e:
                print (e)
                print (X, metric); perf = np.nan
            mylist.append(perf)
        
        if numDEG == 5000:
            metrics_tmp = [i for i in metrics if i != 'wasserstein']
            dat = pd.DataFrame({'performance': mylist, "metric": metrics_tmp})
        else:
            dat = pd.DataFrame({'performance': mylist, "metric": metrics})
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
    results_list = []
    os.chdir('/home/wzt/project/Pertb_benchmark/DataSet2/{}'.format(DataSet))
    filein = 'hvg5000/{}/savedModels{}/result.h5ad'.format(method, seed)
    if not os.path.isfile(filein): return
    adata = sc.read_h5ad(filein)
    perturbations = adata.obs[condition_column].unique()
    perturbations = [i for i in perturbations if i not in control_list]
    for numDEG in numDEG_list:
        for perturb in perturbations:
            results_list.append(calPerfor([adata, DataSet, method, numDEG, seed, perturb, condition_column, control_tag]))
    results = pd.concat(results_list)
    return results


def ff_calPerfor(DataSet, condition_column = 'perturbation', control_tag = 'control', redo = False):
    mylist_parameter = []
    print (DataSet)
    fileout = '/home/wzt/project/Pertb_benchmark/DataSet2/{}/performance_delta.tsv'.format(DataSet)
    if not redo:
        dat = pd.read_csv(fileout, sep='\t')
        addMethods = [i for i in methods if i not in list(dat['method'])]
        for seed in seeds:
            for method in addMethods:
                mylist_parameter.append([DataSet, method, seed, condition_column, control_tag])
        results = myPool(f_calPerfor, mylist_parameter, processes=3)   ### 适当设置
        results = pd.concat(results + [dat])
        results.to_csv(fileout, sep='\t', index=False)
    else:
        for seed in seeds:
            for method in methods:
                mylist_parameter.append([DataSet, method, seed, condition_column, control_tag])
        results = myPool(f_calPerfor, mylist_parameter, processes=3)   ### 适当设置
        results = pd.concat(results)
        results.to_csv(fileout, sep='\t', index=False)


def UMPAPlot(X):
    DataSet, method, outSample, perturb = X
    os.chdir('/home/wzt/project/Pertb_benchmark/DataSet/{}/outSample/hvg5000/{}/{}'.format(DataSet, method, outSample))
    if not os.path.isdir('figures'): os.makedirs('figures')
    filein = '{}_imputed.h5ad'.format(perturb)
    if not os.path.isfile(filein): return   ####  这个病人没有该扰动的真实值
    fileout1 = '_{}.pdf'.format(perturb)
    fileout2 = '_{}_deg.pdf'.format(perturb)
    if os.path.isfile('figures/{}'.format(fileout1)) and os.path.isfile('figures/{}'.format(fileout2)): return  #### 有文件不跑
    adata = sc.read_h5ad(filein)
    adata = checkNan(adata)
    if method in ['bioLord', 'biolord', 'scDisInFact']:  ### 删除多余的预测数据
        try:
            adata_control = adata[adata.obs['perturbation'] == 'control']
            adata = ad.concat([adata_control, adata])
        except Exception as e:
            print (e)
    if doSubSample: adata_subSample = f_subSample(adata, n_samples=2000)
    DEGlist = getDEG(DataSet, 5000, outSample, perturb, 100)   #### 差异基因100个
    adata_deg = adata_subSample[:, DEGlist].copy()

    sc.tl.pca(adata_subSample)
    sc.pp.neighbors(adata_subSample)
    sc.tl.umap(adata_subSample)
    sc.pl.umap(adata_subSample,  color="perturbation", palette=['grey', 'orange', 'green'],
    save = fileout1,  show=False)

    sc.pp.neighbors(adata_deg, use_rep='X')
    sc.tl.umap(adata_deg)
    sc.pl.umap(adata_deg,  color="perturbation", palette=['grey', 'orange', 'green'],
    save = fileout2,  show=False)



def f_UMPAPlot(DataSet):
    mylist = []
    filein_tmp = '/home/wzt/project/Pertb_benchmark/DataSet/{}/filter_hvg500_logNor.h5ad'.format(DataSet)
    adata_tmp = sc.read_h5ad(filein_tmp)
    outSamples = list(adata_tmp.obs['condition1'].unique())
    perturbs = list(adata_tmp.obs['condition2'].unique())
    perturbs = [i for i in perturbs if i != 'control']
    for method in methods:
        for outSample in outSamples:
            for perturb in perturbs:
                mylist.append([DataSet, method, outSample, perturb])
    myPool(UMPAPlot, mylist, processes=10)


#### conda activate pertpyV7
doSubSample = True
classI = ["mse", "mmd", "euclidean", "edistance",  "mean_absolute_error"]  #### ks_test, t-test受到预测细胞数量的影响
classII = ["cosine_distance", "pearson_distance", "r2_distance", ]  ####  classifier_proba 在所有数据都差不多
classIII = ['spearman_distance', 'wasserstein']  ### typeIII本身不建议作为metric, 但是先计算出来看看
control_list = ['control', 'MCF7_control_1.0', 'A549_control_1.0', 'K562_control_1.0']

metrics = (classI + classII + classIII)
metrics = ['mse', 'pearson_distance', 'edistance']  ###
seeds = [1, 2, 3]
numDEG_list = [100, 5000]

methods= ['scGPT', 'GEARS', 'AttentionPert', 'GenePert', 'linearModel', 'trainMean', 'controlMean', 'CPA', 'bioLord', 'scFoundation', 'scouter'] ##  'STAMP'
#methods= ['CPA', 'chemCPA', 'bioLord', 'trainMean', 'controlMean']
#methods= ['CPA', 'trainMean', 'controlMean']

cmd = 'export OPENBLAS_NUM_THREADS=20'; subprocess.call(cmd, shell=True)
cmd = 'export JAX_PLATFORMS=cpu'; subprocess.call(cmd, shell=True)

### export OPENBLAS_NUM_THREADS=20, export JAX_PLATFORMS=cpu 跑之前先设置这个


SinglePertDataSets = ['Adamson', "Frangieh", "TianActivation", "TianInhibition", "Replogle_exp7", "Replogle_exp8", "Papalexi", "Replogle_RPE1essential", "Replogle_K562essential"]
CombPertDataSets = ['Norman', 'Wessels', 'Schmidt', "Replogle_exp6"]

if __name__ == '__main__':
    print ('hello, world')
    for DataSet_tmp in tqdm(CombPertDataSets[:1]):
        ff_calPerfor(DataSet = DataSet_tmp, redo= True)
