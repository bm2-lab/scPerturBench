import sys
sys.path.append('/home/wzt/project/Pertb_benchmark')
from myUtil1 import *   #### type:ignore
import warnings
warnings.filterwarnings('ignore')


#
def getPert(adata):
    adata = adata[~adata.obs['perturbation'].isin(['None', 'control'])]
    tmp = [i.split('+') for i in adata.obs['perturbation']]
    tmp = [i for i in tmp if i != 'control' and i != 'None']
    tmp = np.unique([i for j in tmp for i in j])
    return tmp


def fun2(adata):
    mylist = []
    for gene, cell in zip(adata.obs['perturbation'], adata.obs_names):
        if gene == 'control':
            mylist.append(cell)
        else:
            genes = gene.split('+')
            tmp = [True if i in adata.var_names else False for i in genes]
            if np.all(tmp): mylist.append(cell)
    adata1 = adata[mylist, :]
    return adata1

def f_preData1(DataSet, domaxNumsPerturb=0, domaxNumsControl=0, minNums=50):
    dirName = '/home/wzt/project/Pertb_benchmark/DataSet2/{}'.format(DataSet)
    os.chdir(dirName)
    adata1 = sc.read_h5ad('raw.h5ad')

    filterNoneNums, filterCells, filterMT, filterMinNums, adata = preData(adata1, domaxNumsPerturb=domaxNumsPerturb,  domaxNumsControl=domaxNumsControl, minNums=minNums)
    print (filterNoneNums, filterCells, filterMT, filterMinNums)

    tmp = 'highly_variable_5000'
    hvgs = list(adata.var_names[adata.var[tmp]])
    trainGene = getPert(adata)   #
    trainGene = [i for i in trainGene if i in adata.var_names]  ##
    keepGene = list(set(trainGene + hvgs))
    adata2 = adata[:, keepGene]
    adata2 = fun2(adata2)   ###
    adata2 = adata2.copy()
    if sparse.issparse(adata2.X):
        adata2.X = adata2.X.toarray()
    adata2.write_h5ad('filter_hvg5000_logNor.h5ad')
    print ("OK")

### for scFoundation
def f_preData2(DataSet):
    dirName = '/home/wzt/project/Pertb_benchmark/DataSet2/{}'.format(DataSet)
    os.chdir(dirName)
    Raw_h5ad = sc.read_h5ad('Raw.h5ad')
    filter_h5ad = sc.read_h5ad('filter_hvg5000_logNor.h5ad')
    tmp_adata = Raw_h5ad[filter_h5ad.obs_names, :]
    tmp_adata.obs = filter_h5ad.obs
    
    geneName = pd.read_csv('/home/wzt/software/scFoundation/OS_scRNA_gene_index.19264.tsv', sep='\t')
    geneName = geneName['gene_name']
    adata1 = tmp_adata[:, tmp_adata.var_names.isin(geneName)]
    tmp = [i for i in geneName if i not in adata1.var_names]
    if len(tmp) >= 1:
        adata2 = ad.AnnData(X=np.zeros((tmp_adata.shape[0], len(tmp))), obs=tmp_adata.obs, var=pd.DataFrame(index=tmp))
        result = ad.concat([adata2, adata1], axis=1)
        result = result[:, geneName]
        result.obs = tmp_adata.obs
    else:
        result = adata1
    sc.pp.calculate_qc_metrics(result, percent_top=None, log1p=False, inplace=True)
    result.obs['total_count'] = result.obs['total_counts']
    sc.pp.normalize_total(result, target_sum=1e4)   ##
    sc.pp.log1p(result)
    result.write_h5ad('filter_hvgall_logNor.h5ad')

##
def f_preData2(DataSet):
    prefixs = ['cell_5000', 'cell_20000', 'cell_50000', 'cell_150000']
    prefix = prefixs[0]
    dirName = '/NFS_home/NFS_home_1/wangyiheng/perturbation_benchmark/scalability/perturbation/genetic'
    os.chdir(dirName)
    ###
    tmp_adata = sc.read_h5ad('filter_hvg5000_logNor_scalability_genetic_change_{}.h5ad'.format(prefix))
    
    geneName = pd.read_csv('/home/wzt/software/scFoundation/OS_scRNA_gene_index.19264.tsv', sep='\t')
    geneName = geneName['gene_name']
    adata1 = tmp_adata[:, tmp_adata.var_names.isin(geneName)]
    tmp = [i for i in geneName if i not in adata1.var_names]
    if len(tmp) >= 1:  #### 如果19264基因中, 存在adata1没有的基因，则全部复制为0
        adata2 = ad.AnnData(X=np.zeros((tmp_adata.shape[0], len(tmp))), obs=tmp_adata.obs, var=pd.DataFrame(index=tmp))
        result = ad.concat([adata2, adata1], axis=1)
        result = result[:, geneName]
        result.obs = tmp_adata.obs
    else:
        result = adata1
    sc.pp.calculate_qc_metrics(result, percent_top=None, log1p=False, inplace=True)
    result.obs['total_count'] = result.obs['total_counts']
    sc.pp.normalize_total(result, target_sum=1e4)   ## 预处理, normalize
    sc.pp.log1p(result)
    result.write_h5ad('filter_hvgall_logNor_scalability_genetic_change_{}.h5ad'.format(prefix))
