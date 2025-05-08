import sys
sys.path.append('/home/project/Pertb_benchmark')
from myUtil import *
import warnings
warnings.filterwarnings('ignore')

### sparsity  评估稀疏性的影响
def toSparsity(DataSet, probability = 0.1):
    os.chdir('/home/project/Pertb_benchmark/DataSet/{}/'.format(DataSet))
    adata = sc.read_h5ad('filter_hvg5000_logNor.h5ad')
    random_matrix = np.random.rand(*adata.shape)
    result_matrix = np.where(random_matrix <= probability, 0, adata.X)
    adata.X = result_matrix
    adata.write_h5ad('filter_hvg5000_logNor_sparisty{}.h5ad'.format(int(probability * 10)))

def f_toSparsity():
    for DataSet in tqdm(["kang_pbmc", "salmonella"]):
        for probability in [0.1, 0.3, 0.5, 0.7, 0.9]:
            toSparsity(DataSet, probability)




### https://github.com/zhaofangyuan98/SDMBench/blob/main/Impact%20Analysis/processing%20code/add_noise.ipynb
def toNoise(DataSet):
    os.chdir('/home/project/Pertb_benchmark/DataSet/{}/'.format(DataSet))
    for noise_level in [0.1, 0.3, 0.5, 0.7, 0.9]:
        adata = sc.read_h5ad('filter_hvg5000_logNor.h5ad')
        noise=np.random.poisson(noise_level * adata.X)
        adata.X= adata.X + noise
        adata.write_h5ad('filter_hvg5000_logNor_noise{}.h5ad'.format(int(noise_level * 10)))


def f_toNoise():
    for DataSet in tqdm(["kangCrossCell"]):
        for noise_level in [0.1, 0.3, 0.5, 0.7, 0.9]:
            toNoise(DataSet, probability)


###  calculate heterogeneity_scores
def calHet(DataSet):
    dirName = '/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet/{}'.format(DataSet)
    os.chdir(dirName)
    filein = 'UmapAnalysis/filter_hvg5000_logNor_control.h5ad'
    adata = sc.read_h5ad(filein)
    X_pca = adata.obsm["X_pca"]
    labels = adata.obs['condition1'].values
    cell_lines = adata.obs['condition1'].unique()
    heterogeneity_scores = {}
    for line in cell_lines:
        indices = adata.obs['condition1'] == line
        X = X_pca[indices]
        center = np.mean(X, axis=0)
        combined = np.vstack([X, center])
        adata_tmp = ad.AnnData(X=combined)
        adata_tmp.obs['perturbation'] = ['treat'] * X.shape[0] + ['control']
        adata_tmp.layers['X'] = adata_tmp.X
        Distance = pt.tools.Distance(metric='edistance',  layer_key='X')
        score = Distance.onesided_distances(adata_tmp, groupby="perturbation", groups=['treat'] , selected_group =  'control')
        heterogeneity_scores[line] = score.values[0]


    inter_scores = {}
    with open('outSample_Sim_pca.pkl', 'rb') as fin:
        mydict = pickle.load(fin)
    for line in cell_lines:
        tmp = [i for i in cell_lines if i != line]
        tmp_value = mydict['edistance'].loc[tmp, line].min()
        inter_scores[line] = tmp_value

    results = pd.DataFrame({
        'cell_line': cell_lines,
        'intra_heterogeneity': [heterogeneity_scores[line] for line in cell_lines],
        'inter_heterogeneity': [inter_scores[line] for line in cell_lines]
    })
    results.to_csv('HetScore_median_edistance.tsv', sep='\t', index=False, float_format='%.3f')


if __name__ == '__main__':
    pass