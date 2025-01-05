from myUtil import *
import warnings
warnings.filterwarnings('ignore')


'''

Convert the raw data into a standardized format to facilitate subsequent batch data processing scripts.

'''


def reindexCol(adata1, cols_to_move):
    adata1.obs = adata1.obs[[col for col in adata1.obs.columns if col not in cols_to_move] + cols_to_move]
    return adata1



def delCells(adata1, threshold=10):
    tmp = pd.crosstab(adata1.obs['condition1'], adata1.obs['condition2'])
    for cond1 in tmp.index:
        for cond2 in tmp.columns:
            if tmp.loc[cond1, cond2] < threshold:
                adata1 = adata1[~((adata1.obs['condition1'] == cond1) & (adata1.obs['condition2'] == cond2))].copy()
    return adata1


### sciplex3
def do_Sciplex3():
    os.chdir('/home/wzt/project/Pertb_benchmark/DataSet/sciplex3')
    adata = sc.read_h5ad('raw.h5ad')
    adata.obs['perturbation'] = adata.obs['perturbation'].apply(lambda x: x.split()[0].replace('-', ''))
    adata.obs['perturbation'].replace({'Glesatinib?(MGCD265)': 'Glesatinib', '(+)JQ1': 'JQ1'}, inplace=True)
    adata = adata[adata.obs['time'] == 24]
    adata.obs['perturbation1'] = adata.obs['perturbation']
    tmp = []
    for i, j in zip(adata.obs['dose_value'], adata.obs['perturbation']):
        if j != 'control': tmp.append(j+'_' + str(i))
        else: tmp.append(j)
    adata.obs['perturbation'] = tmp
    *_, adata1 = preData(adata, mtpercent = 15)
    adata1.X = adata1.layers['logNor'].copy()
    adata1.obs['cell_type'] = adata1.obs['cell_line']
    adata1.obs['condition1'] = adata1.obs['cell_type']
    adata1.obs['condition2'] = adata1.obs['perturbation1']
    adata1.obs['dose'] = adata1.obs['dose_value']
        
    adata1.obs['batch'] = adata1.obs['plate']
    adata1.write_h5ad('filter1.h5ad')

    test_pertb = ['control', 'Dacinostat',   'Givinostat', 'Belinostat',  'Hesperadin', 'Quisinostat',  'Alvespimycin',  'Tanespimycin',
    'TAK901', 'Flavopiridol']
    adata1 = adata1[adata1.obs['condition2'].isin(test_pertb)]
    adata1.write_h5ad('filter.h5ad')

    # test_pertb = [
    # 'control',
    # 'Dacinostat_1000.0',    'Dacinostat_10.0',    'Dacinostat_10000.0',    'Dacinostat_100.0',    'Givinostat_1000.0',
    # 'Givinostat_10.0',    'Givinostat_10000.0',    'Givinostat_100.0',    'Belinostat_1000.0',    'Belinostat_10.0',
    # 'Belinostat_10000.0',    'Belinostat_100.0',    'Hesperadin_1000.0',    'Hesperadin_10.0',    'Hesperadin_10000.0',
    # 'Hesperadin_100.0',    'Quisinostat_1000.0',    'Quisinostat_10.0',    'Quisinostat_10000.0',    'Quisinostat_100.0',
    # 'Alvespimycin_1000.0',    'Alvespimycin_10.0',    'Alvespimycin_10000.0',    'Alvespimycin_100.0',    'Tanespimycin_1000.0',
    # 'Tanespimycin_10.0',    'Tanespimycin_10000.0',    'Tanespimycin_100.0',    'TAK901_1000.0',    'TAK901_10.0',
    # 'TAK901_10000.0',    'TAK901_100.0',    'Flavopiridol_1000.0',    'Flavopiridol_10.0',    'Flavopiridol_10000.0',    'Flavopiridol_100.0',
    # ]
'''
    test_pertb = ['control', 'Dacinostat',   'Givinostat', 'Belinostat',  'Hesperadin', 'Quisinostat',  'Alvespimycin',  'Tanespimycin',
    'TAK901', 'Flavopiridol']

    fileins  = ['filter_hvg500_logNor_all.h5ad',    'filter_hvg1000_logNor_all.h5ad',    'filter_hvg2000_logNor_all.h5ad',    'filter_hvg5000_logNor_all.h5ad']
    fileouts = ['filter_hvg500_logNor_subset.h5ad', 'filter_hvg1000_logNor_subset.h5ad', 'filter_hvg2000_logNor_subset.h5ad', 'filter_hvg5000_logNor_subset.h5ad']
    
    os.chdir('/home/wzt/project/Pertb_benchmark/DataSet/sciplex3')
    for filein, fileout in zip(fileins, fileouts):
        adata = sc.read_h5ad(filein)
        adata1 = adata[adata.obs['condition2'].isin(test_pertb)]
        adata1.write_h5ad(fileout)
'''



#### kang   pbmc
def do_PBMC():
    os.chdir('/home/wzt/project/Pertb_benchmark/DataSet/kang_pbmc')
    adata = sc.read_h5ad('raw.h5ad')
    *_, adata1 = preData(adata)
    adata1.obs['seurat_annotations'] = adata1.obs['seurat_annotations'].apply(lambda x: x.replace(' ', '_'))
    adata1.obs['cell_type'] = adata1.obs['cell_type'].apply(lambda x: x.replace(' ', '_'))
    adata1.X = adata1.layers['logNor'].copy()
    adata1.obs['condition1'] = adata1.obs['cell_type']
    adata1.obs['condition2'] = adata1.obs['perturbation']
    adata1.write_h5ad('filter.h5ad')

    
#### haber
def do_haber():
    os.chdir('/home/wzt/project/Pertb_benchmark/DataSet/haber')
    adata = sc.read_h5ad('raw.h5ad')
    *_, adata1 = preData(adata)
    adata1.obs['cell_type'] = adata1.obs['cell_type'].apply(lambda x: x.replace('.', '_'))
    adata1.obs['perturbation'] = adata1.obs['perturbation'].apply(lambda x: x.replace('.', '_'))
    adata1.X = adata1.layers['logNor'].copy()
    adata1.obs['condition1'] = adata1.obs['cell_type']
    adata1.obs['condition2'] = adata1.obs['perturbation']
    adata1 = reindexCol(adata1, ['condition1', 'condition2', 'batch'])
    adata1.write_h5ad('filter.h5ad')

### salmonella
def do_salmonella():
    os.chdir('/home/wzt/project/Pertb_benchmark/DataSet/salmonella')
    adata = sc.read_h5ad('raw.h5ad')
    *_, adata1 = preData(adata)
    adata1.X = adata1.layers['logNor'].copy()
    adata1.obs['condition1'] = adata1.obs['cell_type']
    adata1.obs['condition2'] = adata1.obs['perturbation']
    adata1.write_h5ad('filter.h5ad')

#### cross species
def do_species():
    os.chdir('/home/wzt/project/Pertb_benchmark/DataSet/crossSpecies')
    adata = sc.read_h5ad('raw.h5ad')
    adata.obs['condition1'] = adata.obs['species'].apply(lambda x: x[:-1])
    adata.obs['perturbation'].replace('unst', 'control', inplace=True)
    adata.obs['condition2'] = adata.obs['perturbation'].apply(lambda x: x[:-1] if x!= 'control' else x)
    adata.obs['condition3'] = adata.obs['perturbation'].apply(lambda x: x[-1] if x!= 'control' else 0)
    adata.obs['condition3'] = adata.obs['condition3'].astype(int)
    #adata.obs['condition3'] = adata.obs['species']
    adata.obs['condition2'].replace('lps', 'LPS', inplace=True)
    *_, adata1 = preData(adata)
    adata1.X = adata1.layers['logNor'].copy()
    adata1 = reindexCol(adata1, ['condition1', 'condition2', 'condition3'])
    adata1.write_h5ad('filter.h5ad')

### cross patient
def do_patient():
    os.chdir('/home/wzt/project/Pertb_benchmark/DataSet/crossPatient')
    adata = sc.read_h5ad('raw.h5ad')
    mydict = {'2.5 uM etoposide': 'Etoposide', 'vehicle (DMSO)': 'control', '0.2 uM panobinostat': 'Panobinostat'}
    adata.obs['treatment'].replace(mydict, inplace=True)
    adata.obs['perturbation'] = adata.obs['treatment']
    *_, adata1 = preData(adata)
    adata1.X = adata1.layers['logNor'].copy()
    adata1.obs['condition1'] = adata1.obs['patient']
    adata1.obs['condition2'] = adata1.obs['perturbation']
    adata1.obs['batch'] = adata1.obs['patientBatch']
    adata1.write_h5ad('filter.h5ad')

### cross kang 
def do_patient1():
    os.chdir('/home/wzt/project/Pertb_benchmark/DataSet/kangCrossPatient')
    adata = sc.read_h5ad('raw.h5ad')
    *_, adata1 = preData(adata)
    adata1.X = adata1.layers['logNor'].copy()
    adata1.obs['condition1'] = 'Pat' + adata1.obs['sample_id'].astype(str)
    adata1.obs['condition2'] = adata1.obs['perturbation']
    #adata1.obs['condition3'] = adata.obs['cell_type_x'].apply(lambda x: x.replace(' ', ''))
    adata1.write_h5ad('filter.h5ad')

#### multiDose DataSet
def do_TCDD():
    os.chdir('/home/wzt/project/Pertb_benchmark/DataSet/TCDD')
    adata = sc.read_h5ad('raw.h5ad')
    *_, adata1 = preData(adata)
    adata1.X = adata1.layers['logNor'].copy()
    mylist = []
    for index, i in adata.obs.iterrows():
        if i['perturbation'] == 'control': mylist.append('control')
        else: mylist.append(i['perturbation'] + '_' + str(i['Dose']))
    
    adata1.obs['condition1'] = adata1.obs['celltype']
    adata1.obs['condition2'] = adata1.obs['perturbation']
    adata1.obs['batch'] = adata1.obs['batch'].astype(str)
    adata1.obs['dose'] = adata1.obs['Dose']
    adata1 = reindexCol(adata1, ['condition1', 'condition2', 'batch', 'dose'])
    adata1.write_h5ad('filter.h5ad')

### cross patient kaggle
def crossPatient_kaggle():
    os.chdir('/home/wzt/project/Pertb_benchmark/DataSet/crossPatientKaggle')
    adata = sc.read_h5ad('raw.h5ad')
    *_, adata1 = preData(adata)
    adata1.X = adata1.layers['logNor'].copy()
    adata1.obs['condition1'] = adata1.obs['donor_id']
    adata1.obs['condition2'] = adata1.obs['perturbation']
    adata1.obs['condition3'] = adata1.obs['cell_type']
    adata1.write_h5ad('filter1.h5ad')

### cross patient kaggle
### conda activate pertpyV7
def crossPatient_kaggle1():
    os.chdir('/home/wzt/project/Pertb_benchmark/DataSet/crossPatientKaggle')
    if not os.path.isfile('perturbDistance.tsv'):
        adata = sc.read_h5ad('filter1.h5ad')
        adata.uns['log1p']["base"] = None
        sc.pp.highly_variable_genes(adata, n_top_genes=5000)
        sc.tl.pca(adata)
        import pertpy as pt  # type: ignore
        from tqdm import tqdm
        mydict = {}
        distance = pt.tl.Distance('edistance', 'X_pca')
        pertGenes = adata.obs['condition2'].unique()
        X = adata.obsm['X_pca'][adata.obs['condition2'] == 'control']
        for pertGene in tqdm(pertGenes):
            if pertGene == 'control': continue
            Y = adata.obsm['X_pca'][adata.obs['condition2'] == pertGene]
            result = distance(X, Y)
            mydict[pertGene] = result
        dat = pd.DataFrame({'perturbation': mydict.keys(), 'distance': mydict.values()})
        dat.sort_values(by='distance', ascending=False, inplace=True)
        dat.to_csv('perturbDistance.tsv', sep='\t', index=False)
    dat = pd.read_csv('perturbDistance.tsv', sep='\t')
    adata = sc.read_h5ad('filter1.h5ad')
    keept = list(dat['perturbation'][:5]) + list(dat['perturbation'][-5:]) + ['control']
    adata1 = adata[adata.obs['condition2'].isin(keept)]
    adata1.obs['batch'] = adata1.obs['plate_name']
    adata1.write_h5ad('filter.h5ad')

    adata = sc.read_h5ad('filter.h5ad')
    del adata.obs['condition3']
    adata.write_h5ad('filter.h5ad')


### 进行cell line transfer
def crossPatient_kaggleCell():
    os.chdir('/home/wzt/project/Pertb_benchmark/DataSet/crossPatientKaggleCell')
    # adata = sc.read_h5ad('../crossPatientKaggle/filter.h5ad')
    # adata.obs.rename(columns={"condition1": "condition3", "condition3":"condition1"}, inplace=True)
    # adata1 = reindexCol(adata, ['condition1', 'condition2', 'condition3', 'batch'])
    # adata1.write_h5ad('filter.h5ad')

    adata = sc.read_h5ad('filter1.h5ad')
    adata.obs['condition1'] = adata.obs['cell_type']
    adata = reindexCol(adata, ['condition1', 'condition2', 'batch'])
    adata = delCells(adata, threshold=100)
    adata = adata[adata.obs['condition1'] != 'Bcells']
    adata.write_h5ad('filter.h5ad')


def doMcFarland():
    os.chdir('/home/wzt/project/Pertb_benchmark/DataSet/McFarland')
    adata = sc.read_h5ad('raw.h5ad')
    adata = adata[adata.obs['time'].isin(["6", "24"])]
    adata.obs['time'] = adata.obs['time'].astype(int)
    *_, adata1 = preData(adata)
    adata1.X = adata1.layers['logNor'].copy()

    adata1.obs['condition1'] = adata1.obs['cell_line']
    adata1.obs['condition2'] = adata1.obs['perturbation']
    adata1.obs['condition3'] = adata1.obs['time']
    adata1.write_h5ad('filter.h5ad')

    adata1 = sc.read_h5ad('filter.h5ad')
    adata1 = delCells(adata1, threshold=20)
    adata1.write_h5ad('filter.h5ad')

def doPRJNA419230():
    os.chdir('/home/wzt/project/Pertb_benchmark/DataSet/PRJNA419230')
    adata = sc.read_h5ad('raw.h5ad')
    adata.obs['perturbation'] = adata.obs['gene']
    adata.obs['condition1'] = adata.obs['media']
    adata.obs['condition2'] = adata.obs['gene']
    *_, adata1 = preData(adata, mtpercent=10)
    geneList = list(adata1.obs['condition2'].value_counts().index[:11])
    adata1 = adata1[adata1.obs['condition2'].isin(geneList)].copy()
    adata1.obs['condition2'].replace({'CTRL': 'control'}, inplace=True)
    del adata1.obs['batch']
    adata1 = reindexCol(adata1, cols_to_move=['condition1', 'condition2'])
    adata1.write_h5ad('filter.h5ad')


def doAfriat():
    os.chdir('/home/wzt/project/Pertb_benchmark/DataSet/Afriat')
    adata = sc.read_h5ad('adata_infected.h5ad')
    adata.obs['condition1'] = adata.obs['zone']
    adata.obs['condition2'] = adata.obs['status_control'].replace({'Control': 'control'})
    adata.obs['condition3'] = adata.obs['time_int'].astype(int)
    adata.obs['perturbation'] = adata.obs['condition2']
    adata = reindexCol(adata, ['condition1', 'condition2', 'condition3'])

    tmp = adata.var.sort_values('highly_variable_rank').index
    adata1 = adata[:, tmp]
    adata1.var['highly_variable_500'] = [True] * 500 + [False] * (adata1.shape[1] - 500)
    adata1.var['highly_variable_1000'] = [True] * 1000 + [False] * (adata1.shape[1] - 1000)
    adata1.var['highly_variable_2000'] = [True] * 2000 + [False] * (adata1.shape[1] - 2000)
    adata1.var['highly_variable_5000'] = [True] * 5000 + [False] * (adata1.shape[1] - 5000)
    adata1.write_h5ad('filter.h5ad')
