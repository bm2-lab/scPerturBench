from myUtil import *





'''
Convert the raw data into a standardized format to facilitate subsequent batch data processing scripts.
'''


### sciplex3
def do_Sciplex3():
    os.chdir('/home/wzt/project/Pertb_benchmark/DataSet/sciplex3')
    adata = sc.read_h5ad('SrivatsanTrapnell2020_sciplex3.h5ad')
    adata1 = adata[~((adata.obs['cell_line'].isna()) | (adata.obs['time'].isna()))] ## drop NA data
    adata1.write_h5ad('raw.h5ad')
    adata = sc.read_h5ad('raw.h5ad')
    adata.var.set_index('ensembl_id', inplace=True)
    adata.var.index.name = 'ENSEMBL'
    adata.var_names = list(adata.var_names)
    adata.var_names_make_unique()
    adata = adata[:, 1:]
    adata = transID(adata, 'Hsa')
    adata.var_names_make_unique()
    adata.write_h5ad('raw.h5ad')


#### kang   pbmc
def do_PBMC():
    os.chdir('/home/wzt/project/Pertb_benchmark/DataSet/kang_pbmc')
    adata = sc.read_h5ad('kang_count.h5ad')
    adata = transID(adata, 'Hsa')
    adata.obs['perturbation'] = adata.obs['condition']
    adata.write_h5ad('raw.h5ad')

#### haber   
def do_haber():
    os.chdir('/home/wzt/project/Pertb_benchmark/DataSet/haber')
    adata = sc.read_h5ad('haber_count.h5ad')
    adata = transID(adata, 'Mmu')
    adata.obs['perturbation'] = adata.obs['condition']
    adata.obs['cell_type'] = adata.obs['cell_label']
    adata.obs['perturbation'].replace('Control', 'control', inplace=True)
    adata.write_h5ad('raw.h5ad')
    
#### salmonella
def do_salmonella():
    os.chdir('/home/wzt/project/Pertb_benchmark/DataSet/salmonella')
    dat=  pd.read_csv('GSE92332_SalmonellaInfect_UMIcounts.txt', sep='\t')
    myobs = [i.split('_')[0] for i in dat.columns]
    adata = ad.AnnData(X=sparse.csr_matrix(dat.values.T), obs=pd.DataFrame(index=myobs), var=pd.DataFrame(index=dat.index))
    adata.obs['cell_type'] = [i.split('_')[3] for i in dat.columns]
    adata.obs['cell_type'] = adata.obs['cell_type'].apply(lambda x: x.replace('.', ''))
    adata.obs['perturbation'] = [i.split('_')[1] for i in dat.columns]
    adata.obs['perturbation'].replace('Control', 'control', inplace=True)
    adata.write_h5ad('raw.h5ad')

### cross species
def do_species(species='mouse'):
    os.chdir('/home/wzt/project/Pertb_benchmark/DataSet/crossSpecies')
    adataList = []
    files = glob.glob('/home/wzt/software/scGen-reproducibility/datasets/crossSpecies/LPS/{}*'.format(species))
    for file in files:
        dat = pd.read_csv(file, sep=' ', index_col=0)
        adata = ad.AnnData(X=sparse.csr_matrix(dat.values.T), obs=pd.DataFrame(index=dat.columns), var=pd.DataFrame(index=dat.index))
        adata.obs['species'] = os.path.basename(file).split('_')[0]
        adata.obs['perturbation'] = os.path.basename(file).split('_')[1]
        adataList.append(adata)
    adata_r = ad.concat(adataList)
    adata_r = transID(adata_r, 'mouse')
    adata_r.write_h5ad('{}_raw.h5ad'.format(species))

    ### 
    adata1 = sc.read_h5ad('mouse_raw.h5ad'); adata2 = sc.read_h5ad('pig_raw.h5ad')
    adata3 = sc.read_h5ad('rabbit_raw.h5ad'); adata4 = sc.read_h5ad('rat_raw.h5ad')
    adata1.var_names = [i.upper() for i in adata1.var_names]; adata1.var_names_make_unique();adata1.obs_names_make_unique() 
    adata2.var_names = [i.upper() for i in adata2.var_names]; adata2.var_names_make_unique();adata2.obs_names_make_unique()
    adata3.var_names = [i.upper() for i in adata3.var_names]; adata3.var_names_make_unique();adata3.obs_names_make_unique()
    adata4.var_names = [i.upper() for i in adata4.var_names]; adata4.var_names_make_unique();adata4.obs_names_make_unique()
    a = np.intersect1d(adata1.var_names, adata2.var_names)
    b = np.intersect1d(a, adata3.var_names)
    c = np.intersect1d(b, adata4.var_names)
    adata1 = adata1[:, c]; adata2 = adata2[:, c]; adata3 = adata3[:, c]; adata4 = adata4[:, c]
    adata_r = ad.concat([adata1, adata2, adata3, adata4])
    adata_r.write_h5ad('raw.h5ad')



### cross patient
def do_patient():
    os.chdir('/home/wzt/project/Pertb_benchmark/DataSet/crossPatient')
    adataList = []
    files = glob.glob('/home/wzt/data/crossPatient/*cts.txt')
    metaData = pd.read_csv('/home/wzt/data/crossPatient/SraRunTable.txt', sep=',')
    mydict = metaData[['GEO_Accession (exp)', 'treatment']].set_index('GEO_Accession (exp)').to_dict()['treatment']
    for file in files:
        GSM_ID = os.path.basename(file).split('_')[0]
        if mydict[GSM_ID] in ['vehicle (DMSO)', '2.5 uM etoposide', '0.2 uM panobinostat']:
            dat = pd.read_csv(file, sep='\t', index_col=1)
            dat = dat.iloc[:,1:]
            adata = ad.AnnData(X=sparse.csr_matrix(dat.values.T), obs=pd.DataFrame(index=dat.columns), var=pd.DataFrame(index=dat.index))
            adata.var_names_make_unique()
            adata.obs['GSM_ID'] = GSM_ID
            adata.obs['patient'] = os.path.basename(file)[11:16]
            adata.obs['patientBatch'] = os.path.basename(file)[17:20]
            adataList.append(adata)
    adata_r = ad.concat(adataList)
    adata_r.obs['treatment'] = adata_r.obs['GSM_ID'].apply(lambda x: mydict[x])
    adata_r.write_h5ad('raw.h5ad')

### kang new
def do_kangNew():
    os.chdir('/home/wzt/project/Pertb_benchmark/DataSet/kangCrossPatient')
    adata1 = sc.read_h5ad('/home/wzt/software/cellot/datasets/scrna-lupuspatients/kang_new.h5ad')
    metaData = adata1.obs
    adata2 = sc.read_h5ad('/home/wzt/project/Pertb_benchmark/DataSet/kang_pbmc/raw.h5ad')
    adata2.obs_names = [i.replace('.', '-') for i in adata2.obs_names]
    df, adata_obs = metaData.align(adata2.obs, join="inner", axis=0)
    adata3 = adata2[adata_obs.index, :]
    adata3.obs = adata3.obs.merge(df, left_index=True, right_index=True)
    adata3 = adata3[adata3.obs['multiplets'] == 'singlet']
    adata3.write_h5ad('raw.h5ad')


#### multiDose DataSet
def do_TCDD():
    os.chdir('/home/wzt/project/Pertb_benchmark/DataSet/TCDD')
    adata = sc.read_h5ad('nault2021_multiDose.h5ad')
    adata.obs['celltype'] = adata.obs['celltype'].astype(str)
    KeepCelltype = ['Cholangiocytes', 'Endothelial Cells', 'Hepatocytes - portal', 
                    'Hepatocytes - central', 'Portal Fibroblasts', 'Stellate Cells']
    adata = adata[adata.obs['celltype'].isin(KeepCelltype)]
    adata.obs['perturbation'] = adata.obs['Dose'].apply(lambda x: 'control' if x == 0 else 'TCDD')
    mydict = {'Stellate Cells': 'StellateCells', 'Hepatocytes - portal': 'HepatocytesPortal', 
              'Endothelial Cells': 'EndothelialCells', 'Hepatocytes - central': 'HepatocytesCentral', 
              'Portal Fibroblasts': 'PortalFibroblasts'}
    adata.obs['celltype'].replace(mydict, inplace=True)
    adata.write_h5ad('raw.h5ad')


def crossPatient_kaggle():
    os.chdir('/home/wzt/project/Pertb_benchmark/DataSet/crossPatientKaggle')
    dat = pd.read_csv('/home/wzt/software/scPRAM/data/adata_obs_meta.csv', sep=',', index_col=0)
    dat = dat[dat['sm_name'] != 'Belinostat']; dat = dat[dat['sm_name'] != 'Dabrafenib']
    dat['sm_name'].replace({'Dimethyl Sulfoxide': 'control'}, inplace=True)
    dat['perturbation'] = dat['sm_name'].apply(lambda x: x.split()[0].replace('-', ''))
    dat['perturbation'].replace({'5(9Isopropyl8methyl2morpholino9Hpurin6yl)pyrimidin2amine': 'Pyrimidin'}, inplace=True)
    dat['donor_id'].replace({'donor_0': 'Pat0', 'donor_1': 'Pat1', 'donor_2': 'Pat2'}, inplace=True)
    dat['cell_type'].replace({'T cells CD4+': 'CD4', 'NK cells': 'NK', 'T cells CD8+': 'CD8', 'Myeloid cells': 'Myeloid', 
                              'T regulatory cells': 'Treg', 'B cells': 'Bcells'}, inplace=True)
    
    adata_par = pd.read_parquet('/home/wzt/software/scPRAM/data/adata_train.parquet')
    adata1 = adata_par.pivot(index='obs_id', columns='gene', values='count')
    adata1.fillna(0, inplace=True)
    adata = ad.AnnData(X=sparse.csr_matrix(adata1.values), obs=pd.DataFrame(index=adata1.index), var=pd.DataFrame(index=adata1.columns))

    df, adata_obs = dat.align(adata.obs, join="inner", axis=0)
    adata1 = adata[adata_obs.index, :]
    adata1.obs = adata1.obs.merge(df, left_index=True, right_index=True)
    adata1.write_h5ad('raw.h5ad')

    
def doMcFarland():
    os.chdir('/home/wzt/project/Pertb_benchmark/DataSet/McFarland')
    dat = sc.read_h5ad('McFarlandTsherniak2020.h5ad')
    dat1 = dat[dat.obs['cell_quality'] == 'normal']
    dat1 = dat1[~dat1.obs['perturbation'].isin(['AZD5591', 'BRD3379', 'sgLACZ', 'sgOR2J2', 'sgGPX4-2', 'sgGPX4-1'])]
    keept_cellLine = list(dat1.obs['cell_line'].value_counts()[:5].index)  ### subset dataset to reduce run time
    dat1 = dat1[dat1.obs['cell_line'].isin(keept_cellLine)]
    dat1.write_h5ad('raw.h5ad')
