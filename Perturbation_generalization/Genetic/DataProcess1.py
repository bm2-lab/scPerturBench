import sys
sys.path.append('/home/wzt/project/Pertb_benchmark')
from myUtil import *
import pickle

import scanpy as sc
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

####   genetic perturbation
####   Convert the raw data into a standardized format to facilitate subsequent batch data processing scripts.

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


def fun4(adata):
    with open('/home/wzt/project/GW_PerturbSeq/geneEmbedding/geneList.pkl', 'rb') as fin:
        geneList = pickle.load(fin)
    tmp = adata.var_names.isin(geneList)
    adata = adata[:, tmp]
    adata1 = fun2(adata)
    adata1.write_h5ad('raw.h5ad')

'''
adata = sc.read_h5ad('raw.h5ad')
[i for i in adata.obs['gene'].unique() if i not in geneList]
[i for i in adata.var_names if i not in geneList]
[i for i in adata.obs['gene'].unique() if i not in adata.var_names]
'''

def myfun1(x):
    xs = x.split(',') 
    xs = sorted(xs)
    return '+'.join(xs)

###  ***************  combination   *************
### Exploring genetic interaction manifolds constructed from rich single-cell phenotypes
def pre_Norman():  #### K562  activation, 
    os.chdir('/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/Norman')
    adata = sc.read_h5ad('Raw.h5ad')
    adata.obs['perturbation'] =  adata.obs['gene'].apply(lambda x: myfun1(x))  ### 
    adata.obs['perturbation'].replace({'CTRL': 'control'}, inplace=True)
    adata.write_h5ad('raw.h5ad')

### PRJNA787633
### CRISPR activation and interference screens decode stimulation responses in primary human T cells   
def pre_Schmidt():  #### K562  activation,
    os.chdir('/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/Schmidt')
    adata = sc.read_h5ad('Raw.h5ad')
    adata.obs['perturbation'] =  adata.obs['gene'].apply(lambda x: myfun1(x))  ###
    adata.obs['perturbation'].replace({'CTRL': 'control'}, inplace=True)
    adata.write_h5ad('raw.h5ad')

### Combinatorial single-cell CRISPR screens by direct guide RNA capture and targeted sequencing
### PRJNA609688
def pre_Replogle_exp6():
    os.chdir('/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/Replogle_exp6')
    adata = sc.read_h5ad('Raw.h5ad')
    adata.obs['perturbation'] =  adata.obs['gene'].apply(lambda x: myfun1(x))  ### 
    adata.obs['perturbation'].replace({'CTRL': 'control'}, inplace=True)
    adata.write_h5ad('raw.h5ad')

def pre_Replogle_exp9():
    os.chdir('/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/Replogle_exp9')
    adata = sc.read_h5ad('Raw.h5ad')
    adata.obs['perturbation'] =  adata.obs['gene'].apply(lambda x: myfun1(x))  ### 
    adata.obs['perturbation'].replace({'CTRL': 'control'}, inplace=True)
    adata.write_h5ad('raw.h5ad')

def pre_Replogle_exp10():
    os.chdir('/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/Replogle_exp10')
    adata = sc.read_h5ad('Raw.h5ad')
    adata.obs['perturbation'] =  adata.obs['gene'].apply(lambda x: myfun1(x))  ###
    adata.obs['perturbation'].replace({'CTRL': 'control'}, inplace=True)
    adata.write_h5ad('raw.h5ad')


def pre_Wessels():
    os.chdir('/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/Wessels')
    adata = sc.read_h5ad('Raw.h5ad')
    adata.obs['perturbation'] =  adata.obs['gene'].apply(lambda x: myfun1(x))  ##
    adata.obs['perturbation'].replace({'CTRL': 'control'}, inplace=True)
    adata.write_h5ad('raw.h5ad')




#### ***************  single perturbation  *************
###  A multiplexed single-cell CRISPR screening platform enables systematic dissection of the unfolded protein response  
def pre_Adamson():
    os.chdir('/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/Adamson')
    adata = sc.read_h5ad('Raw.h5ad')
    adata.var_names_make_unique()  ##
    adata.obs['perturbation'] = adata.obs['gene']
    adata.obs['perturbation'].replace({'CTRL': 'control'}, inplace=True)
    fun4(adata)


### Multimodal pooled Perturb-CITE-seq screens in patient models define mechanisms of cancer immune evasion
### Perturb-CITE-seq  
def pre_Frangieh():
    os.chdir('/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/Frangieh')
    adata = sc.read_h5ad('Raw.h5ad')
    adata.obs['perturbation'] =  adata.obs['gene'].apply(lambda x: myfun1(x))
    adata.obs['perturbation'].replace({'CTRL': 'control'}, inplace=True)
    adata.write_h5ad('raw.h5ad')

### PRJNA641125a
#### Genome-wide CRISPRi/a screens in human neurons link lysosomal failure to ferroptosis
def pre_TianActivation():
    os.chdir('/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/TianActivation')
    adata = sc.read_h5ad('Raw.h5ad')
    adata.obs['perturbation'] =  adata.obs['gene'].apply(lambda x: myfun1(x))
    adata.obs['perturbation'].replace({'CTRL': 'control'}, inplace=True)
    adata.write_h5ad('raw.h5ad')
## PRJNA641125i
def pre_TianInhibition():
    os.chdir('/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/TianInhibition')
    adata = sc.read_h5ad('Raw.h5ad')
    adata.obs['perturbation'] =  adata.obs['gene'].apply(lambda x: myfun1(x))
    adata.obs['perturbation'].replace({'CTRL': 'control'}, inplace=True)
    adata.write_h5ad('raw.h5ad')

### Combinatorial single-cell CRISPR screens by direct guide RNA capture and targeted sequencing
### PRJNA609688
def pre_Replogle_exp7():
    os.chdir('/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/Replogle_exp7')
    adata = sc.read_h5ad('Raw.h5ad')
    adata.obs['perturbation'] =  adata.obs['gene'].apply(lambda x: myfun1(x))
    adata.obs['perturbation'].replace({'CTRL': 'control'}, inplace=True)
    adata.write_h5ad('raw.h5ad')

def pre_Replogle_exp8():
    os.chdir('/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/Replogle_exp8')
    adata = sc.read_h5ad('Raw.h5ad')
    adata.obs['perturbation'] =  adata.obs['gene'].apply(lambda x: myfun1(x))
    adata.obs['perturbation'].replace({'CTRL': 'control'}, inplace=True)
    adata.write_h5ad('raw.h5ad')


### Characterizing the molecular regulation of inhibitory immune checkpoints with multimodal single-cell screens
###  PRJNA641353
def pre_Papalexi():
    os.chdir('/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/Papalexi')
    adata = sc.read_h5ad('Raw.h5ad')
    adata.obs['perturbation'] =  adata.obs['gene'].apply(lambda x: myfun1(x))
    adata.obs['perturbation'].replace({'CTRL': 'control'}, inplace=True)
    adata.write_h5ad('raw.h5ad')


####  Replogle_K562essential
def pre_Replogle_K562essential():
    os.chdir('/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/Replogle_K562essential')
    adata = sc.read_h5ad('Raw.h5ad')
    adata.obs['perturbation'] =  adata.obs['gene'].apply(lambda x: myfun1(x))
    adata.obs['perturbation'].replace({'CTRL': 'control'}, inplace=True)
    adata.write_h5ad('raw.h5ad')

def pre_Replogle_RPE1essential():
    os.chdir('/NFS2_home/NFS2_home_1/wzt/project/Pertb_benchmark/DataSet2/Replogle_RPE1essential')
    adata = sc.read_h5ad('Raw.h5ad')
    adata.obs['perturbation'] =  adata.obs['gene'].apply(lambda x: myfun1(x))
    adata.obs['perturbation'].replace({'CTRL': 'control'}, inplace=True)
    adata.write_h5ad('raw.h5ad')

