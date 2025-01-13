from myUtil import *
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


#### iid  split the data into train  validation  test

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



def split_cell_data_train_test(
    adata, groupby=None, random_state=0, **kwargs
):

    split = pd.Series(None, index=adata.obs.index, dtype=object)
    groups = {None: adata.obs.index}
    if groupby is not None:
        groups = adata.obs.groupby(groupby).groups

    for key, index in groups.items():
        trainobs, testobs = train_test_split(index, random_state=random_state, test_size=.5, **kwargs) ### 一半用来训练   train用来训练，test用来test
        split.loc[trainobs] = "train"
        split.loc[testobs] = "test"

    return split


def do_Split():    
    for DataSet in tqdm(DataSets):
        DataSet = '/home/wzt/project/Pertb_benchmark/DataSet/{}'.format(DataSet)
        if os.path.isdir(DataSet):
            os.chdir(DataSet)
            adata = sc.read_h5ad('filter.h5ad')
            if 'train_val_test' in adata.obs.columns:
                del adata.obs['train_val_test']
            split = split_cell_data_train_test(adata, groupby=['condition1', 'condition2'], random_state=0)
            adata.obs['iid_test'] = split
            adata.write_h5ad('filter.h5ad')


DataSets = ["kang_pbmc", "kangCrossPatient", "salmonella", "PRJNA419230", "haber", "crossPatient", "crossPatientKaggle",
            "crossPatientKaggleCell", "crossSpecies", "McFarland", "Afriat", "TCDD", "sciplex3"]

if __name__ == '__main__':
    do_Split()
