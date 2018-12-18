import pandas as pd
import numpy as np
from scipy.stats import spearmanr

def drop_high_cor(dm, cor_thresh=0.95):
    """Drop dataframe columns that have correlation > threshold
    
    Args:
        dm (pd.DataFrame): feature values
        cor_thresh (float): drop columns with correlation > that threshold
        
    Returns:
        pd.DataFrame that dropped cor columns
    """
    # Calculate spearman correlation
    sp_cor, sp_corp = spearmanr(dm)
    
    ## Determine columns with correlation > threshold using diagonal matrix
    cor1 = pd.DataFrame(sp_cor, columns=dm.columns, index=dm.columns)

    # Select upper triangle of correlation matrix 
    upper = cor1.where(np.triu(np.ones(cor1.shape), k=1).astype(np.bool))
    
    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if 
               any(np.abs(upper[column]) > cor_thresh)]
    print("Drop Col Number: {:d}".format(len(to_drop)))
    
    # Drop selected columns
    dm1 = dm.copy()
    dm1.drop(to_drop, axis=1, inplace=True)
    
    return dm1

def load_features(src, zscore=False):
    f = pd.read_csv(src, sep=',', index_col=0, header=0)
    case_ids = f['case_id']

    f = f.drop('case_id', axis=1)
    # z-score them
    if zscore:
        f = f.transform(lambda x: (x - np.mean(x)) / np.std(x))
    print('Features:')
    print(f.head())

    return f, case_ids

def load_labels(src):
    f = pd.read_csv(src, sep='\t', index_col=0, header=0)
    print('Labels:')
    print(f.head())

    return f