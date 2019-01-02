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


def drop_nan_inf(data):
    isinfs = np.sum(np.isinf(data.values), axis=0); print('isinfs', isinfs.shape)
    isnans = np.sum(np.isnan(data.values), axis=0); print('isnans', isnans.shape)
    print(np.argwhere(isinfs))
    print(np.argwhere(isnans))
    # data = data.dropna(axis='index')
    inf_cols = data.columns.values[np.squeeze(np.argwhere(isinfs))]
    nan_cols = data.columns.values[np.squeeze(np.argwhere(isnans))]
    print('inf_cols', inf_cols)
    print('nan_cols', nan_cols)
    data.drop(inf_cols, axis=1, inplace=True)
    data.drop(nan_cols, axis=1, inplace=True)
    print(data.shape)
    return data


def drop_var(data, thresh=0.3):
    var = []
    for c in data.columns:
        v = np.var(data[c])
        var.append(v)
    var = np.array(var)
    # var = data.transform(lambda x: np.var(x), axis=0).values
    print('vars:', var.shape)
    lowvar = var < 0.3
    print('Got {} low var columns'.format(np.sum(lowvar)))
    data.drop(data.columns.values[lowvar], axis=1, inplace=True)
    return data