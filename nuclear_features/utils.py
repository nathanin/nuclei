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

    # z-score them
    if zscore:
        f = f.transform(lambda x: (x - np.mean(x)) / np.std(x))
    return f

def load_labels(src):
    f = pd.read_csv(src, sep='\t', index_col=0, header=0)

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

nepc_strs = ['NEPC']
adeno_strs = ['M0 NP', 'M0 oligo poly', 'M0 oligo', 'M0 poly', 'M1 oligo poly',
              'M1 oligo', 'M1 poly', 'MX Diffuse', 'NXMX P']
m0_strs = ['M0 NP']
m0p_strs = ['M0 oligo poly', 'M0 oligo', 'M0 poly']
m1_strs = ['M1 oligo poly', 'M1 oligo', 'M1 poly']
def split_sets(feat, lab):
  """
  Return a tuple:
  ((nepc_f, nepc_lab), (m0_f, m0_lab),... )
  """
  is_nepc = np.array([x in nepc_strs for x in lab['stage_str']])
  is_m0 = np.array([x in m0_strs for x in lab['stage_str']])
  is_m0p = np.array([x in m0p_strs for x in lab['stage_str']])
  is_m1 = np.array([x in m1_strs for x in lab['stage_str']])

  nepc_f = feat.loc[is_nepc, :]; nepc_lab = lab.loc[is_nepc, :]
  m0_f = feat.loc[is_m0, :]; m0_lab = lab.loc[is_m0, :]
  m0p_f = feat.loc[is_m0p, :]; m0p_lab = lab.loc[is_m0p, :]
  m1_f = feat.loc[is_m1, :]; m1_lab = lab.loc[is_m1, :]

  ret = ((nepc_f, nepc_lab),
         (m0_f, m0_lab), 
         (m0p_f, m0p_lab), 
         (m1_f, m1_lab),)
  return ret