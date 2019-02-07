import numpy as np
# import modin.pandas as pd
import pandas as pd

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

data = pd.read_csv('nuclear_features.csv', index_col=0, header=0)
data = data.reset_index(drop=True)
print('Data')
print(data.head())
print(data.shape)

caseids = np.copy(data['case_id'].values)
data.drop(['case_id', 'tile_id'], axis=1, inplace=True)
data = drop_nan_inf(data)
print('Data')
print(data.head())
print(data.shape)

data = data.groupby(caseids).mean()
print('Data')
print(data.head())
print(data.shape)

caseids = np.unique(caseids)
data.index = caseids
print('Data')
print(data.head())
print(data.shape)

data.to_csv('handcrafted_nuclear_features_case.csv')