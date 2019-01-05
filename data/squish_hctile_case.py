import modin.pandas as pd
import numpy as np

data = pd.read_csv('handcrafted_tile_features.csv', index_col=0, header=0)
data = data.reset_index(drop=True)
print('Data')
print(data.head())
print(data.shape)

caseids = np.copy(data['case_id'].values)
data.drop('case_id', axis=1, inplace=True)
data = data.groupby(caseids).mean()
print('Data')
print(data.head())
print(data.shape)

caseids = np.unique(caseids)
data.index = caseids
print('Data')
print(data.head())
print(data.shape)

data.to_csv('handcrafted_tile_features_case.csv')