#import modin.pandas as pd
import pandas as pd
import numpy as np
import hashlib

data_file = '../data/nuclear_features.csv'
labels_file = '../data/case_stage_files.tsv'

data = pd.read_csv(data_file, index_col=0, header=0)
labels = pd.read_csv(labels_file, index_col=0, header=0, sep='\t')
print('Data: ', data.shape)
print('labels: ', labels.shape)

groupstr = labels['stage_str']
data_tiles = data['tile_id']
data.drop(['case_id', 'tile_id'], inplace=True, axis=1)
data = data.groupby(data_tiles, as_index=True, sort=False).mean()
print(data.head())
print(data.shape)

#labels_tiles = labels.index
#groupstr = groupstr.groupby(labels_cases, sort=False).first()
#groupstr = pd.DataFrame({'': stage_str}, index=labels.index)
print('Dropping rows from groupstr: ', groupstr.shape)
groupstr.drop([x for x in groupstr.index if x not in data.index], inplace=True, axis=0)
print('groupstr: ', groupstr.shape)

# data['stage_str'] = groupstr
data = pd.concat([data, groupstr], axis=1)
data.index = groupstr.index
print(data.head())
print(data.shape)

data.to_csv('nuclear_features_group_tile.csv')
