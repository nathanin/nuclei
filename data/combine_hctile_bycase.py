#import modin.pandas as pd
import pandas as pd
import numpy as np
import hashlib

data_file = '../data/handcrafted_tile_features.csv'
labels_file = '../data/case_stage_files.tsv'

data = pd.read_csv(data_file, index_col=0, header=0)
labels = pd.read_csv(labels_file, index_col=0, header=0, sep='\t')
print('Data: ', data.shape)
print('labels: ', labels.shape)

groupstr = labels['stage_str']
data_cases = data['case_id']
data.drop(['case_id'], inplace=True, axis=1)
data = data.groupby(data_cases, sort=False).mean()
data.reset_index(drop=True)
print(data.head())
print(data.shape)

def get_md5(x):
    return hashlib.md5(x.encode()).hexdigest()

labels_cases = labels['case_id']
labels_cases = [get_md5(x) for x in labels_cases]
groupstr = groupstr.groupby(labels_cases, sort=False).first()

# data['stage_str'] = groupstr
data = pd.concat([data, groupstr], axis=1)
print(data.head())
print(data.shape)

data.to_csv('tile_features_group_case.csv')
