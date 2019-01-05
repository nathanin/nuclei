""" 

index, nucleusIDs, tileIDs, caseIDs, stage_str
1    , ...       , ...    , ...    , ...
.
.
.
N

Grab the IDs from nuclear features
Grab the stage_str from case_stage_files
"""

import pandas as pd
import numpy as np

ids = pd.read_csv('handcrafted_nuclear_features.csv', usecols=[0, 64, 65], index_col=0, header=0)
ids['nucleus_id'] = ids.index
ids.index = range(ids.shape[0])
print(ids.head())
print(ids.shape)
print(len(ids['tile_id'].unique()))

labels = pd.read_csv('case_stage_files.tsv', sep='\t', index_col=0, header=0)
print(labels.head())
print(labels.shape)
print(len(labels.index.unique()))

stage_strs = labels['stage_str']

expanded_tileids = np.copy(ids['tile_id'].values)
stage_strs_expanded = np.zeros_like(expanded_tileids)
print(stage_strs_expanded.shape, stage_strs_expanded.dtype)
for k, i in enumerate(ids['tile_id'].unique()):
    tile_idx = expanded_tileids == i
    stage_str = stage_strs.loc[i]
    stage_strs_expanded[tile_idx] = stage_str 
    if k % 100 == 0:
        print(k, i, stage_str)

ids['stage_str'] = stage_strs_expanded
print(ids.head())

ids.to_csv('label_table.csv')