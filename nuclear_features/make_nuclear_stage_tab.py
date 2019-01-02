import modin.pandas as pd
import numpy as np
import hashlib
import os

features_src = '../data/nuclear_features.csv'
labels_src = '../data/case_stage_files.csv'

features = pd.read_csv(features_src, memory_map=True)
print(features.head())

lab = pd.read_csv(labels_src)
print(lab.head())

stages = lab['stage_str'].values
cases = lab['case_id'].values
cases_uid = np.array(
    [hashlib.md5(x.encode()).hexdigest() for x in cases]
)
nuclei_cases = features['case_id'].values

nucleus_stage = np.zeros_like(nuclei_cases)
for cid in np.unique(cases_uid):
    cidx = nuclei_cases == cid
    st = np.squeeze(stages[cases_uid == cid])[0]
    print(cid, st)
    nucleus_stage[cidx] = st

print(nucleus_stage.shape)
features['stage_str'] = nucleus_stage

print(features.head())
features.to_csv(features_src)