"""
Merge autoencoder - and handcrafted nuclear features by unique nucleus ID
"""
import pandas as pd
# import modin.pandas as pd
import numpy as np
import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

def main(args):
    ae = pd.read_csv(args.ae_src, sep=',', index_col=0)
    print(ae.shape)
    # print(ae.head())

    hc = pd.read_csv(args.hc_src, sep=',', index_col=0)
    print(hc.shape)
    # print(hc.head())

    matching_ids = np.intersect1d(ae.index.values, hc.index.values)
    print('Matched: ', len(matching_ids))

    ae = ae.loc[matching_ids, :]
    hc = hc.loc[matching_ids, :]
    print('Matching IDs only:')
    print(ae.shape)
    print(hc.shape)
    # print(hc.head())

    cids = hc['case_id'].values
    tids = hc['tile_id'].values

    # Do some clearning 
    print('Dropping id cols:')
    ae.drop(['case_id', 'tile_id'], axis=1, inplace=True)
    hc.drop(['case_id', 'tile_id'], axis=1, inplace=True)
    print(ae.shape)
    print(hc.shape)
    # print(hc.head())

    merged = ae.join(hc, how='inner', lsuffix='ae', rsuffix='hc')
    del ae
    del hc
    print(merged.shape)
    merged['case_id'] = cids
    merged['tile_id'] = tids
    print(merged.shape)
    # print(merged.head())

    merged.to_csv(args.dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ae_src', default='../data/autoencoder_features.csv', type=str)
    parser.add_argument('--hc_src', default='../data/handcrafted_nuclear_features.csv', type=str)
    parser.add_argument('--dst',    default='../data/nuclear_features.csv', type=str)

    args = parser.parse_args()
    main(args)
