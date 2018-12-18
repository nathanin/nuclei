import pandas as pd
import numpy as np
import hashlib
import shutil
import glob
import os

from argparse import ArgumentParser
from scipy.stats import spearmanr
from matplotlib import pyplot as plt

from sklearn.linear_model import ElasticNetCV, ElasticNet
from utils import (drop_high_cor, load_features, load_labels)

def holdout_cases(feat, lab, n=5):
    is_nepc = np.array(['NEPC' in x for x in lab['stage_str']])
    not_nepc = np.array(['NEPC' not in x for x in lab['stage_str']])
    nepc_case_feat = feat.loc[is_nepc,:]
    nepc_case_labs = lab.loc[is_nepc,:]

    adeno_case_feat = feat.loc[not_nepc,:]
    adeno_case_labs = lab.loc[not_nepc,:]

    nepc_case_ids = nepc_case_labs['case_id'].values
    unique_nepc = np.unique(nepc_case_ids)
    adeno_case_ids = adeno_case_labs['case_id'].values
    unique_adeno = np.unique(adeno_case_ids)

    choice_nepc = np.random.choice(unique_nepc, n, replace=False)
    print('Choice unique_nepc:', choice_nepc)
    choice_nepc_vec = np.array([x in choice_nepc for x in nepc_case_ids])
    not_choice_nepc_vec = np.array([x not in choice_nepc for x in nepc_case_ids])
    choice_adeno = np.random.choice(unique_adeno, n, replace=False)
    print('Choice unique_adeno:', choice_adeno)
    choice_adeno_vec = np.array([x in choice_adeno for x in adeno_case_ids])
    not_choice_adeno_vec = np.array([x not in choice_adeno for x in adeno_case_ids])

    train_x_nepc = nepc_case_feat.loc[choice_nepc_vec, :]
    train_x_adeno = adeno_case_feat.loc[choice_adeno_vec, :]
    test_x_nepc  = nepc_case_feat.loc[not_choice_nepc_vec, :]
    test_x_adeno = adeno_case_feat.loc[not_choice_adeno_vec, :]

    train_y = np.array([1]*train_x_nepc.shape[0] + [0]*train_x_adeno.shape[0])
    test_y = np.array([1]*test_x_nepc.shape[0] + [0]*test_x_adeno.shape[0])

    train_x = pd.concat([train_x_nepc, train_x_adeno])
    test_x = pd.concat([test_x_nepc, test_x_adeno])

    return train_x, train_y, test_x, test_y

def main(args):
    feat, case_ids = load_features(args.src)
    lab  = load_labels(args.labsrc)

    feat = drop_high_cor(feat, cor_thresh = 0.8)
    print('Features after high cor drop')
    print(feat.head())

    for _ in range(10):
        train_x, train_y, test_x, test_y = holdout_cases(feat, lab)
        model = ElasticNet(max_iter=5000).fit(train_x, train_y)

        yhat = model.predict(test_x)
        yhat_max = yhat > 0.5
        print((yhat_max == test_y).mean())

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--src', default='../data/handcrafted_tile_features.csv')
    parser.add_argument('--labsrc', default='../data/case_stage_files.tsv')

    args = parser.parse_args()
    main(args)