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

m0_strs = ['M0 NP']
m1_strs = ['M1 oligo poly', 'M1 oligo', 'M1 poly']
def split_sets(feat, lab):
    is_nepc = np.array(['NEPC' in x for x in lab['stage_str']])
    is_m0 = np.array([x in m0_strs for x in lab['stage_str']])
    is_m1 = np.array([x in m1_strs for x in lab['stage_str']])

    nepc_x = feat.loc[is_nepc, :]
    m0_x = feat.loc[is_m0, :]
    m1_x = feat.loc[is_m1, :]

    train_x = pd.concat([m0_x, nepc_x])
    train_y = np.array([0]*m0_x.shape[0]+ [1]*nepc_x.shape[0])
    m1_case_vect = lab['case_id'][is_m1]
    return train_x, train_y, m1_x, m1_case_vect

def main(args):
    feat, case_ids = load_features(args.src, zscore=True)
    lab  = load_labels(args.labsrc)

    feat = drop_high_cor(feat, cor_thresh = 0.8)
    print('Features after high cor drop')
    print(feat.head())

    # train_x, train_y, test_x, test_y = holdout_cases(feat, lab)
    train_x, train_y, m1_x, m1_case_vect = split_sets(feat, lab)
    model = ElasticNet(alpha=1e-3, max_iter=10000).fit(train_x, train_y)

    yhat = model.predict(m1_x)
    print(yhat)
    case_mean = []
    for uc in np.unique(m1_case_vect):
        yx = yhat[m1_case_vect == uc]
        case_mean.append(np.mean(yx))


    yhat_train = model.predict(train_x)

    plt.hist(yhat_train, density=True, bins=25, alpha=0.2)
    plt.hist(yhat, density=True, bins=25, alpha=0.2)
    # plt.hist(case_mean, density=True, alpha=0.2)
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--src', default='../data/handcrafted_tile_features.csv')
    parser.add_argument('--labsrc', default='../data/case_stage_files.tsv')

    args = parser.parse_args()
    main(args)