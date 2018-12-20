import pandas as pd
import numpy as np
import hashlib
import shutil
import glob
import os

from argparse import ArgumentParser
from scipy.stats import spearmanr
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon

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
    m0_case_vect = lab['case_id'][is_m0]
    nepc_case_vect = lab['case_id'][is_nepc]
    m1_case_vect = lab['case_id'][is_m1]

    m0_cases = lab['case_id'][is_m0]
    nepc_cases = lab['case_id'][is_nepc]
    train_case_vect = np.concatenate([m0_cases, nepc_cases])
    return train_x, train_y, m1_x, train_case_vect, m1_case_vect

"""
feat, lab has M0 (lab=0), and NEPC (lab=1)
Split off a subset of M0 cases for testing
"""
def holdout_m0(feat, lab, caseids, n=10):
    is_m0 = lab == 0
    is_nepc = lab == 1

def main(args):
    feat, case_ids = load_features(args.src, zscore=True)
    lab  = load_labels(args.labsrc)

    feat = drop_high_cor(feat, cor_thresh = 0.8)
    print('Features after high cor drop')
    print(feat.head())

    # train_x, train_y, test_x, test_y = holdout_cases(feat, lab)
    train_x, train_y, m1_x, train_case_vect, m1_case_vect = split_sets(feat, lab)
    model = ElasticNet(alpha=1e-3, max_iter=10000).fit(train_x, train_y)

    yhat_m1 = model.predict(m1_x)
    print(yhat_m1)
    case_mean = []
    m1_case_numbers = []
    print('M1 Cases:')
    for uc in np.unique(m1_case_vect):
        yx = yhat_m1[m1_case_vect == uc]
        case_mean.append(np.mean(yx))
        case_num = int(uc.split('-')[1])
        print(uc, case_num)
        m1_case_numbers.append(case_num)
    case_mean = np.array(case_mean)
    m1_case_numbers = np.array(m1_case_numbers)

    yhat_train = model.predict(train_x)
    train_mean, train_case_y = [], []
    for uc in np.unique(train_case_vect):
        idx = train_case_vect == uc
        train_mean.append(np.mean(yhat_train[idx]))
        train_case_y.append(train_y[idx][0])
    train_mean = np.array(train_mean)
    train_case_y = np.array(train_case_y)

    dotest = mannwhitneyu
    test_args = {'equal_var': False}
    test_args = {}
    test_m0_m1 =   dotest(yhat_train[train_y==0], yhat_m1, **test_args)
    test_m0_nepc = dotest(yhat_train[train_y==0], yhat_train[train_y==1], **test_args)
    test_nepc_m1 = dotest(yhat_train[train_y==1], yhat_m1, **test_args)
    print('Tiles M0 vs M1', test_m0_m1)
    print('Tiles M1 vs NPEC', test_m0_nepc)
    print('Tiles NEPC vs M1', test_nepc_m1)

    test_m0_m1 =   dotest(train_mean[train_case_y==0], case_mean, **test_args)
    test_m0_nepc = dotest(train_mean[train_case_y==0], 
                          train_mean[train_case_y==1], **test_args)
    test_nepc_m1 = dotest(train_mean[train_case_y==1], case_mean, **test_args)
    print('M0 vs M1', test_m0_m1)
    print('M1 vs NPEC', test_m0_nepc)
    print('NEPC vs M1', test_nepc_m1)

    print('------------------------------------------------------------------------------------')
    gene_scores = pd.read_csv('../data/signature_scores_matched.csv', index_col=None, header=0, sep=',')
    print(gene_scores.head())
    gene_score_caseid = []
    drop_rows = []
    matching_scores = []
    matching_indices = []
    for i, (idx, sn) in enumerate(zip(gene_scores.index.values, gene_scores['Surgical Number'].values)):
        try:
            x = int(sn.split(' ')[-1])
            if x in m1_case_numbers:
                gene_score_caseid.append(x)
                matching_indices.append(idx)
                matching_scores.append(case_mean[m1_case_numbers==x][0])
            else:
                drop_rows.append(idx)
        except:
            drop_rows.append(idx)
            print(sn)

    print(gene_scores.shape)
    gene_scores.drop(drop_rows, inplace=True)
    print(gene_scores.shape)
    gene_scores['NEPC Score'] = pd.Series(matching_scores, index=matching_indices)
    print(gene_scores.head())

    if args.save_scores:
        gene_scores.to_csv('../data/signature_scores_nepc_scores.csv')

    label_cols = ['caseid', 'Disease Stage', 'sample name', 'Surgical Number']
    gene_scores.drop(label_cols, inplace=True, axis=1)
    print(gene_scores.head())

    plt.figure(figsize=(5,5), dpi=300)
    sns.pairplot(gene_scores, kind='reg')
    plt.savefig('gene_scores_nepc_score.png', bbox_inches='tight')

    test_cols = [x for x in gene_scores.columns if x != 'NEPC Score']
    scores = gene_scores['NEPC Score'].values
    for c in test_cols:
        ctest = spearmanr(scores, gene_scores[c].values)
        print('{}: {}'.format(c, ctest))

    print('------------------------------------------------------------------------------------')

    if args.boxplot:
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.35, .65)})
        plt_m0 = train_mean[train_case_y==0]
        plt_nepc = train_mean[train_case_y==1]
        plt_m1 = case_mean
        sns.distplot(plt_m0, 
                    bins=25, 
                    norm_hist=True,
                    kde=True,
                    label='M0 training',
                    ax=ax_hist,
        )
        sns.distplot(plt_nepc, 
                    bins=25, 
                    norm_hist=True,
                    kde=True,
                    label='NEPC training',
                    ax=ax_hist,
        )
        sns.distplot(plt_m1, 
                    kde=True,
                    norm_hist=True,
                    bins=25, 
                    label='M1 testing',
                    ax=ax_hist,
        )
        ax_hist.set_xlabel('Score')
        ax_hist.set_ylabel('Frequency')
        concat_scores = np.concatenate([plt_m0, plt_nepc, plt_m1])
        concat_labels = np.array(['M0'] * len(plt_m0) + ['NEPC'] * len(plt_nepc) + ['M1'] * len(plt_m1))
        plt_df = pd.DataFrame({'Set': concat_labels, 'Score': concat_scores})

        # fig = plt.figure(figsize=(2,2), dpi=300)
        sns.boxplot(y='Set', x='Score', data=plt_df, ax=ax_box)
        sns.stripplot(y='Set', x='Score', data=plt_df, size=2.5, jitter=True, linewidth=0.5, ax=ax_box)
        # ax_box.set_ylabel('')
        # ax_box.set_xlabel('')
        # plt.show()
        plt.savefig('NEPC_score.png', bbox_inches='tight')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--src',    default='../data/handcrafted_tile_features.csv')
    parser.add_argument('--labsrc', default='../data/case_stage_files.tsv')
    parser.add_argument('--boxplot', default=False, action='store_true')
    parser.add_argument('--save_scores', default=True, action='store_true')
    parser.add_argument('--score_correlation', default=False, action='store_true')

    args = parser.parse_args()
    main(args)