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

nepc_strs = ['NEPC']
adeno_strs = ['M0 NP', 'M0 oligo poly', 'M0 oligo', 'M0 poly', 'M1 oligo poly',
              'M1 oligo', 'M1 poly', 'MX Diffuse', 'NXMX P']
m0_strs = ['M0 NP']
m0p_strs = ['M0 oligo poly', 'M0 oligo', 'M0 poly']
m1_strs = ['M1 oligo poly', 'M1 oligo', 'M1 poly']

def split_sets(feat, lab):
    """
    Return a tuple:
    ((nepc_f, nepc_lab), (m0_f, m0_lab),... )
    """
    is_nepc = np.array([x in nepc_strs for x in lab['stage_str']])
    is_m0 = np.array([x in m0_strs for x in lab['stage_str']])
    is_m0p = np.array([x in m0p_strs for x in lab['stage_str']])
    is_m1 = np.array([x in m1_strs for x in lab['stage_str']])

    nepc_f = feat.loc[is_nepc, :]; nepc_lab = lab.loc[is_nepc, :]
    m0_f = feat.loc[is_m0, :]; m0_lab = lab.loc[is_m0, :]
    m0p_f = feat.loc[is_m0p, :]; m0p_lab = lab.loc[is_m0p, :]
    m1_f = feat.loc[is_m1, :]; m1_lab = lab.loc[is_m1, :]

    ret = ((nepc_f, nepc_lab),
           (m0_f, m0_lab), 
           (m0p_f, m0p_lab), 
           (m1_f, m1_lab),)
    return ret

def make_training(label_0, label_1):
    train_x = pd.concat([label_0, label_1])
    train_y = np.array([0]*label_0.shape[0]+ [1]*label_1.shape[0])
    return train_x, train_y

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
    ((nepc_f, nepc_lab), (m0_f, m0_lab), (m0p_f, m0p_lab), (m1_f, m1_lab)) = split_sets(feat, lab)
    train_x, train_y = make_training(m0_f, nepc_f)


    model = ElasticNet(alpha=1e-3, max_iter=10000).fit(train_x, train_y)

    """ Predict the M1 cases and gather by max and mean """
    yhat_m1 = model.predict(m1_f)
    print(yhat_m1)
    case_mean = []
    m1_case_numbers = []
    m1_case_vect = m1_lab['case_id'].values
    print('M1 Cases:')
    for uc in np.unique(m1_case_vect):
        yx = yhat_m1[m1_case_vect == uc]
        case_mean.append(np.mean(yx))
        case_num = int(uc.split('-')[1])
        print(uc, case_num)
        m1_case_numbers.append(case_num)
    m1_case_mean = np.array(case_mean)
    m1_case_numbers = np.array(m1_case_numbers)    
    
    """ Predict M0P cases """
    yhat_m0p = model.predict(m0p_f)
    print(yhat_m0p)
    case_mean = []
    m0p_case_numbers = []
    m0p_case_vect = m0p_lab['case_id'].values
    print('M1 Cases:')
    for uc in np.unique(m0p_case_vect):
        yx = yhat_m0p[m0p_case_vect == uc]
        case_mean.append(np.mean(yx))
        case_num = int(uc.split('-')[1])
        print(uc, case_num)
        m0p_case_numbers.append(case_num)
    m0p_case_mean = np.array(case_mean)
    m0p_case_numbers = np.array(m0p_case_numbers)

    """ Check on training data """
    m0_cases = m0_lab['case_id'].values
    nepc_cases = nepc_lab['case_id'].values
    train_case_vect = np.concatenate([m0_cases, nepc_cases])
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
    test_m0_m0p =   dotest(yhat_train[train_y==0], yhat_m0p, **test_args)
    test_m0_nepc = dotest(yhat_train[train_y==0], yhat_train[train_y==1], **test_args)
    test_nepc_m1 = dotest(yhat_train[train_y==1], yhat_m1, **test_args)
    print('Tiles M0 vs M1', test_m0_m1)
    print('Tiles M0 vs M0P', test_m0_m0p)
    print('Tiles M0 vs NPEC', test_m0_nepc)
    print('Tiles NEPC vs M1', test_nepc_m1)

    test_m0_m1 =   dotest(train_mean[train_case_y==0], m1_case_mean, **test_args)
    test_m0_m0p =   dotest(train_mean[train_case_y==0], m0p_case_mean, **test_args)
    test_m0_nepc = dotest(train_mean[train_case_y==0], 
                          train_mean[train_case_y==1], **test_args)
    test_nepc_m1 = dotest(train_mean[train_case_y==1], m1_case_mean, **test_args)
    print('Mean M0 vs M1', test_m0_m1)
    print('Mean M0 vs M0P', test_m0_m0p)
    print('Mean M0 vs NPEC', test_m0_nepc)
    print('Mean NEPC vs M1', test_nepc_m1)

    print('------------------------------------------------------------------------------------')
    gene_scores = pd.read_csv('../data/signature_scores_beltram.csv', index_col=None, header=0, sep=',')
    gene_scores.drop(gene_scores.columns[-1], axis=1, inplace=True)
    print(gene_scores.head())
    gene_score_caseid = []
    drop_rows = []
    matching_scores = []
    matching_indices = []
    for i, (idx, sn) in enumerate(zip(gene_scores.index.values, gene_scores['Surgical Number'].values)):
        try:
            x = int(sn.split(' ')[-1])
            if x in m1_case_numbers:
                print('M1 matched SN {}'.format(x))
                gene_score_caseid.append(x)
                matching_indices.append(idx)
                matching_scores.append(m1_case_mean[m1_case_numbers==x][0])
            elif x in m0p_case_numbers:
                print('M0P matched SN {}'.format(x))
                gene_score_caseid.append(x)
                matching_indices.append(idx)
                matching_scores.append(m0p_case_mean[m0p_case_numbers==x][0])
            else:
                drop_rows.append(idx)
        except:
            drop_rows.append(idx)
            print(sn)

    print(gene_scores.shape)
    gene_scores.drop(drop_rows, inplace=True)
    print(gene_scores.shape)
    gene_scores['NEPC HCTile'] = pd.Series(matching_scores, index=matching_indices)
    print(gene_scores.head())

    if args.save_scores:
        gene_scores.to_csv('../data/signature_scores_nepc_scores_mean.csv')

    label_cols = ['caseid', 'Disease Stage', 'sample name', 'Surgical Number']
    gene_scores.drop(label_cols, inplace=True, axis=1)
    print(gene_scores.head())

    plt.figure(figsize=(5,5), dpi=300)
    sns.pairplot(gene_scores, kind='reg')
    plt.savefig('gene_scores_nepc_score_mean.png', bbox_inches='tight')

    test_cols = [x for x in gene_scores.columns if x != 'NEPC HCTile']
    scores = gene_scores['NEPC HCTile'].values
    for c in test_cols:
        ctest = spearmanr(scores, gene_scores[c].values)
        print('{}: {}'.format(c, ctest))

    print('------------------------------------------------------------------------------------')
    if args.boxplot:
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.35, .65)})
        plt_m0 = train_mean[train_case_y==0]
        plt_nepc = train_mean[train_case_y==1]
        plt_m1 = m1_case_mean
        plt_m0p = m0p_case_mean
        sns.distplot(plt_m0, 
                    bins=25, 
                    norm_hist=True,
                    kde=True,
                    label='M0',
                    ax=ax_hist,)
        sns.distplot(plt_nepc, 
                    bins=25, 
                    norm_hist=True,
                    kde=True,
                    label='NEPC',
                    ax=ax_hist,)
        sns.distplot(plt_m1, 
                    kde=True,
                    norm_hist=True,
                    bins=25, 
                    label='M1',
                    ax=ax_hist,)
        sns.distplot(plt_m0p, 
                    kde=True,
                    norm_hist=True,
                    bins=25, 
                    label='M0-P',
                    ax=ax_hist,)
        ax_hist.set_xlabel('Score')
        ax_hist.set_ylabel('Frequency')
        concat_scores = np.concatenate([plt_m0, plt_nepc, plt_m1, plt_m0p])
        concat_labels = np.array(['M0'] * len(plt_m0) + ['NEPC'] * len(plt_nepc) + ['M1'] * len(plt_m1) + ['M0P'] * len(plt_m0p))
        plt_df = pd.DataFrame({'Set': concat_labels, 'Score': concat_scores})

        # fig = plt.figure(figsize=(2,2), dpi=300)
        sns.boxplot(y='Set', x='Score', data=plt_df, ax=ax_box)
        sns.stripplot(y='Set', x='Score', data=plt_df, size=2.5, jitter=True, linewidth=0.5, ax=ax_box)
        # ax_box.set_ylabel('')
        # ax_box.set_xlabel('')
        # plt.show()
        plt.savefig('NEPC_score_mean.png', bbox_inches='tight')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--src',     default='../data/handcrafted_tile_features.csv')
    parser.add_argument('--labsrc',  default='../data/case_stage_files.tsv')
    parser.add_argument('--boxplot', default=False, action='store_true')
    parser.add_argument('--save_scores', default=False, action='store_true')

    args = parser.parse_args()
    main(args)
