import pandas as pd
# import modin.pandas as pd
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
from utils import drop_high_cor

m0_strs = ['M0 NP']
m1_strs = ['M1 oligo poly', 'M1 oligo', 'M1 poly']
def split_sets(feat, case_ids, stages):
    is_nepc = np.array(['NEPC' in x for x in stages])
    is_m0 = np.array([x in m0_strs for x in  stages])
    is_m1 = np.array([x in m1_strs for x in  stages])

    nepc_x = feat.loc[is_nepc, :]
    m0_x = feat.loc[is_m0, :]
    m1_x = feat.loc[is_m1, :]
    print('nepc_x', nepc_x.shape)
    print('m0_x', m0_x.shape)
    print('m1_x', m1_x.shape)

    train_x = pd.concat([m0_x, nepc_x], axis=0)
    train_y = np.array([0]*m0_x.shape[0]+ [1]*nepc_x.shape[0])
    m0_case_vect   = case_ids[is_m0]
    nepc_case_vect = case_ids[is_nepc]
    m1_case_vect   = case_ids[is_m1]

    m0_cases =   case_ids[is_m0]
    nepc_cases = case_ids[is_nepc]
    train_case_vect = np.concatenate([m0_cases, nepc_cases])
    return train_x, train_y, m1_x, train_case_vect, m1_case_vect

def drop_nan_inf(data):
    isinfs = np.sum(np.isinf(data.values), axis=0); print('isinfs', isinfs.shape)
    isnans = np.sum(np.isnan(data.values), axis=0); print('isnans', isnans.shape)
    print(np.argwhere(isinfs))
    print(np.argwhere(isnans))
    # data = data.dropna(axis='index')
    inf_cols = data.columns.values[np.squeeze(np.argwhere(isinfs))]
    nan_cols = data.columns.values[np.squeeze(np.argwhere(isnans))]
    print('inf_cols', inf_cols)
    print('nan_cols', nan_cols)
    data.drop(inf_cols, axis=1, inplace=True)
    data.drop(nan_cols, axis=1, inplace=True)
    print(data.shape)
    return data

def get_case_id_num(labsrc):
    lab = pd.read_csv(labsrc, index_col=0, header=0, sep='\t')
    case_id2num = {}
    case_ids = lab['case_id'].values
    for c in np.unique(case_ids):
        print(c)
        try:
            case_num = int(c.split('-')[1])
            case_uid = hashlib.md5(c.encode()).hexdigest()
            case_id2num[case_uid] = case_num
        except:
            print('No case number')
    
    return case_id2num

def main(args):
    case_id2num = get_case_id_num(args.labsrc)

    feat = pd.read_csv(args.src, index_col=0, header=0)

    tile_ids = feat['tile_id']
    stages   = feat['stage_str']
    feat.drop(['case_id', 'tile_id', 'stage_str'], 
        axis=1, inplace=True)

    if args.ae_only:
        to_drop = [x for x in feat.columns if 'ae' not in x]
        feat.drop(to_drop, axis=1, inplace=True)

    if args.hc_only:
        to_drop = [x for x in feat.columns if 'hc' not in x]
        feat.drop(to_drop, axis=1, inplace=True)

    feat = drop_high_cor(feat, 0.8)
    print('Features after high cor drop')
    print(feat.shape)
    print(feat.head())

    feat = feat.transform(lambda x: (x - np.mean(x)) / np.std(x))
    print('Features after zscore')
    print(feat.shape)
    print(feat.head())

    feat = drop_nan_inf(feat)
    print('Features after dropping nan and infs')
    print(feat.shape)
    print(feat.head())

    # train_x, train_y, test_x, test_y = holdout_cases(feat, lab)
    train_x, train_y, m1_x, train_case_vect, m1_case_vect = \
        split_sets(feat, case_ids, stages)
    print('train_x', train_x.shape)
    # print(train_x)
    print('train_y', train_y.shape)
    print('m1_x', m1_x.shape)
    print('train_case_vect', train_case_vect.shape)
    print('m1_case_vect', m1_case_vect.shape)
    model = ElasticNet(alpha=1e-3, max_iter=25000).fit(train_x, train_y)

    """ Predict the M1 cases and gather by max and mean """
    yhat_m1 = model.predict(m1_x)
    case_mean = []
    m1_case_numbers = []
    print('M1 Cases:')
    for uc in np.unique(m1_case_vect):
        yx = yhat_m1[m1_case_vect == uc]
        case_mean.append(np.mean(yx))
        # case_num = int(uc.split('-')[1])
        case_num = case_id2num[uc]
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

    dotest = ttest_ind
    test_args = {'equal_var': False}
    # test_args = {}
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
    print('Mean M0 vs M1', test_m0_m1)
    print('Mean M1 vs NPEC', test_m0_nepc)
    print('Mean NEPC vs M1', test_nepc_m1)

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
        gene_scores.to_csv('../signature_scores_nepc_scores_nuclei_mean.csv')

    label_cols = ['caseid', 'Disease Stage', 'sample name', 'Surgical Number']
    gene_scores.drop(label_cols, inplace=True, axis=1)
    print(gene_scores.head())

    plt.figure(figsize=(5,5), dpi=300)
    sns.pairplot(gene_scores, kind='reg')
    plt.savefig('gene_scores_nepc_score_mean_tile.png', bbox_inches='tight')

    test_cols = [x for x in gene_scores.columns if x != 'NEPC Score']
    scores = gene_scores['NEPC Score'].values
    print('------------------------------------------------------------------------------------')
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
 
        plt.savefig('NEPC_score_nuclei_group_tile_mean.png', bbox_inches='tight')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--src',    default='../data/nuclear_features_group_tile.csv')
    parser.add_argument('--labsrc',    default='../data/case_stage_files.tsv')
    parser.add_argument('--boxplot', default=False, action='store_true')
    parser.add_argument('--save_scores', default=False, action='store_true')
    parser.add_argument('--ae_only', default=False, action='store_true')
    parser.add_argument('--hc_only', default=False, action='store_true')

    args = parser.parse_args()
    main(args)
