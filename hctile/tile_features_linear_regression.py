import pandas as pd
import numpy as np
import hashlib
import shutil
import glob
import os

from argparse import ArgumentParser
from scipy.stats import spearmanr, pearsonr
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNetCV, ElasticNet
from utils import (drop_high_cor, load_features, load_labels, split_sets)

from sklearn.metrics import roc_auc_score

nepc_strs = ['NEPC']
adeno_strs = ['M0 NP', 'M0 oligo poly', 'M0 oligo', 'M0 poly', 'M1 oligo poly',
              'M1 oligo', 'M1 poly', 'MX Diffuse', 'NXMX P']
m0_strs = ['M0 NP']
m0p_strs = ['M0 oligo poly', 'M0 oligo', 'M0 poly']
m1_strs = ['M1 oligo poly', 'M1 oligo', 'M1 poly']

scnepc = []
with open('../data/nepc_small_cell_list.txt', 'r') as f:
  for L in f:
    scnepc.append(L.replace('\n', ''))

def make_training(label_0_feat, label_1_feat):
  train_x = pd.concat([label_0_feat, label_1_feat])
  train_y = np.array([0]*label_0_feat.shape[0]+ [1]*label_1_feat.shape[0])
  return train_x, train_y

def split_case(feats, lab, cid):
  cidx = lab['case_id'].values == cid
  cidx_ = lab['case_id'].values != cid
  feats_case = feats.loc[cidx, :]
  feats_other = feats.loc[cidx_, :]

  return feats_case, feats_other

"""
feat, lab has M0 (lab=0), and NEPC (lab=1)
Split off a subset of M0 cases for testing
"""
def holdout_m0(feat, lab, caseids, n=10):
  is_m0 = lab == 0
  is_nepc = lab == 1

def filter_stats(feat_1, feat_2, thresh=1e-10):
  remove_cols = []
  for i in feat_1.columns:
    res = ttest_ind(feat_1.loc[:, i], feat_2.loc[:, i])
    if res.pvalue > thresh:
      remove_cols.append(i)
  print('Removing {} columns'.format(len(remove_cols)))
  return remove_cols

def main(args):
  feat, case_ids = load_features(args.src, zscore=True)
  lab  = load_labels(args.labsrc)

  feat = drop_high_cor(feat, cor_thresh = 0.8)
  print('Features after high cor drop')

  # train_x, train_y, test_x, test_y = holdout_cases(feat, lab)
  ((nepc_f, nepc_lab), (m0_f, m0_lab), (m0p_f, m0p_lab), (m1_f, m1_lab)) = split_sets(feat, lab)
  del feat

  # Split out non-small-cell-NEPC:
  nepc_is_sc = np.array([x in scnepc for x in nepc_lab['case_id'].values])
  nepc_not_sc = np.array([x not in scnepc for x in nepc_lab['case_id'].values])
  nepc_f_sc = nepc_f.loc[nepc_is_sc, :]
  nepc_lab_sc = nepc_lab.loc[nepc_is_sc, :]

  nepc_f_not_sc = nepc_f.loc[nepc_not_sc, :]
  nepc_lab_not_sc = nepc_lab.loc[nepc_not_sc, :]
  del nepc_f, nepc_lab

  print('NEPC SC lab')
  print(nepc_lab_sc.head())
  print(nepc_lab_sc.shape)
  print('NEPC not SC lab')
  print(nepc_lab_not_sc.head())
  print(nepc_lab_not_sc.shape)

  if args.filter_stats:
    remove_cols = filter_stats(nepc_f_sc, m0_f)
    nepc_f_sc.drop(remove_cols, inplace=True, axis=1)
    nepc_f_not_sc.drop(remove_cols, inplace=True, axis=1)
    m0_f.drop(remove_cols, inplace=True, axis=1)
    m0p_f.drop(remove_cols, inplace=True, axis=1)
    m1_f.drop(remove_cols, inplace=True, axis=1)

  train_x, train_y = make_training(m0_f, nepc_f_sc)
  train_lab = pd.concat([m0_lab, nepc_lab_sc], axis=0)
  print('train lab')
  print(train_lab.head())
  print(train_lab.shape)

  # model = ElasticNet(alpha=1e-3, max_iter=50000).fit(train_x, train_y)
  # model = ElasticNetCV(cv=25).fit(train_x, train_y)
  # model = ElasticNetCV(alphas=np.arange(1e-5, 1e-1, 20), 
  #   cv=10, max_iter=10000, n_jobs=-1).fit(train_x, train_y)

  model = RandomForestRegressor(oob_score=True, max_depth=25, 
    n_estimators=100, n_jobs=-1).fit(train_x, train_y)

  with open('feature_importance.txt', 'w+') as f:
    for v, coef in zip(train_x.columns, model.feature_importances_):
      f.write('{}\t{}\n'.format(v, coef))

  if args.aggr_fn == 'max':
    aggr_fn = np.max
  elif args.aggr_fn == 'mean':
    aggr_fn = np.mean

  # """ Get M0 case numbers """
  # m0_case_numbers = []
  # m0_case_vect = m1_lab['case_id'].values
  # print('M0 Cases:')
  # for uc in np.unique(m0_case_vect):
  #   case_num = int(uc.plist('-')[1])
  #   m0_case_numbers.append(case_num)

  """ Predict the M1 cases and gather by mean """
  yhat_m1 = model.predict(m1_f)
  case_aggr = []
  m1_case_numbers = []
  m1_case_vect = m1_lab['case_id'].values
  for uc in np.unique(m1_case_vect):
    yx = yhat_m1[m1_case_vect == uc]
    case_aggr.append(aggr_fn(yx))
    case_num = int(uc.split('-')[1])
    m1_case_numbers.append(case_num)
  m1_case_aggr = np.array(case_aggr)
  m1_case_numbers = np.array(m1_case_numbers)    

  # Print out 
  m1_lab['NEPC_score'] = yhat_m1
  print('m1 lab')
  print(m1_lab.head())
  
  """ Predict M0P cases """
  yhat_m0p = model.predict(m0p_f)
  case_aggr = []
  m0p_case_numbers = []
  m0p_case_vect = m0p_lab['case_id'].values
  for uc in np.unique(m0p_case_vect):
    yx = yhat_m0p[m0p_case_vect == uc]
    case_aggr.append(aggr_fn(yx))
    case_num = int(uc.split('-')[1])
    m0p_case_numbers.append(case_num)
  m0p_case_aggr = np.array(case_aggr)
  m0p_case_numbers = np.array(m0p_case_numbers)

  # Print out 
  m0p_lab['NEPC_score'] = yhat_m0p
  print('m0p lab')
  print(m0p_lab.head())

  """ Predict NEPC not SC cases """
  yhat_nepc_not_sc = model.predict(nepc_f_not_sc)
  case_aggr = []
  nepc_not_sc_case_numbers = []
  nepc_not_sc_case_vect = nepc_lab_not_sc['case_id'].values
  for uc in np.unique(nepc_not_sc_case_vect):
    yx = yhat_nepc_not_sc[nepc_not_sc_case_vect == uc]
    case_aggr.append(aggr_fn(yx))
  nepc_not_sc_case_aggr = np.array(case_aggr)
  nepc_not_sc_case_numbers = np.array(nepc_not_sc_case_numbers)

  # Print out 
  nepc_lab_not_sc['NEPC_score'] = yhat_nepc_not_sc
  print('NEPC not sc lab')
  print(nepc_lab_not_sc.head())

  """ Check on training data
  Run a LOOCV on the training data """

  # yhat_train = []
  # # Just do m0 and nepc separately
  # for cid in np.unique(m0_lab['case_id'].values):
  #   feat_case, feat_other = split_case(m0_f, m0_lab, cid)
  #   feat_split = pd.concat([feat_other, nepc_f])
  #   y_split = [0]*feat_other.shape[0] + [1]*nepc_f.shape[0]
  #   model = RandomForestRegressor(n_estimators=100, n_jobs=-1).fit(feat_split, y_split)
  #   yh = model.predict(feat_case)
  #   print(cid, yh)
  #   yhat_train += list(yh)
  # for cid in np.unique(nepc_lab['case_id'].values):
  #   feat_case, feat_other = split_case(nepc_f, nepc_lab, cid)
  #   feat_split = pd.concat([m0_f, feat_other])
  #   y_split = [0]*m0_f.shape[0] + [1]*feat_other.shape[0]
  #   model = RandomForestRegressor(n_estimators=100, n_jobs=-1).fit(feat_split, y_split)
  #   yh = model.predict(feat_case)
  #   print(cid, yh)
  #   yhat_train += list(yh)
  # yhat_train = np.asarray(yhat_train)
  # print(yhat_train.shape)

  m0_cases = m0_lab['case_id'].values
  nepc_cases = nepc_lab_sc['case_id'].values
  train_case_vect = np.concatenate([m0_cases, nepc_cases])
  # yhat_train = model.predict(train_x)
  yhat_train = model.oob_prediction_
  train_aggr, train_case_y = [], []
  for uc in np.unique(train_case_vect):
    idx = train_case_vect == uc
    train_aggr.append(aggr_fn(yhat_train[idx]))
    train_case_y.append(train_y[idx][0])
  train_aggr = np.array(train_aggr)
  train_case_y = np.array(train_case_y)

  # Print out 
  train_lab['NEPC_score'] = yhat_train
  print('train lab')
  print(train_lab.head())

  score_lab = pd.concat([m1_lab, m0p_lab, train_lab], axis=0)
  print(score_lab.shape)
  score_lab.to_csv('tile_paths_with_NEPC_score.csv')

  """ write out scores """
  with open('nepc_case_scores.txt', 'w+') as f:
    for mop, mop_score in zip(np.unique(m0p_case_vect), m0p_case_aggr):
      s = '{}\t{}\n'.format(mop, mop_score)
      f.write(s)

    for mop, mop_score in zip(np.unique(m1_case_vect), m1_case_aggr):
      s = '{}\t{}\n'.format(mop, mop_score)
      f.write(s)

    for mop, mop_score in zip(np.unique(train_case_vect), train_aggr):
      s = '{}\t{}\n'.format(mop, mop_score)
      f.write(s)

  """ Do some statistical tests """
  dotest = mannwhitneyu
  # test_args = {'equal_var': True}
  test_args = {}
  test_m0_m1   = dotest(yhat_train[train_y==0], yhat_m1, **test_args)
  test_m0_m0p  = dotest(yhat_train[train_y==0], yhat_m0p, **test_args)
  test_m0_nepc = dotest(yhat_train[train_y==0], yhat_train[train_y==1], **test_args)
  test_nepc_m1 = dotest(yhat_train[train_y==1], yhat_m1, **test_args)
  test_m0_nepc_not_sc = dotest(yhat_train[train_y==0], yhat_nepc_not_sc, **test_args)
  test_nepc_sc_nepc_not_sc = dotest(yhat_train[train_y==1], yhat_nepc_not_sc, **test_args)
  print('Tiles M0 vs M1',   test_m0_m1)
  print('Tiles M0 vs M0P',  test_m0_m0p)
  print('Tiles M0 vs NPEC SC', test_m0_nepc)
  print('Tiles M0 vs NPEC NOT SC', test_m0_nepc_not_sc)
  print('Tiles NEPC vs M1', test_nepc_m1)
  print('Tiles NEPC SC vs NEPC NOT SC', test_nepc_sc_nepc_not_sc)

  test_m0_m1   = dotest(train_aggr[train_case_y==0], m1_case_aggr, **test_args)
  test_m0_m0p  = dotest(train_aggr[train_case_y==0], m0p_case_aggr, **test_args)
  test_m0_nepc = dotest(train_aggr[train_case_y==0], 
                        train_aggr[train_case_y==1], **test_args)
  test_nepc_m1 = dotest(train_aggr[train_case_y==1], m1_case_aggr, **test_args)
  test_m0_nepc_not_sc = dotest(train_aggr[train_case_y==0], nepc_not_sc_case_aggr, **test_args)
  test_nepc_sc_nepc_not_sc = dotest(train_aggr[train_case_y==1], nepc_not_sc_case_aggr, **test_args)
  print('aggr M0 vs M1', test_m0_m1)
  print('aggr M0 vs M0P', test_m0_m0p)
  print('aggr M0 vs NPEC SC', test_m0_nepc)
  print('aggr M0 vs NPEC NOT SC', test_m0_nepc_not_sc)
  print('aggr NEPC vs M1', test_nepc_m1)
  print('aggr NPEC SC vs NEPC NOT SC', test_nepc_sc_nepc_not_sc)

  print('------------------------------------------------------------------------------------')
  if args.genescore:
    gene_scores = pd.read_csv('../data/signature_scores_beltram.csv', index_col=None, header=0, sep=',')
    gene_score_caseid = []
    drop_rows = []
    matching_scores = []
    matching_indices = []
    for i, (idx, sn) in enumerate(zip(gene_scores.index.values, gene_scores['Surgical Number'].values)):
      try:
        x = int(sn.split(' ')[-1])
        if x in m1_case_numbers:
          # print('M1 matched SN {}'.format(x))
          gene_score_caseid.append(x)
          matching_indices.append(idx)
          matching_scores.append(m1_case_aggr[m1_case_numbers==x][0])
        # if x in m0_case_numbers:
        #   print('M0 matched SN {}'.format(x))
        #   gene_score_caseid.append(x)
        #   matching_indices.append(idx)
        #   matching_scores.append(m1_case_mean[m1_case_numbers==x][0])
        elif x in m0p_case_numbers:
          # print('M0P matched SN {}'.format(x))
          gene_score_caseid.append(x)
          matching_indices.append(idx)
          matching_scores.append(m0p_case_aggr[m0p_case_numbers==x][0])
        else:
          drop_rows.append(idx)
      except:
        drop_rows.append(idx)
        print(sn)

    gene_scores.drop(drop_rows, inplace=True)
    print(gene_scores.shape)
    gene_scores['NEPC Score'] = pd.Series(matching_scores, index=matching_indices)

    # if args.save_scores:
      # gene_scores.to_csv('../data/signature_scores_nepc_scores_mean.csv')

    label_cols = ['caseid', 'Disease Stage', 'sample name', 'Surgical Number']
    gene_scores.drop(label_cols, inplace=True, axis=1)

    # plt.figure(figsize=(5,5), dpi=300)
    # sns.pairplot(gene_scores, kind='reg')
    # plt.savefig('gene_scores_nepc_score_{}.png'.format(args.aggr_fn), bbox_inches='tight')

    test_cols = [x for x in gene_scores.columns if x != 'NEPC Score']
    scores = gene_scores['NEPC Score'].values
    for c in test_cols:
      try:
        ctest = spearmanr(scores, gene_scores[c].values)
        print('spearman {:40}: {:3.5f} p={:3.5f}'.format(c, ctest.correlation, ctest.pvalue))
        ctest = pearsonr(scores, gene_scores[c].values)
        print('pearson  {:40}: {:3.5f} p={:3.5f}'.format(c, ctest[0], ctest[1]))
      except:
        print('Test column {} failed'.format(c))


  print('------------------------------------------------------------------------------------')
  if args.boxplot:
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.35, .65)})
    plt_m0 = train_aggr[train_case_y==0]
    plt_nepc_sc = train_aggr[train_case_y==1]
    plt_nepc_not_sc = nepc_not_sc_case_aggr
    plt_m1 = m1_case_aggr
    plt_m0p = m0p_case_aggr

    auc_ = roc_auc_score(y_true=train_case_y, y_score=train_aggr)
    print('M0 NEPC SC AUC = ', auc_)

    m0m1 = np.concatenate([plt_m0, plt_m1])
    m0m1_y = np.array([0]*len(plt_m0) + [1]*len(plt_m1))
    auc_ = roc_auc_score(y_true=m0m1_y, y_score=m0m1)
    print('M0 M1 AUC = ', auc_)

    m0m0p = np.concatenate([plt_m0, plt_m0p])
    m0m0p_y = np.array([0]*len(plt_m0) + [1]*len(plt_m0p))
    auc_ = roc_auc_score(y_true=m0m0p_y, y_score=m0m0p)
    print('M0 M0P AUC = ', auc_)

    m0nepc_not_sc = np.concatenate([plt_m0, plt_nepc_not_sc])
    m0nepc_not_sc_y = np.array([0]*len(plt_m0) + [1]*len(plt_nepc_not_sc))
    auc_ = roc_auc_score(y_true=m0nepc_not_sc_y, y_score=m0nepc_not_sc)
    print('M0 NEPC not SC AUC = ', auc_)

    sns.distplot(plt_m0, 
                bins=25, 
                norm_hist=True,
                kde=True,
                label='M0',
                ax=ax_hist,)
    sns.distplot(plt_nepc_sc, 
                bins=25, 
                norm_hist=True,
                kde=True,
                label='NEPC SC',
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
    sns.distplot(plt_nepc_not_sc, 
                kde=True,
                norm_hist=True,
                bins=25, 
                label='NEPC not SC',
                ax=ax_hist,)
    ax_hist.set_xlabel('Score')
    ax_hist.set_ylabel('Frequency')
    concat_scores = np.concatenate([plt_m0, plt_nepc_sc, plt_m1, plt_m0p, plt_nepc_not_sc])
    concat_labels = np.array(['M0'] * len(plt_m0) + ['NEPC SC'] * len(plt_nepc_sc) + ['M1'] * len(plt_m1) + ['M0P'] * len(plt_m0p) + ['NEPC not SC'] * len(plt_nepc_not_sc))

    plt_df = pd.DataFrame({'Set': concat_labels, 'Score': concat_scores})

    # fig = plt.figure(figsize=(2,2), dpi=300)
    sns.boxplot(y='Set', x='Score', data=plt_df, ax=ax_box)
    sns.stripplot(y='Set', x='Score', data=plt_df, size=2.5, jitter=True, linewidth=0.5, ax=ax_box)
    # ax_box.set_ylabel('')
    # ax_box.set_xlabel('')
    # plt.show()
    plt.savefig('NEPC_score_{}.png'.format(args.aggr_fn), bbox_inches='tight')

if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--src',     default='../data/handcrafted_tile_features.csv')
  parser.add_argument('--labsrc',  default='../data/case_stage_files.tsv')
  parser.add_argument('--boxplot', default=False, action='store_true')
  parser.add_argument('--aggr_fn',   default='mean', type=str)
  parser.add_argument('--genescore', default=False, action='store_true')
  parser.add_argument('--save_scores', default=False, action='store_true')
  parser.add_argument('--filter_stats', default=False, action='store_true')

  args = parser.parse_args()
  main(args)
