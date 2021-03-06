import pandas as pd
# import modin.pandas as pd
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

from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from utils import drop_high_cor


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
  return data

def filter_stats(feat_1, feat_2, thresh=0.5):
  remove_cols = []
  for i in feat_1.columns:
    res = ttest_ind(feat_1.loc[:, i], feat_2.loc[:, i])
    if res.pvalue > thresh:
      remove_cols.append(i)
  print('Removing {} columns'.format(len(remove_cols)))
  return remove_cols

def main(args):
  feat = pd.read_csv(args.src, index_col=0, header=0)
  labels = pd.read_csv(args.labsrc, index_col=0, header=0, sep='\t')
  # print(feat.head())
  # print(labels.head())

  case_ids = labels['case_id'].values
  tile_ids = labels.index.values
  stages   = labels['stage_str'].values
  
  feat = drop_high_cor(feat, 0.8)
  print('Features after high cor drop')
  # print(feat.shape)
  # print(feat.head())

  feat = feat.transform(lambda x: (x - np.mean(x)) / np.std(x))
  print('Features after zscore')
  # print(feat.shape)
  # print(feat.head())

  feat = feat.fillna(value=0)
  # feat = drop_nan_inf(feat)
  # print('Features after dropping nan and infs')
  # print(feat.shape)
  # print(feat.head())

  ((nepc_f, nepc_lab), (m0_f, m0_lab), (m0p_f, m0p_lab), (m1_f, m1_lab)) = split_sets(feat, labels)
  del feat

  if args.filter_stats:
    remove_cols = filter_stats(nepc_f, m0_f)
    nepc_f.drop(remove_cols, inplace=True, axis=1)
    m0_f.drop(remove_cols, inplace=True, axis=1)
    m0p_f.drop(remove_cols, inplace=True, axis=1)
    m1_f.drop(remove_cols, inplace=True, axis=1)

  train_x, train_y = make_training(m0_f, nepc_f)

  print('train_x', train_x.shape)
  print('train_y', train_y.shape)
  print('m1_f', m1_f.shape)
  # model = ElasticNet(alpha=1e-3, max_iter=10000).fit(train_x, train_y)
  # model = ElasticNetCV(cv=25).fit(train_x, train_y)
  # model = ElasticNetCV(alphas=np.arange(1e-5, 1e-1, 20), 
  #   cv=10, max_iter=20000, n_jobs=-1).fit(train_x, train_y)

  model = RandomForestRegressor(oob_score=True, 
                                max_features='sqrt', 
                                max_depth=20, 
                                n_estimators=50, 
                                n_jobs=-1).fit(train_x, train_y)

  with open('feature_importance.txt', 'w+') as f:
    for v, coef in zip(train_x.columns, model.feature_importances_):
      f.write('{}\t{}\n'.format(v, coef))

  if args.aggr_fn == 'max':
    aggr_fn = np.max
  elif args.aggr_fn == 'mean':
    aggr_fn = np.mean

  """ Predict the M1 cases and gather by max and mean """
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

  """ Check on the training data """
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
  nepc_cases = nepc_lab['case_id'].values
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

  """ Do some statistical tests """
  dotest = mannwhitneyu
  # test_args = {'equal_var': True}
  test_args = {}
  test_m0_m1   = dotest(yhat_train[train_y==0], yhat_m1, **test_args)
  test_m0_m0p  = dotest(yhat_train[train_y==0], yhat_m0p, **test_args)
  test_m0_nepc = dotest(yhat_train[train_y==0], yhat_train[train_y==1], **test_args)
  test_nepc_m1 = dotest(yhat_train[train_y==1], yhat_m1, **test_args)
  print('Tiles M0 vs M1',   test_m0_m1)
  print('Tiles M0 vs M0P',  test_m0_m0p)
  print('Tiles M0 vs NPEC', test_m0_nepc)
  print('Tiles NEPC vs M1', test_nepc_m1)

  test_m0_m1   = dotest(train_aggr[train_case_y==0], m1_case_aggr, **test_args)
  test_m0_m0p  = dotest(train_aggr[train_case_y==0], m0p_case_aggr, **test_args)
  test_m0_nepc = dotest(train_aggr[train_case_y==0], 
                        train_aggr[train_case_y==1], **test_args)
  test_nepc_m1 = dotest(train_aggr[train_case_y==1], m1_case_aggr, **test_args)
  print('aggr M0 vs M1', test_m0_m1)
  print('aggr M0 vs M0P', test_m0_m0p)
  print('aggr M0 vs NPEC', test_m0_nepc)
  print('aggr NEPC vs M1', test_nepc_m1)

  print('------------------------------------------------------------------------------------')
  gene_scores = pd.read_csv('../data/signature_scores_beltram.csv', index_col=None, header=0, sep=',')
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
        matching_scores.append(m1_case_aggr[m1_case_numbers==x][0])
      elif x in m0p_case_numbers:
        gene_score_caseid.append(x)
        matching_indices.append(idx)
        matching_scores.append(m0p_case_aggr[m0p_case_numbers==x][0])
      else:
        drop_rows.append(idx)
    except:
      drop_rows.append(idx)

  gene_scores.drop(drop_rows, inplace=True)
  gene_scores['NEPC Score'] = pd.Series(matching_scores, index=matching_indices)

  # if args.save_scores:
    # gene_scores.to_csv('../signature_scores_nepc_scores_nuclei_mean.csv')

  label_cols = ['caseid', 'Disease Stage', 'sample name', 'Surgical Number']
  gene_scores.drop(label_cols, inplace=True, axis=1)

  plt.figure(figsize=(5,5), dpi=300)
  sns.pairplot(gene_scores, kind='reg')
  if args.dry_run:
    pass
  else:
    plt.savefig('gene_scores_nepc_score_{}_tile.png'.format(args.aggr_fn), bbox_inches='tight')

  test_cols = [x for x in gene_scores.columns if x != 'NEPC Score']
  scores = gene_scores['NEPC Score'].values
  print('------------------------------------------------------------------------------------')
  for c in test_cols:
    ctest = spearmanr(scores, gene_scores[c].values)
    print('spearman {:40}: {:3.5f} p={:3.5f}'.format(c, ctest.correlation, ctest.pvalue))
    ctest = pearsonr(scores, gene_scores[c].values)
    print('pearson  {:40}: {:3.5f} p={:3.5f}'.format(c, ctest[0], ctest[1]))

  print('------------------------------------------------------------------------------------')
  if args.boxplot:
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.35, .65)})
    plt_m0 = train_aggr[train_case_y==0]
    plt_nepc = train_aggr[train_case_y==1]
    plt_m1 = m1_case_aggr
    plt_m0p = m0p_case_aggr
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
    if args.dry_run:
      pass
    else:
      plt.savefig('NEPC_score_{}_tile.png'.format(args.aggr_fn), bbox_inches='tight')

if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--src',    default='../data/nuclear_features_tile.csv')
  parser.add_argument('--labsrc', default='../data/case_stage_files.tsv')
  parser.add_argument('--boxplot', default=False, action='store_true')
  parser.add_argument('--aggr_fn', default='mean', type=str)
  parser.add_argument('--save_scores', default=False, action='store_true')
  parser.add_argument('--filter_stats', default=False, action='store_true')
  parser.add_argument('--dry_run', default=False, action='store_true')

  args = parser.parse_args()
  main(args)
