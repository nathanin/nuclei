import pandas as pd
# import modin.pandas as pd
import numpy as np
import hashlib
import pickle
import shutil
import glob
import os

from joblib import dump, load

from scipy.stats import spearmanr, pearsonr
from matplotlib import pyplot as plt
from argparse import ArgumentParser
import seaborn as sns
sns.set(style='whitegrid')

from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon

from utils import drop_high_cor, drop_var, drop_nan_inf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import roc_auc_score

m0_strs = ['M0 NP']
m0p_strs = ['M0 oligo poly', 'M0 oligo', 'M0 poly']
m1_strs = ['M1 oligo poly', 'M1 oligo', 'M1 poly']
ignore_strs = ['MX Diffuse', 'NXMX P']
nepc_strs = ['NEPC']

def do_boxplot(plt_m0, plt_nepc, plt_m1, plt_m0p, dst):
  f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.35, .65)})
  sns.distplot(plt_m0,   bins=25, norm_hist=True, kde=True, label='M0', ax=ax_hist,)
  sns.distplot(plt_nepc, bins=25, norm_hist=True, kde=True, label='NEPC', ax=ax_hist,)
  sns.distplot(plt_m1,   bins=25, norm_hist=True, kde=True, label='M1', ax=ax_hist,)
  sns.distplot(plt_m0p,  bins=25, norm_hist=True, kde=True, label='M1', ax=ax_hist,)

  ax_hist.set_xlabel('Score')
  ax_hist.set_ylabel('Frequency')
  concat_scores = np.concatenate([plt_m0, plt_nepc, plt_m1, plt_m0p])
  concat_labels = np.array(['M0'] * len(plt_m0) + \
    ['NEPC'] * len(plt_nepc) + ['M1'] * len(plt_m1) + ['M0P'] * len(plt_m0p))
  plt_df = pd.DataFrame({'Set': concat_labels, 'Score': concat_scores})

  # fig = plt.figure(figsize=(2,2), dpi=300)
  sns.boxplot(y='Set', x='Score', data=plt_df, ax=ax_box)
  sns.stripplot(y='Set', x='Score', data=plt_df, size=2.5, jitter=True, linewidth=0.5, ax=ax_box)
  plt.savefig(dst,  bbox_inches='tight')

def get_y(nuclei_case_ids, labels):
  # case dict
  case_ids = labels['case_id'].values
  case_ys  = labels['stage_str'].values

  case_dict = {}
  for cid in np.unique(case_ids):
    case_y_id = case_ys[case_ids == cid][0]
    cid_hex = hashlib.md5(cid.encode()).hexdigest()
    print(cid_hex, case_y_id, end=' ')
    if case_y_id in m0_strs:
      case_dict[cid_hex] = 0
    elif case_y_id in nepc_strs:
      case_dict[cid_hex] = 1 # for convenience
    elif case_y_id in m0p_strs:
      case_dict[cid_hex] = 2
    elif case_y_id in m1_strs:
      case_dict[cid_hex] = 3
    else:
      case_dict[cid_hex] = 4

  # populate y according to case dict
  yvect = np.zeros(len(nuclei_case_ids))
  for i, ncid in enumerate(nuclei_case_ids):
    # ncid_hex = hashlib.md5(ncid.encode()).hexdigest()
    yvect[i] = case_dict[ncid]

  for i in range(3):
    print('\t{} = {}'.format(i, np.sum(yvect==i)))
  return yvect

def train(args):
  feat = pd.read_csv(args.src, index_col=0, header=0)
  print(feat.head())
  print(feat.shape)
  labels = pd.read_csv(args.labsrc, sep='\t')
  print(labels.shape)

  yvect = get_y(feat['case_id'], labels)
  print(yvect.shape)

  # Drop rows that come from cases we want to exclude
  usable_data = yvect < 4
  yvect = yvect[usable_data]
  print(yvect.shape)

  feat = feat.loc[usable_data, :]
  nuclei_case_ids = feat['case_id']
  nuclei_tile_ids = feat['tile_id']
  feat.drop(['case_id', 'tile_id'], axis=1, inplace=True)
  print('dropped label cols', feat.shape)

  # drop_cols = [x for x in feat.columns if 'hc' not in x]
  # feat.drop(drop_cols, inplace=True, axis=1)
  # print('dropped chosen cols', feat.shape)

  # Drop columns of features
  feat = drop_var(feat)
  print('dropped low var', feat.shape)

  feat = drop_high_cor(feat, 0.8)
  print('dropped corr', feat.shape)

  feat = drop_nan_inf(feat)
  print('dropped nan inf', feat.shape)

  feat = feat.transform(lambda x: (x - np.mean(x)) / np.std(x))
  print(feat.head())
  print(feat.shape)

  # Split off M1
  m1rows = yvect == 2
  m0nepc_rows = yvect < 2
  yvect_m0nepc = yvect[m0nepc_rows]
  feat_m0nepc = feat.loc[m0nepc_rows, :]
  feat_m1 = feat.loc[m1rows, :]
  del feat, yvect

  train_idx, test_idx = train_test_split(np.arange(len(yvect_m0nepc)))
  train_x = feat_m0nepc.iloc[train_idx, :]
  train_y = yvect_m0nepc[train_idx]
  test_x = feat_m0nepc.iloc[test_idx, :]
  test_y = yvect_m0nepc[test_idx]
  print(train_x.shape)
  print(test_x.shape)
  model = RandomForestRegressor(max_depth=25, 
                                max_features='sqrt', 
                                n_estimators=100, 
                                n_jobs=-1).fit(train_x, train_y)

  ypred = model.predict(test_x)
  print(ypred.shape)
  print(ypred.mean())
  print(ypred)

  m1pred = model.predict(feat_m1)

  plt_m0 = ypred[test_y == 0]
  plt_nepc = ypred[test_y == 1]
  plt_m1 = m1pred
  dst = 'nucleus_classifier_features.npy'
  do_boxplot(plt_m0, plt_nepc, plt_m1, args.figout)

  dump(model, args.save)
  np.save('nucleus_classifier_features.npy', train_x.columns.values)

def test(args):

  model = load(args.load)
  # use_cols = pickle.load(open('nucleus_classifier_features.pkl', 'r'))
  use_cols = np.load('nucleus_classifier_features.npy')

  # print out the list of feature importances
  perm = np.argsort(model.feature_importances_)[::-1]
  fimps  = model.feature_importances_[perm]
  fnames = use_cols[perm]
  with open('feature_importance_single_nuclei.txt', 'w+') as f:
    for fn, fi in zip(fnames, fimps):
      f.write('{}\t{}\n'.format(fn, fi))

  feat = pd.read_csv(args.src, index_col=0, header=0)
  labels = pd.read_csv(args.labsrc, sep='\t')

  yvect = get_y(feat['case_id'], labels)
  use_rows = yvect < 4
  yvect = yvect[use_rows]
  feat = feat.iloc[use_rows, :]
  print(yvect.shape)
  print(feat.shape)
  for i in range(3):
    print('\t{} = {}'.format(i, (yvect==i).sum()))
  nuclei_case_ids = feat['case_id'].values
  nuclei_tile_ids = feat['tile_id'].values
  print(nuclei_case_ids.shape)

  feat = feat.loc[:, use_cols]
  feat = feat.transform(lambda x: (x - np.mean(x) / np.std(x)))
  print(feat.shape)

  ypred = model.predict(feat)
  plt.figure(figsize=(2,2), dpi=180)
  sns.distplot(ypred[yvect==0], bins=30, label='Adeno')
  sns.distplot(ypred[yvect==1], bins=30, label='NEPC')
  sns.distplot(ypred[yvect==2], bins=30, label='M0P')
  sns.distplot(ypred[yvect==3], bins=30, label='M1')
  plt.title('Nucleus NEPC scores')
  plt.legend(loc=2)
  plt.savefig(args.figout, bbox_inches='tight')

  # Squish case
  aggr_fn = np.mean
  case_aggr, case_labels, case_ids_ = [], [], []
  for k in np.unique(nuclei_case_ids):
    print(k)
    ix = nuclei_case_ids == k
    y_true_ix = yvect[ix][0]
    y_pred_ix = aggr_fn(ypred[ix])
    case_aggr.append(y_pred_ix)
    case_labels.append(y_true_ix)
    case_ids_.append(k)

  case_aggr = np.array(case_aggr)
  case_labels = np.array(case_labels)
  print(case_aggr)
  print(case_labels)

  case_labels_bin = case_labels[case_labels < 2]
  case_aggr_bin = case_aggr[case_labels < 2]
  auc_ = roc_auc_score(y_true=case_labels_bin, y_score=case_aggr_bin)
  print('M0 - NEPC AUC = ', auc_)

  m0m1 = case_aggr[ (case_labels==0) + (case_labels==3) ]
  m0m1_y = case_labels[ (case_labels==0) + (case_labels==3) ]
  auc_ = roc_auc_score(y_true=m0m1_y, y_score=m0m1)
  print('M0 - M1 AUC = ', auc_)

  m0m0p = case_aggr[ (case_labels==0) + (case_labels==2) ]
  m0m0p_y = case_labels[ (case_labels==0) + (case_labels==2) ]
  auc_ = roc_auc_score(y_true=m0m0p_y, y_score=m0m0p)
  print('M0 - M0P AUC = ', auc_)

  plt_m0   = case_aggr[case_labels==0]
  plt_nepc = case_aggr[case_labels==1]
  plt_m0p  = case_aggr[case_labels==2]
  plt_m1   = case_aggr[case_labels==3]
  dst = 'nucleus_classifier_features.npy'
  do_boxplot(plt_m0, plt_nepc, plt_m1, plt_m0p, 'nepc_score_case_distplot.png')

  if args.save_scores:
    with open('nepc_case_scores.txt', 'w+') as f:
      for cid, cagg in zip(case_ids_, case_aggr):
        s = '{}\t{}\n'.format(cid, cagg)
        f.write(s)

if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--src',    default='../data/nuclear_features_sample.csv')
  parser.add_argument('--test',   default=False, action='store_true')
  parser.add_argument('--save',   default= 'nucleus_classifier.joblib', type=str)
  parser.add_argument('--load',   default= 'nucleus_classifier.joblib', type=str)
  parser.add_argument('--figout', default= 'nepc_score_distplot.png', type=str)
  parser.add_argument('--labsrc', default='../data/case_stage_files.tsv')
  parser.add_argument('--aggr_fn', default='mean', type=str)
  parser.add_argument('--save_scores', default=False, action='store_true')

  args = parser.parse_args()

  if args.test:
    test(args)
  else:
    train(args)