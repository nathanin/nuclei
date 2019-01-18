# import modin.pandas as pd
import pandas as pd

import numpy as np
import hashlib
import os

from argparse import ArgumentParser
from scipy.stats import spearmanr, pearsonr, ks_2samp
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

from utils import drop_high_cor, load_labels, load_features
# from statsmodels.stats.multitest import multipletests

nepc_strs = ['NEPC']
adeno_strs = ['M0 NP', 'M0 oligo poly', 'M0 oligo', 'M0 poly', 'M1 oligo poly',
              'M1 oligo', 'M1 poly', 'MX Diffuse', 'NXMX P']
m0_strs = ['M0 NP']
m0p_strs = ['M0 oligo poly', 'M0 oligo', 'M0 poly']
m1_strs = ['M1 oligo poly', 'M1 oligo', 'M1 poly']
def split_sets(feat, lab):
  """
  Return a tuple:
  (nepc_f, m0_f,... )
  """
  case_ids = feat['case_id'].values
  feat = feat.drop(['case_id', 'tile_id'], axis=1)
  print('case_ids:', case_ids.shape)

  def hashid(x):
    return hashlib.md5(x.encode()).hexdigest()
  case_ids_hashed = np.asarray([hashid(x) for x in lab['case_id'].values])
  print('case_ids_hashed:', case_ids_hashed.shape)

  # build case 2 stage dict
  case2stage = {}
  for cid in np.unique(case_ids):
    case_idx = case_ids_hashed == cid
    case_stage = lab.loc[case_idx, :].values
    case_stage = case_stage[0,7]
    print(cid, case_stage)
    case2stage[cid] = case_stage

  stages = [case2stage[x] for x in case_ids]

  is_nepc = np.array([x in nepc_strs for x in stages])
  is_m0   = np.array([x in m0_strs for x in stages])
  is_m0p  = np.array([x in m0p_strs for x in stages])
  is_m1   = np.array([x in m1_strs for x in stages])

  nepc_f = feat.loc[is_nepc, :]
  m0_f   = feat.loc[is_m0, :]  
  m0p_f  = feat.loc[is_m0p, :] 
  m1_f   = feat.loc[is_m1, :]  

  return nepc_f, m0_f, m0p_f, m1_f


def main(args):
  # feat = load_features(args.src, zscore=False)
  feat = pd.read_csv(args.src, index_col=0, header=0)
  lab = pd.read_csv(args.labsrc, index_col=0, header=0)
  print('features:', feat.shape)
  print('lab:', lab.shape)

  nepc_f, m0_f, m0p_f, m1_f = split_sets(feat, lab)
  print(nepc_f.shape)
  print(m0_f.shape)
  del feat

  if args.test == 'ttest':
    stat_test = ttest_ind
    get_p = lambda x: x.pvalue
  elif args.test == 'ks':
    stat_test = ks_2samp
    get_p = lambda x: x.pvalue

  yvect = ['M0']*m0_f.shape[0] + ['NPEC']*nepc_f.shape[0]
  test_results = []
  fig = plt.figure()
  for f in nepc_f.columns:
    m0_ = m0_f.loc[:, f]
    nepc_ = nepc_f.loc[:, f]
    m0p_ = m0p_f.loc[:, f]
    m1_ = m1_f.loc[:, f]
    tt = stat_test(m0_, nepc_)
    p = get_p(tt)
    if p < 1e-10:
      feature_data = pd.DataFrame({'group': yvect, 
        'feature': np.concatenate([m0_, nepc_], axis=0)})
      print(f, tt)
      out = os.path.join(args.dst, 'f_{}.png'.format(f))
      plt.clf()
      # sns.boxplot(x='group', y='feature', data=feature_data)
      sns.distplot(m0_, label='M0')
      sns.distplot(nepc_, label='NEPC')
      sns.distplot(m0p_, label='M0P')
      sns.distplot(m1_, label='M1')
      plt.legend()
      plt.title('Feature {}'.format(f))
      plt.savefig(out, bbox_inches='tight')

if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--dst', default='ttests')
  parser.add_argument('--src', default='../data/nuclear_features.csv')
  parser.add_argument('--test', default='ttest')
  parser.add_argument('--labsrc',  default='../data/case_stage_files.csv')

  args = parser.parse_args()
  main(args)
