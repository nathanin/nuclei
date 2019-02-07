import numpy as np
# import modin.pandas as pd
import pandas as pd
import hashlib
import shutil
import glob
import os

from argparse import ArgumentParser
from scipy.stats import spearmanr, pearsonr
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

from utils import drop_high_cor, load_labels, load_features, split_sets
from statsmodels.stats.multitest import multipletests

def main(args):
  feat = load_features(args.src, zscore=True)
  lab = load_labels(args.labsrc)

  ((nepc_f, nepc_lab), (m0_f, m0_lab), (m0p_f, m0p_lab), (m1_f, m1_lab)) = split_sets(feat, lab)

  yvect = ['M0']*m0_f.shape[0] + ['NPEC']*nepc_f.shape[0]
  ttests = []
  fig = plt.figure()
  for f in feat.columns:
    m0_ = m0_f.loc[:, f]
    nepc_ = nepc_f.loc[:, f]
    m0p_ = m0p_f.loc[:, f]
    m1_ = m1_f.loc[:, f]
    tt = ttest_ind(m0_, nepc_)
    if tt.pvalue < 0.1:
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
  parser.add_argument('--src', default='../data/nuclear_features_tile.csv')
  parser.add_argument('--dst', default='ttests_tile')
  parser.add_argument('--labsrc',  default='../data/case_stage_files.tsv')

  args = parser.parse_args()
  main(args)
