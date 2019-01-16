from __future__ import print_function

import numpy as np
import argparse
import glob
import os

import seaborn as sns
from matplotlib import pyplot as plt
sns.set(style='whitegrid')
plt.style.use('seaborn-whitegrid')

import MulticoreTSNE as TSNE
from scipy.stats import ttest_ind, wilcoxon, mannwhitneyu

def load_features(stat_test=False):
  adeno = np.load('./adeno_features.npy')
  nepc  = np.load('./nepc_features.npy')

  if stat_test:
    print('Filtering features by ttest')
    keep = np.zeros(adeno.shape[1], dtype=np.bool)
    for i in range(adeno.shape[1]):
      res = mannwhitneyu(adeno[:,i], nepc[:,i])
      if res.pvalue < 1e-30:
        keep[i] = 1
    print('Keeping {} features'.format(keep.sum()))
    adeno = adeno[:, keep]
    nepc = nepc[:, keep]

  features = np.concatenate([adeno, nepc], axis=0)
  print('features:', features.shape)

  labels = np.asarray([0]*adeno.shape[0] + [1]*nepc.shape[0])
  print('labels:', labels.shape)

  return features, labels

def draw(z, y):
  for c in range(2):
    idx = y == c
    plt.scatter(z[idx, 0], z[idx, 1], label='{}'.format(c))

  plt.legend(frameon=True)
  plt.show()

def main(args):
  features, ys = load_features(stat_test=True)

  z = TSNE.MulticoreTSNE(n_jobs=-1).fit_transform(features)

  draw(z, ys)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--save', default='./tsne.png', type=str)

  args = parser.parse_args()
  main(args)
