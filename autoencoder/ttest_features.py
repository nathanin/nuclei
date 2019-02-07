import numpy as np
import argparse
import glob
import os

import seaborn as sns
from matplotlib import pyplot as plt
sns.set(style='whitegrid')
plt.style.use('seaborn-whitegrid')

from scipy.stats import ttest_ind

def load_features():
  adeno = np.load('./adeno_features.npy')
  nepc  = np.load('./nepc_features.npy')

  return adeno, nepc

def main(args):
  adeno, nepc = load_features()

  fig = plt.figure(figsize=(3,3), dpi=180)

  for i in range(adeno.shape[1]):
    res = ttest_ind(adeno[:,i], nepc[:,i])
    if res.pvalue < 1e-50:
      print(i, res)
      plt.clf()

      sns.distplot(adeno[:,i], label='adeno')
      sns.distplot(nepc[:,i], label='nepc')
      
      dst = os.path.join(args.save, 'ae_{:04d}.png'.format(i))
      plt.legend()
      plt.savefig(dst, bbox_inches='tight')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--save', default='ttest', type=str)

  args = parser.parse_args()
  main(args)
