import pandas as pd
import numpy as np
import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

nepc_strs = ['NEPC']
adeno_strs = ['M0 NP', 'M0 oligo poly', 'M0 oligo', 'M0 poly', 'M1 oligo poly',
              'M1 oligo', 'M1 poly', 'MX Diffuse', 'NXMX P']
m0_strs = ['M0 NP']
m0p_strs = ['M0 oligo poly', 'M0 oligo', 'M0 poly']
m1_strs = ['M1 oligo poly', 'M1 oligo', 'M1 poly']

def main(args):
  feat_importance = pd.read_csv(args.src, sep='\t', index_col=0, header=None)
  # features , _ = load_features(args.featsrc, zscore=True)
  # labels = load_labels(args.labelsrc)
  
  feat_importance.sort_values(1, ascending=False, inplace=True)
  sns.distplot(feat_importance)
  plt.savefig('combined_model_feature_importance_dist.png', bbox_inches='tight')

  feat_importance = feat_importance.iloc[:args.n, :]
  print('highest feature importance:')
  for f in feat_importance.index.values:
    print(f, feat_importance.loc[f].values)
  
  # usecols = [str(f) for f in feat_importance.index.values]

  # features = features.loc[:, usecols]
  # print(features.shape)

  # all_labels = np.unique(labels['stage_str'].values); print(len(all_labels))
  # palette = sns.color_palette(n_colors=len(all_labels))
  # palette_dict = {l: palette[k] for k, l in enumerate(all_labels)}
  # row_colors = [palette_dict[l] for l in labels['stage_str'].values]

  # sns.clustermap(features, metric='cosine', row_colors=row_colors)
  # plt.savefig('feature_clustermap.png', bbox_inches='tight')

  # sns.clustermap(features.corr())
  # plt.savefig('feature_correlations.png', bbox_inches='tight')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('src', type=str)

  parser.add_argument('-n', default=10, type=int)
  # parser.add_argument('--featsrc', default='../data/handcrafted_tile_features.csv', type=str)
  # parser.add_argument('--labelsrc', default='../data/case_stage_files.tsv', type=str)

  args = parser.parse_args()
  main(args)
