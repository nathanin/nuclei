# import numpy as np
import modin.pandas as pd
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

from sklearn.decomposition import TruncatedSVD, PCA
from utils import drop_high_cor, drop_nan_inf, drop_var

m0_strs = ['M0 NP']
m1_strs = ['M1 oligo poly', 'M1 oligo', 'M1 poly']

usecols = [os.path.basename(x) for x in glob.glob('boxplots/*.png')]
usecols = [x.replace('.png', '') for x in usecols]

def main(args):
    feat = pd.read_csv(args.feature_src, index_col=0)
    lab  = pd.read_csv(args.label_src)
    case_ids = feat['case_id']
    tile_ids = feat.index
    stages   = lab['stage_str']
    feat.drop(['case_id'], axis=1, inplace=True)
    # feat.drop([c for c in feat.columns if 'Unnamed' in c], axis=1, inplace=True)

    feat = feat.loc[:, usecols]

    # case_ids = case_ids.loc[feat.index]
    # tile_ids = tile_ids.loc[feat.index]
    # stages   = stages.loc[feat.index]
    print(feat.shape)
    print(case_ids.shape)
    print(tile_ids.shape)
    print(stages.shape)

    print('Dropping nan, inf and high corr')
    feat = drop_high_cor(feat, 0.8)
    feat = feat.transform(lambda x: (x - np.mean(x)) / np.std(x))
    feat = drop_nan_inf(feat)
    feat = drop_var(feat, 0.5)
    print(feat.shape)
    print(feat.head())

    if args.average == 'case':
        print('Average by case')
        feat = feat.groupby(by=case_ids.values).mean()
        stages   = stages.groupby(by=case_ids.values).max()

    print(feat.shape)
    print(stages.shape)

    row_p = sns.color_palette('muted', 3)
    row_colors = []
    print(np.unique(stages.values))
    for s in stages.values:
        if s in m0_strs:
            row_colors.append(row_p[0])
        elif s in m1_strs:
            row_colors.append(row_p[1])
        elif 'NEPC' in s:
            row_colors.append(row_p[2])
        else:
            row_colors.append(row_p[1])
    
    print('row_colors', len(row_colors))

    # projected = TruncatedSVD(n_components=10).fit_transform(feat.values)
    # projected = PCA(n_components=10).fit_transform(feat.values)

    sns.clustermap(feat.values, 
                   metric=args.metric, 
                   standard_scale=1,
                   row_colors=row_colors)
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--scores_src',  default='../data/signature_scores_matched.csv')
    parser.add_argument('--feature_src', default='../data/handcrafted_tile_features.csv')
    parser.add_argument('--label_src',   default='../data/case_stage_files.csv')
    parser.add_argument('--dst',     default='clustergram.png')
    parser.add_argument('--average', default=None)
    parser.add_argument('--metric',  default='euclidean')

    args = parser.parse_args()
    main(args)