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
    feat = pd.read_csv(args.feature_src, index_col=None)
    case_ids = feat['case_id']
    tile_ids = feat['tile_id']
    stages   = feat['stage_str']
    feat.drop(['case_id', 'tile_id', 'stage_str'], axis=1, inplace=True)
    feat.drop([c for c in feat.columns if 'Unnamed' in c], axis=1, inplace=True)

    feat = feat.sample(frac=args.pct)
    case_ids = case_ids.loc[feat.index]
    tile_ids = tile_ids.loc[feat.index]
    stages   = stages.loc[feat.index]
    print(feat.shape)
    print(case_ids.shape)
    print(tile_ids.shape)

    feat = feat.loc[:, usecols]

    print('Dropping nan, inf and high corr')
    feat = drop_high_cor(feat, 0.8)
    feat = feat.transform(lambda x: (x - np.mean(x)) / np.std(x))
    feat = drop_nan_inf(feat)
    feat = drop_var(feat, 0.5)
    print(feat.shape)
    print(feat.head())

    if args.average == 'tile':
        print('Average by tile')
        feat = feat.groupby(by=tile_ids).mean()
        stages   = stages.groupby(by=tile_ids).max()
        print(feat.shape)
    elif args.average == 'case':
        print('Average by case')
        feat = feat.groupby(by=case_ids).mean()
        stages   = stages.groupby(by=case_ids).max()
        print(feat.shape)

    col_p = sns.color_palette('deep', 2)
    col_colors = [col_p[int('ae' in x)] for x in feat.columns]

    row_p = sns.color_palette('muted', 4)
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
            row_colors.append(row_p[3])
    
    print('col_colors', len(col_colors))
    print('row_colors', len(row_colors))

    # projected = TruncatedSVD(n_components=10).fit_transform(feat.values)
    # projected = PCA(n_components=10).fit_transform(feat.values)

    sns.clustermap(feat.values, 
                   metric=args.metric, 
                   standard_scale=1,
                   col_colors=col_colors,
                   row_colors=row_colors)
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dst', default='clustergram.png')
    parser.add_argument('--pct', default=0.01)
    parser.add_argument('--metric', default='euclidean')
    parser.add_argument('--average', default=None)
    parser.add_argument('--scores_src', default='../data/signature_scores_matched.csv')
    parser.add_argument('--feature_src', default='../data/nuclear_features.csv')

    args = parser.parse_args()
    main(args)