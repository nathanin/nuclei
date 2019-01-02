import pandas as pd
from argparse import ArgumentParser
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon
from utils import (drop_high_cor, load_features, load_labels)

def main(args):
    feat, case_ids = load_features(args.src, zscore=True)
    lab  = load_labels(args.labsrc)

    feat = drop_high_cor(feat, cor_thresh = 0.7)
    print('Features after high cor drop')
    print(feat.head())

    is_nepc = np.array(['NEPC' in x for x in lab['stage_str']])
    not_nepc = np.array(['NEPC' not in x for x in lab['stage_str']])
    nepc_case_feat = feat.loc[is_nepc,:]
    nepc_case_labs = lab.loc[is_nepc,:]

    adeno_case_feat = feat.loc[not_nepc,:]
    adeno_case_labs = lab.loc[not_nepc,:]

    print('NEPC features:')
    print(nepc_case_feat.shape)
    print('Adeno features:')
    print(adeno_case_feat.shape)


    for c in nepc_case_feat.columns:
        nepc_ = nepc_case_feat[c].values
        adeno_ = adeno_case_feat[c].values
        tt = ttest_ind(nepc_, adeno_)
        print('{}\t{:3.3f}\t{:3.3f}'.format(c, tt[0], tt[1]))
        if tt[1] < args.thresh:
            plt.clf()
            # df = pd.DataFrame({'NEPC': nepc_,
            #                    'Adeno': adeno_})
            sns.distplot(nepc_, label='NEPC')
            sns.distplot(adeno_, label='Adeno')
            plt.legend(frameon=True)
            plt.title('{}\np={}'.format(c, tt[1]))

            saveto = os.path.join(args.dst, '{}.png'.format(c))
            plt.savefig(saveto, bbox_inches='tight')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--src',    default='../data/handcrafted_tile_features.csv')
    parser.add_argument('--labsrc', default='../data/case_stage_files.tsv')
    parser.add_argument('--dst',    default='boxplots')
    parser.add_argument('--thresh', default=1e-8)

    args = parser.parse_args()
    main(args)