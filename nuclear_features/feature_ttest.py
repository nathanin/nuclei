import modin.pandas as pd
# import pandas as pd
from argparse import ArgumentParser
import numpy as np
import os
import hashlib

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon
from utils import (drop_high_cor, load_features, load_labels)

def main(args):
    data = pd.read_csv(args.src, index_col=0, memory_map=True)
    lab  = pd.read_csv(args.lab, index_col=0)
    print('DATA')
    print(data.shape)
    print('LAB')
    print(lab.shape)
    print(lab.head())

    data = data.sample(frac=args.pct)
    print(data.shape)
    # print(data.head())

    # Grab the id columns
    case_ids = data['case_id'].values
    tile_ids = data['tile_id'].values
    data.drop(['case_id', 'tile_id'], inplace=True, axis=1)
    print(data.shape)
    # print(data.head())

    data = drop_high_cor(data, cor_thresh = 0.7)
    print('Features after high cor drop')
    print(data.head())

    lab_case_uid = np.array(
        [hashlib.md5(x.encode()).hexdigest() for x in lab['case_id'].values]
    )
    is_nepc = np.zeros_like(case_ids, dtype=np.bool)
    not_nepc = np.zeros_like(case_ids, dtype=np.bool)
    for t_id in np.unique(case_ids):
        t_idx = case_ids == t_id
        print('{}: {} {}'.format(t_id, t_idx.shape, t_idx.sum()))

        assert t_id in lab_case_uid
        t_label = lab.loc[lab_case_uid == t_id].values
        t_label = t_label[0, -3]

        if t_label == 'NEPC':
            is_nepc[t_idx] = 1
        else:
            not_nepc[t_idx] = 1

    nepc_case_feat = data.loc[is_nepc,:].values
    adeno_case_feat = data.loc[not_nepc,:].values

    # nepc_case_feat = nepc_case_feat.sample(n=args.nsample).values
    # adeno_case_feat = adeno_case_feat.sample(n=args.nsample).values

    print('NEPC features:')
    print(nepc_case_feat.shape)
    print('Adeno features:')
    print(adeno_case_feat.shape)

    for c in range(nepc_case_feat.shape[1]):
        nepc_ = nepc_case_feat[:,c]
        adeno_ = adeno_case_feat[:,c]
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
    parser.add_argument('--src', default='../data/nuclear_features.csv', type=str)
    parser.add_argument('--lab', default='../data/case_stage_files.csv', type=str)
    parser.add_argument('--pct', default=0.05, type=float)
    parser.add_argument('--nsample', default=1000, type=float)
    parser.add_argument('--ae_only', default=False, action='store_true')
    parser.add_argument('--hc_only', default=False, action='store_true')
    parser.add_argument('--dst',    default='boxplots')
    parser.add_argument('--thresh', default=1e-5)

    args = parser.parse_args()
    main(args)