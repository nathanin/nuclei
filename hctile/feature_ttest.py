import pandas as pd
from argparse import ArgumentParser
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon
from utils import (drop_high_cor, load_features, load_labels)

from statsmodels.stats.multitest import multipletests

nepc_strs = ['NEPC']
adeno_strs = ['M0 NP', 'M0 oligo poly', 'M0 oligo', 'M0 poly', 'M1 oligo poly',
              'M1 oligo', 'M1 poly', 'MX Diffuse', 'NXMX P']
m0_strs = ['M0 NP']
m0p_strs = ['M0 oligo poly', 'M0 oligo', 'M0 poly']
m1_strs = ['M1 oligo poly', 'M1 oligo', 'M1 poly']

def main(args):
    feat, case_ids = load_features(args.src, zscore=True)
    lab  = load_labels(args.labsrc)

    feat = drop_high_cor(feat, cor_thresh = 0.8)
    print('Features after high cor drop')
    print(feat.head())

    is_nepc = np.array([x in nepc_strs for x in lab['stage_str']])
    is_adeno = np.array([x in adeno_strs for x in lab['stage_str']])
    is_m0 = np.array([x in m0_strs for x in lab['stage_str']])
    is_m0p = np.array([x in m0p_strs for x in lab['stage_str']])
    is_m1 = np.array([x in m1_strs for x in lab['stage_str']])
    
    nepc_case_feat = feat.loc[is_nepc,:]; nepc_lab = lab.loc[is_nepc, :]
    adeno_case_feat = feat.loc[is_adeno,:]; adeno_lab = lab.loc[is_adeno, :]
    m0_case_feat = feat.loc[is_m0,:]; m0_lab = lab.loc[is_m0, :]
    m0p_case_feat = feat.loc[is_m0p,:]; m0p_lab = lab.loc[is_m0p, :]
    m1_case_feat = feat.loc[is_m1,:]; m1_lab = lab.loc[is_m1, :]

    if args.reduce_case:
        nepc_case_feat = nepc_case_feat.groupby(nepc_lab['case_id']).mean()
        adeno_case_feat = adeno_case_feat.groupby(adeno_lab['case_id']).mean()
        m0_case_feat = m0_case_feat.groupby(m0_lab['case_id']).mean()
        m0p_case_feat = m0p_case_feat.groupby(m0p_lab['case_id']).mean()
        m1_case_feat = m1_case_feat.groupby(m1_lab['case_id']).mean()

    print('NEPC features:', nepc_case_feat.shape)
    print('Adeno features:', adeno_case_feat.shape)
    print('M0 features:',  m0_case_feat.shape)
    print('M0p features:', m0p_case_feat.shape)
    print('M1 features:',  m1_case_feat.shape)

    nepc_adeno_p = []
    m0_m1_p = []
    m0_m0p_p = []
    for c in nepc_case_feat.columns:
        nepc_ = nepc_case_feat[c].values
        adeno_ = adeno_case_feat[c].values
        m0_ = m0_case_feat[c].values
        m0p_ = m0p_case_feat[c].values
        m1_ = m1_case_feat[c].values

        tt_nepc_adeno = ttest_ind(nepc_, adeno_)
        tt_m0_m1 = ttest_ind(m0_, m1_)
        tt_m0_m0p = ttest_ind(m0_, m0p_)

        nepc_adeno_p.append(tt_nepc_adeno[1])
        m0_m1_p.append(tt_m0_m1[1])
        m0_m0p_p.append(tt_m0_m0p[1])


    nepc_adeno_reject, nepc_adeno_q, _, _ = multipletests(nepc_adeno_p, alpha=0.01, method='fdr_bh')
    m0_m1_reject,  m0_m1_q,  _, _ = multipletests(m0_m1_p, alpha=0.01, method='fdr_bh')
    m0_m0p_reject, m0_m0p_q, _, _ = multipletests(m0_m0p_p,alpha=0.01,  method='fdr_bh')

    print('Rejecting {} '.format( np.sum(nepc_adeno_reject)) )
    print('Rejecting {} '.format( np.sum(m0_m1_reject)) )
    print('Rejecting {} '.format( np.sum(m0_m0p_reject)) )

    np.save('nepc_adeno_reject.npy', np.array(nepc_adeno_reject))
    np.save('m0_m1_reject.npy', np.array(m0_m1_reject))
    np.save('m0_m0p_reject.npy', np.array(m0_m0p_reject))

    for i, c in enumerate(nepc_case_feat.columns):
        print('plotting feature ', c)
        if not nepc_adeno_reject[i]:
            tt = True
        elif not m0_m1_reject[i]:
            tt = True
        elif not m0_m0p_reject[i]:
            tt = True
        else:
            tt = False

        if tt:
            nepc_ = nepc_case_feat[c].values
            adeno_ = adeno_case_feat[c].values
            m0_ = m0_case_feat[c].values
            m0p_ = m0p_case_feat[c].values
            m1_ = m1_case_feat[c].values

            plt.clf()
            sns.distplot(nepc_,  label='NEPC')
            sns.distplot(adeno_, label='Adeno')
            sns.distplot(m0_,    label='M0')
            sns.distplot(m0p_,   label='M0-P')
            sns.distplot(m1_,    label='M1')

            plt.legend(frameon=True)
            plt.title('{}\nnepc q={:.3E}\nm1 q={:.3E}\nm0p q={:.3E}'.format(
                c, nepc_adeno_q[i], m0_m1_q[i], m0_m0p_q[i]))

            saveto = os.path.join(args.dst, '{}.png'.format(c))
            plt.savefig(saveto, bbox_inches='tight')
        else:
            print('skipping feature ', c)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dst',    default='boxplots')
    parser.add_argument('--src',    default='../data/handcrafted_tile_features.csv')
    parser.add_argument('--labsrc', default='../data/case_stage_files.tsv')
    parser.add_argument('--thresh', default=1e-8)
    parser.add_argument('--reduce_case', default=False, action='store_true')

    args = parser.parse_args()
    main(args)