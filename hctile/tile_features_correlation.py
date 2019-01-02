import numpy as np
# import modin.pandas as pd
import pandas as pd
import hashlib
import shutil
import glob
import os

from argparse import ArgumentParser
from scipy.stats import spearmanr, pearsonr
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

from utils import drop_high_cor

def translate_sn2hash(x):
    if x == '0':
        return 'drop_me'
    parts = x.split(' ')
    case_id = '{} {}-{}'.format(*parts)
    cid_hash = hashlib.md5(case_id.encode()).hexdigest()
    print(x, case_id, cid_hash)
    return hashlib.md5(case_id.encode()).hexdigest()

# https://stackoverflow.com/questions/7450957/how-to-implement-rs-p-adjust-in-python
def p_adjust_bh(p):
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]


def main(args):
    scores = pd.read_csv(args.scores_src)
    scores_caseids = scores['Surgical Number']
    scores_caseids = np.array([translate_sn2hash(x) for x in scores_caseids])
    drop_rows = np.squeeze(scores.index.values[scores_caseids == 'drop_me'])
    print('Dropping: ', drop_rows)
    scores['case_id'] = scores_caseids
    scores.drop(drop_rows, inplace=True)
    print(scores.head())
    print(scores.shape)

    features = pd.read_csv(args.feature_src, index_col=0)
    print('Features')
    print(features.head())
    print(features.shape)

    caseids = features['case_id'].values
    features.drop('case_id', axis=1, inplace=True)
    # features = drop_high_cor(features, 0.8)
    remaining_features = features.columns

    indices = []
    feature_case_mean = []
    for cid in np.unique(caseids):
        cid_idx = caseids == cid
        f = features.loc[cid_idx, :].values
        fmean = np.mean(f, axis=0)
        print('{}:'.format(cid), fmean.shape)
        feature_case_mean.append(np.expand_dims(fmean, axis=0))
        indices.append(cid)
    
    features = pd.DataFrame(np.concatenate(feature_case_mean, axis=0), columns=remaining_features)
    features['case_id'] = indices
    print('Features grouped by case')
    print(features.head())
    print(features.shape)

    matching_indices = np.intersect1d(features['case_id'], scores['case_id'])
    print('Matched indices:', matching_indices, len(matching_indices))

    # Drop rows from features and scores -- then sort them
    drop_rows = [x for x,c in \
        zip(features.index.values, features['case_id']) if c not in matching_indices]
    features.drop(drop_rows, axis=0, inplace=True)
    print('FEATURES BEFORE SORTING\n', features.head())
    features.sort_values(by='case_id', inplace=True)
    sorted_caseids_features = features['case_id'].values
    features.drop('case_id', axis=1, inplace=True)
    print('FEATURES AFTER SORTING\n', features.head())
    features = features.transform(lambda x: (x - np.mean(x)) / np.std(x))
    print(features.shape)

    drop_rows = [x for x,c in \
        zip(scores.index.values, scores['case_id'].values) if c not in matching_indices]
    scores.drop(drop_rows, axis=0, inplace=True)
    # shuffle columns
    # scores.index = scores['case_id'].values
    print('SCORES BEFORE SORTING\n', scores.head())
    scores.sort_values(by='case_id', inplace=True)
    sorted_caseids_scores = scores['case_id'].values
    to_drop = ['case_id', 'caseid', 'Disease Stage', 'sample name', 'Surgical Number']
    scores.drop(to_drop, axis=1, inplace=True)
    print('SCORES AFTER SORTING\n', scores.head())
    print(scores.shape)

    for fid, sid in zip(sorted_caseids_features, sorted_caseids_scores):
        print(fid, sid)
        assert fid == sid

    fig = plt.figure(figsize=(2,2), dpi=300)

    logfile = os.path.join(args.dst, 'qvalues.csv')
    comparison_ids = []
    pvalues = []
    # with open(logfile, 'w+') as f:
    for c in features.columns:
        cx = features[c].values
        for s in scores.columns:
            sy = scores[s].values
            corr = spearmanr(cx, sy)
            pcorr = pearsonr(cx, sy)
            comparison_ids.append('{}_{}'.format(c, s))
            pvalues.append(corr.pvalue)
            if corr.pvalue < 0.001:
                outstr = '*{}\t{}\tr={:3.3f}\tp={:3.3f}\tpr={:3.3f}\tpp={:3.3f}'.format(
                    c, s, corr.correlation, corr.pvalue, pcorr[0], pcorr[1])
                plt.clf()
                plt.scatter(cx, sy)
                plt.title('sr={:3.3f} sp={:3.3f}\npr={:3.3f} pp={:3.3f}'.format(
                        corr.correlation, corr.pvalue,
                        pcorr[0], pcorr[1],
                        ))
                plt.xlabel(c)
                plt.ylabel(s)
                # plt.legend(frameon=True)
                plt.savefig(os.path.join(args.dst, '{}_{}.png'.format(c, s)), bbox_inches='tight')
            else:
                outstr = ' {}\t{}\tr={:3.3f}\tp={:3.3f}\tpr={:3.3f}\tpp={:3.3f}'.format(
                    c, s, corr.correlation, corr.pvalue, pcorr[0], pcorr[1])

            print(outstr)
            # f.write(outstr+'\n')

    qvalues = p_adjust_bh(pvalues)
    qdf = pd.DataFrame({'q': qvalues}, index=comparison_ids)
    qdf.sort_values('q', inplace=True)
    qdf.to_csv(logfile)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--scores_src', default='../data/signature_scores_matched.csv')
    parser.add_argument('--feature_src', default='../data/handcrafted_tile_features.csv')
    parser.add_argument('--dst', default='correlations')

    args = parser.parse_args()
    main(args)