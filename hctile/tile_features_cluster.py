import pandas as pd
import numpy as np
import shutil
import glob
import os

import MulticoreTSNE as TSNE
from argparse import ArgumentParser
from scipy.stats import spearmanr
from matplotlib import pyplot as plt
plt.style.use('seaborn-whitegrid')

from utils import (drop_high_cor, load_features, load_labels)

def run_tsne(feat, lab):
    projection = TSNE.MulticoreTSNE(n_jobs=-1).fit_transform(feat)
    is_nepc = np.array(['NEPC' in x for x in lab['stage_str'].values])

    for c in range(2):
        idx = is_nepc == c
        print(idx.sum())
        plt.scatter(projection[idx,0], projection[idx,1], label='{}'.format(c))

    plt.legend(frameon=True)
    plt.show()

def main(args):
    feat, case_ids = load_features(args.src)
    lab  = load_labels(args.labsrc)

    feat = drop_high_cor(feat, cor_thresh = 0.8)
    print('Features after high cor drop')
    print(feat.head())

    run_tsne(feat, lab)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--src', default='../data/handcrafted_tile_features.csv')
    parser.add_argument('--labsrc', default='../data/case_stage_files.tsv')

    args = parser.parse_args()
    main(args)
