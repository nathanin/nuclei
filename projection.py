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

feature_dirs = {0: './nasnet_mobile/adeno',
                1: './nasnet_mobile/nepc'}

def draw(z, y):
    for c in range(2):
        idx = y == c
        plt.scatter(z[idx, 0], z[idx, 1], label=feature_dirs[c])

    plt.legend(frameon=True)
    plt.show()

def main(args):
    features = []
    ys = []
    for c in range(2):
        feature_list = glob.glob(os.path.join(feature_dirs[c], '*.npy'))
        ftrs = []
        for feature_path in feature_list:
            f = np.load(feature_path)
            ftrs.append(f)

        ftrs = np.stack(ftrs, axis=0)
        features.append(ftrs)
        ys.append(np.zeros(ftrs.shape[0]) + c)
        print('{}: {}'.format(c, ftrs.shape))

    features = np.concatenate(features, axis=0)
    ys = np.concatenate(ys, axis=0)
    print('features: {}'.format(features.shape))
    print('ys: {}'.format(ys.shape))

    z = TSNE.MulticoreTSNE(n_jobs=-1).fit_transform(features)

    draw(z, ys)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='./tsne.png', type=str)

    args = parser.parse_args()
    main(args)
