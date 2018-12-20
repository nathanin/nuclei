# import pandas as pd
import modin.pandas as pd
import numpy as np
import argparse

from MulticoreTSNE import MulticoreTSNE
import umap

from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

def plot_embedded(emb, labels):
    unique_labels = np.unique(labels)
    print('Got {} unique labels'.format(len(unique_labels)))
    for ulab in unique_labels:
        idx = labels == ulab
        plt.scatter(emb[idx, 0], emb[idx, 1], label=ulab)

    plt.show()

def main(args):
    data = pd.read_csv(args.src, index_col=0)
    print(data.shape)

    data = data.sample(frac=args.pct)
    print(data.shape)
    print(data.head())

    # Grab the id columns
    case_id = data['case_id']
    tile_id = data['tile_id']
    data.drop(['case_id', 'tile_id'], inplace=True, axis=1)
    print(data.shape)
    print(data.head())

    if args.ae_only:
        to_drop = [x for x in data.columns if 'ae' not in x]
        data.drop(to_drop, axis=1, inplace=True)

    if args.hc_only:
        to_drop = [x for x in data.columns if 'hc' not in x]
        data.drop(to_drop, axis=1, inplace=True)

    data = data.transform(lambda x: (x - np.mean(x)) / np.std(x))
    print(data.shape)

    isinfs = np.sum(np.isinf(data.values), axis=0); print('isinfs', isinfs.shape)
    isnans = np.sum(np.isnan(data.values), axis=0); print('isnans', isnans.shape)
    print(np.argwhere(isinfs))
    print(np.argwhere(isnans))
    # data = data.dropna(axis='index')
    inf_cols = list(data.columns.values[np.squeeze(np.argwhere(isinfs))])
    nan_cols = list(data.columns.values[np.squeeze(np.argwhere(isnans))])
    print('inf_cols', inf_cols)
    print('nan_cols', nan_cols)
    data.drop(inf_cols, axis=1, inplace=True)
    data.drop(nan_cols, axis=1, inplace=True)
    print(data.shape)

    print(data.head())

    emb = MulticoreTSNE(n_jobs=-1, perplexity=100, 
                        learning_rate=500).fit_transform(data)
    # emb = umap.UMAP().fit_transform(data)

    plot_embedded(emb, case_id)
    plot_embedded(emb, tile_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='../data/nuclear_features.csv', type=str)
    parser.add_argument('--pct', default=0.01, type=float)
    parser.add_argument('--ae_only', default=False, action='store_true')
    parser.add_argument('--hc_only', default=False, action='store_true')
    args = parser.parse_args()
    main(args)