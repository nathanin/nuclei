import pandas as pd
# import modin.pandas as pd
import numpy as np
import argparse

from MulticoreTSNE import MulticoreTSNE
import umap

from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

from utils import drop_high_cor

def plot_embedded(emb, labels):
    unique_labels = np.unique(labels)
    print('Got {} unique labels'.format(len(unique_labels)))
    for ulab in unique_labels:
        idx = labels == ulab
        plt.scatter(emb[idx, 0], emb[idx, 1], label=ulab)

    plt.show()

def main(args):
    data = pd.read_csv(args.src, index_col=0, memory_map=True)
    lab  = pd.read_csv(args.lab)
    print(data.shape)
    print(lab.shape)
    print(lab.head())

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

    isinfs = np.sum(np.isinf(data.values), axis=0); print('isinfs', isinfs.shape)
    isnans = np.sum(np.isnan(data.values), axis=0); print('isnans', isnans.shape)
    print(np.argwhere(isinfs))
    print(np.argwhere(isnans))
    # data = data.dropna(axis='index')
    inf_cols = data.columns.values[np.squeeze(np.argwhere(isinfs))]
    nan_cols = data.columns.values[np.squeeze(np.argwhere(isnans))]
    print('inf_cols', inf_cols)
    print('nan_cols', nan_cols)
    data.drop(inf_cols, axis=1, inplace=True)
    data.drop(nan_cols, axis=1, inplace=True)
    print(data.shape)

    # Drop correlated columns
    data = drop_high_cor(data, 0.7)

    if args.average:
        print('Averaging features')
        if args.average_by == 'case':
            print('by: case')
            data = data.groupby(by=case_id, group_keys=True).mean()
            lab  = lab.groupby('case_id').max()
        elif args.average_by == 'tile':
            print('by: tile')
            data = data.groupby(by=tile_id, group_keys=True).mean()
            lab  = lab.groupby('tile_id').max()
        else:
            pass

        print(data.shape)
        print(data.head())
        print(lab.head())

        is_nepc = []
        for x, t in zip(lab['stage_str'].values, lab.index.values):
            if t in data.index:
                is_nepc.append(x == 'NEPC')
        is_nepc = np.array(is_nepc)

        print(is_nepc.shape)
    else:
        pass

    emb = MulticoreTSNE(n_jobs=-1).fit_transform(data)
    # emb = umap.UMAP().fit_transform(data)

    plot_embedded(emb, is_nepc)
    # plot_embedded(emb, tile_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='../data/nuclear_features.csv', type=str)
    parser.add_argument('--lab', default='../data/case_stage_files.csv', type=str)
    parser.add_argument('--pct', default=0.01, type=float)
    parser.add_argument('--ae_only', default=False, action='store_true')
    parser.add_argument('--hc_only', default=False, action='store_true')
    parser.add_argument('--average', default=False, action='store_true')
    parser.add_argument('--average_by', default='tile', type=str)
    args = parser.parse_args()
    main(args)