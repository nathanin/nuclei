import pandas as pd
import modin.pandas as pd
import numpy as np
from MulticoreTSNE import MulticoreTSNE
import argparse

from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

def plot_tsne(tsne, labels):

    for ulab in np.unique(labels):
        idx = labels == ulab
        plt.scatter(tsne[idx, 0], tsne[idx, 1], label=ulab)

    plt.show()

def main(args):
    data = pd.read_csv(args.src, index_col=0)
    print(data.shape)

    # Grab the id columns
    case_id = data['case_id'].values
    tile_id = data['tile_id'].values
    data.drop(['case_id', 'tile_id'], inplace=True, axis=1)
    print(data.shape)

    tsne = MulticoreTSNE(n_jobs=-1).fit_transform(data)

    plot_tsne(tsne, case_id)
    plot_tsne(tsne, tile_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='../data/nuclear_features.csv', type=str)
    args = parser.parse_args()
    main(args)