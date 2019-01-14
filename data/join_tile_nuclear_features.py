import pandas as pd
import numpy as np

hctile = pd.read_csv('handcrafted_tile_features.csv', header=0, index_col=0)
nuclear = pd.read_csv('nuclear_features_tile.csv', header=0, index_col=0)
hctile.drop('case_id', axis=1, inplace=True)
hctile.columns = [x + 'tile' for x in hctile.columns]

print(hctile.head())
print(hctile.shape)
print(nuclear.head())
print(nuclear.shape)

joined = pd.concat([hctile, nuclear], join='inner', axis=1)
print(joined.head())
print(joined.shape)

joined.to_csv('joined_features.csv')