import numpy as np
import pandas as pd

from argparse import ArgumentParser
from scipy.stats import spearmanr, pearsonr
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon

from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import roc_auc_score

m0_strs = ['M0-NM']
m0p_strs = ['M0-oligo', 'M0-poly']
m1_strs = ['M1-oligo', 'M1-poly']

def build_y(df):
  stage_strs = [x for x in df['Disease Stage'].values]

  y = np.zeros(len(stage_strs))
  for i, x in enumerate(stage_strs):
    if x in m0_strs:
      y[i] = 0
    elif x in m1_strs:
      y[i] = 1
    elif x in m0p_strs:
      y[i] = 2

  return y

def main():
  df = pd.read_csv('../data/signature_scores_all_nepc_scores.csv', header=0, index_col=0)
  print(df.columns)
  print(df.head())
  print(df.shape)

  print(df['Disease Stage'].unique())
  y = build_y(df)

  for i in range(3):
    print( i, (y == i).sum())

  is_m0m1 = y < 2
  m0p = y == 2
  train_y = y[is_m0m1]
  train_ne_jco = df.loc[is_m0m1, 'NE-JCO score']
  train_cin7 = df.loc[is_m0m1, 'z-score CIN7']
  train_cin70 = df.loc[is_m0m1, 'z-score CIN70']
  train_nuclei = df.loc[is_m0m1, 'nucleus_nepc_score']
  train_combined = df.loc[is_m0m1, 'combined_scores']

  test_ne_jco =   df.loc[m0p, 'NE-JCO score']
  test_cin7 =     df.loc[m0p, 'z-score CIN7']
  test_cin70 =    df.loc[m0p, 'z-score CIN70']
  test_nuclei =   df.loc[m0p, 'nucleus_nepc_score']
  test_combined = df.loc[m0p, 'combined_scores']


  col_names = ['NE-JCO score', 'z-score CIN7', 'z-score CIN70', 'nucleus_nepc_score', 'combined_scores']
  test_x = pd.concat([
    test_ne_jco,
    test_cin7,
    test_cin70,
    test_nuclei,
    test_combined,
  ], axis=1)

  train_x = pd.concat([
    train_ne_jco,
    train_cin7,
    train_cin70,
    train_nuclei,
    train_combined,
  ], axis=1)

  print(train_y)
  print(train_x.head())
  print(train_x.shape)

  model = ElasticNet(alpha=0.001, normalize=True).fit(train_x, train_y)
  print(model.coef_)
  print(model.intercept_)

  train_ypred = model.predict(train_x)
  test_ypred = model.predict(test_x)

  sns.distplot(train_ypred[train_y == 0], label='M0')
  sns.distplot(train_ypred[train_y == 1], label='M1')
  sns.distplot(test_ypred, label='M0-P')

  plt.legend()
  # plt.show()
  plt.savefig('multivariate_enet.png', bbox_inches='tight')

  with open('enet_coeffs.txt', 'w+') as f:
    for c, v in zip(col_names, model.coef_):
      f.write('{}\t{}\n'.format(c, v))


if __name__ == '__main__':
  main()