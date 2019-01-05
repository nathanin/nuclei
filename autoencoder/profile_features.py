import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

x = np.load('./adeno_features.npy')

for i in range(x.shape[1]):
    sns.distplot(x[:, i], bins=50)
    plt.show()
