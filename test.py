import numpy as np
import matplotlib.pyplot as plt

test = np.load('density.npy', allow_pickle=True)
plt.hist(test)
plt.show()