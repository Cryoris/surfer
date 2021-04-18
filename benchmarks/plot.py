import sys 
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print('Usage: plot.py datafile.npy')
    exit()

filename = sys.argv[1]
data = np.load(sys.argv[1], allow_pickle=True).item()
x = 4 * (2 ** np.arange(4)) + 4

plt.loglog()
for label, points in data.items():
    plt.errorbar(x, points['means'], yerr=points['std'], label=label)

plt.legend()
plt.show()