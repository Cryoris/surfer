import numpy as np
import matplotlib.pyplot as plt

num_samples, error_mean, error_std = np.load("efficient_su2_n3_r2.npy")

plt.xlabel("number of samples")
plt.ylabel("frobenius distance")
plt.title("QFI error for EfficientSU2")
plt.loglog()
plt.errorbar(num_samples, error_mean, yerr=error_std)
plt.show()
