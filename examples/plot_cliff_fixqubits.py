import numpy as np
import matplotlib.pyplot as plt

plt.loglog()
plt.grid()

# reformat
files = [
    "./cliffsim_su2r1.npy",
    "./cliffsim_su2r2.npy",
    "./cliffsim_su2r3.npy",
    "./cliffsim_su2r4.npy",  # only 1 run, no stddev!
    "./cliffsim_su2r5.npy",
]
num_qubits = [2, 4, 8, 16]
means = []
stds = []

for f in files:
    data = np.load(f)
    means.append(data[1, :][: len(num_qubits)])
    stds.append(data[2, :][: len(num_qubits)])

means = np.vstack(means)
stds = np.vstack(stds)

print(means)

reps = [1, 2, 3, 4, 5]

for i, n in enumerate(num_qubits):
    plt.plot(reps, means[:, i], label=f"n = {n}")
    plt.fill_between(
        reps,
        means[:, i] - stds[:, i],
        means[:, i] + stds[:, i],
        alpha=0.3,
    )

plt.legend()
plt.show()


# def plot(filename, multiplier, color, label):
#     data = np.load(filename)
#     num_qubits = data[0, :]
#     num_parameters = multiplier * num_qubits
#     times = data[1, :]
#     std = data[2, :]

#     data = num_qubits if xaxis == "qubits" else num_parameters
#     mult = 1 if xaxis == "qubits" else multiplier

#     coeffs = np.polyfit(np.log(data), np.log(times), deg=2)

#     x = np.linspace(np.log(data[0]), np.log(mult * 42))
#     fit = np.polyval(coeffs, x)

#     plt.plot(data, times, color=color, label=label)
#     plt.plot(np.exp(x), np.exp(fit), color=color, ls="--", label="fit")
#     plt.fill_between(data, times - std, times + std, alpha=0.3, color=color)


# plot("./cliffsim.npy", 2, "crimson", "RealAmplitudes(r=1)")
# plot("./cliffsim_su2r1.npy", 4, "royalblue", "EfficientSU2(r=1)")

# singles = np.array([[40], [7418.504902124405]])
# if xaxis != "qubits":
#     singles[0, :] = 4 * singles[0, :]
# plt.plot(singles[0, :], singles[1, :], "o", color="royalblue")

# plot("./cliffsim_su2r3.npy", 8, "seagreen", "EfficientSU2(r=3)")
# plot("./cliffsim_su2r5.npy", 12, "goldenrod", "EfficientSU2(r=5)")

# plt.xlabel(f"number of {xaxis}")
# plt.ylabel("runtime in $s$")

# # plt.axhline(3600, color="k", ls="--", label="1 h")
# # plt.axhline(24 * 3600, color="k", ls="-.", label="1 day")
# # plt.axhline(30 * 24 * 3600, color="k", ls=":", label="1 month")
# # if xaxis == "qubits":
# #     plt.axvline(40, color="gray", ls=":")

# plt.legend(loc="best")

# plt.show()
