import numpy as np

from time import time
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import QFI, StateFn

from surfer.qfi import OverlapQFI, ReverseQFI

num_qubits = [2, 4, 8, 16]
reps = 4
runs = 5


def run_single(num_qubits, reps):
    circuit = EfficientSU2(num_qubits, reps=reps, entanglement="pairwise")
    values = np.zeros(circuit.num_parameters)
    start = time()
    qfi = OverlapQFI(clifford=True).compute(circuit, values)
    return time() - start


times = []
times_std = []
for n in num_qubits:
    results = [run_single(n, reps) for _ in range(runs)]
    times.append(np.mean(results))
    times_std.append(np.std(results))

    print(f"{n} qubits took {times[-1]}s +- {times_std[-1]}")

np.save("cliffsimv2_su2r4.npy", np.vstack((num_qubits, times, times_std)))
