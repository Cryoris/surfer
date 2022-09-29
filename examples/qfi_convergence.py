import numpy as np
from surfer.qfi.reverse_qfi import ReverseQFI
from surfer.qfi.smoothed_approximation import SmoothedApproximation


def circuit1():
    from qiskit.circuit import QuantumCircuit, ParameterVector

    x = ParameterVector("x", 2)
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.rz(x[0], 0)
    circuit.rx(x[1], 0)
    values = np.array([1.23, 2.01])

    return circuit, values


def circuit2():
    from surfer.tools.efficient_su2 import efficient_su2

    circuit = efficient_su2(num_qubits=3, reps=2)
    values = np.arange(1, circuit.num_parameters + 1) / circuit.num_parameters

    return circuit, values


def error(approximation):
    return np.linalg.norm(exact - approximation)


circuit, values = circuit2()

exact = ReverseQFI().compute(circuit, values)

num_batches = 200
batch_size = 10
num_reps = 20


num_samples = batch_size * np.arange(1, num_batches + 1)
all_errors = np.empty((num_reps, num_batches))

for i in range(num_reps):
    approximate = SmoothedApproximation(circuit, values, batch_size, perturbation=0.01)

    for j in range(num_batches):
        approximate.sample()
        all_errors[i, j] = error(approximate.point_estimate)

mean_errors = np.mean(all_errors, axis=0)
std_errors = np.std(all_errors, axis=0)

print(num_samples)
print(mean_errors)
print(std_errors)

np.save("efficient_su2_n2_r1", [num_samples, mean_errors, std_errors])
