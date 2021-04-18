import numpy as np

from time import time
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import RealAmplitudes

from surfer.qfi import StochasticApproximation, LinearCombination, ReverseQFI

circuit = RealAmplitudes(7, reps=2)
values = np.ones(circuit.num_parameters)
# x = ParameterVector('x', 2)
# circuit = QuantumCircuit(1)
# circuit.ry(x[0], 0)
# circuit.ry(x[1], 0)
# values = np.array([0.5, 0])

# reference implementation
lc = LinearCombination()
start = time()
reference = lc.compute(circuit, values)

print(f"Reference: {time() - start}")
# print(reference)

# stochastic approximation
# sa = StochasticApproximation(samples=1000, perturbation=0.1)
# approximated = sa.compute(circuit, values)

# print('Approximated:')
# print(approximated)

rev = ReverseQFI()
start = time()
revd = rev.compute(circuit, values)

print(f"Reverse mode: {time() - start}")
print(np.mean(np.abs(revd - reference)))
