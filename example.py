import numpy as np

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import RealAmplitudes

from qfi import StochasticApproximation, LinearCombination

# circuit = RealAmplitudes(1, reps=2)
# values = np.ones(circuit.num_parameters)
x = ParameterVector('x', 2)
circuit = QuantumCircuit(1)
circuit.rx(x[0], 0)
circuit.ry(x[1], 0)
values = np.array([0.25, 0])

# reference implementation
lc = LinearCombination()
reference = lc.compute(circuit, values)

print('Reference:')
print(reference)

# stochastic approximation
sa = StochasticApproximation(samples=1000, perturbation=0.1)
approximated = sa.compute(circuit, values)

print('Approximated:')
print(approximated)
