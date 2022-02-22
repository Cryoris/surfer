import numpy as np

from time import time
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import QFI, StateFn

from surfer.qfi import StochasticApproximation, LinearCombination, ReverseQFI

circuit = RealAmplitudes(5, reps=5)
values = np.ones(circuit.num_parameters)

# reference implementation with Qiskit
start = time()
qiskit_qfi = np.real(
    QFI()
    .convert(StateFn(circuit))
    .bind_parameters(dict(zip(circuit.parameters, values)))
    .eval()
)
print(f"Qiskit time: {time() - start}s")
# print(qiskit_qfi)

# stochastic approximation
rev = ReverseQFI()
start = time()
rev_qfi = rev.compute(circuit, values)
print(f"Reverse mode time: {time() - start}s")

print("(L2) Difference:", np.linalg.norm(rev_qfi - qiskit_qfi))
