import sys
from time import time
import numpy as np
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.gradients import ReverseEstimatorGradient, ReverseQGT

from surfer.gradient.psr import ParameterShiftGradient
from surfer.qfi.psr import ParameterShiftQFI

n = int(sys.argv[1])
J = 0.5
hx = hz = 1

hamiltonian = SparsePauliOp.from_sparse_list(
    [("ZZ", (i, i + 1), J) for i in range(n - 1)]
    + [("X", [i], hx) for i in range(n)]
    + [("Z", [i], hx) for i in range(n)],
    num_qubits=n,
)

ansatz = RealAmplitudes(n, reps=1)
ansatz.x(ansatz.qubits)
ansatz.h(ansatz.qubits)
initial_parameters = np.zeros(ansatz.num_parameters)

psr = ParameterShiftGradient(clifford=True)
start = time()
grad = psr.compute(hamiltonian, ansatz, initial_parameters)
print("Clifford took:", time() - start)

# reference = ReverseEstimatorGradient()
# start = time()
# ref_grad = (
#     reference.run([ansatz], [hamiltonian], [initial_parameters])
#     .result()
#     .gradients[0]
#     .real
# )
# print("Reverse took:", time() - start)
# print(grad)
# print(np.linalg.norm(grad - ref_grad))

psr = ParameterShiftQFI(clifford=True)
start = time()
qgt = psr.compute(ansatz, initial_parameters) / 4
print("Clifford took:", time() - start)

# reference = ReverseQGT()
# start = time()
# ref_qgt = reference.run([ansatz], [initial_parameters]).result().qgts[0].real
# print("Reverse took:", time() - start)
# print(np.linalg.norm(qgt - ref_qgt))
