import numpy as np

from qiskit.circuit.library import RealAmplitudes
from surfer.qfi import OverlapQFI

num_qubits = 8
reps = 3

circuit = RealAmplitudes(num_qubits, reps=reps)
values = np.zeros(circuit.num_parameters)
qfi = OverlapQFI(clifford=True).compute(circuit, values)
