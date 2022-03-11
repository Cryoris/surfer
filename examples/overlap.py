import numpy as np

from time import time
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import QFI, StateFn

from surfer.qfi import OverlapQFI

circuit = RealAmplitudes(10, reps=3)
values = np.zeros(circuit.num_parameters)
qfi = OverlapQFI(clifford=False).compute(circuit, values)
