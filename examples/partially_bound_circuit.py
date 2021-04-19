import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from surfer.qfi import ReverseQFI

x = ParameterVector('x', 2)
circuit = QuantumCircuit(1)
circuit.rx(x[0], 0)
circuit.rx(x[1], 0)

circuit = circuit.bind_parameters({x[0]: 0})

values = np.ones(circuit.num_parameters)
print(circuit.num_parameters)

rev = ReverseQFI()
revd = rev.compute(circuit, values)

print("Reverse mode")
print(revd)
