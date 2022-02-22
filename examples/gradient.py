import numpy as np
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import Y, X, Z, Gradient, StateFn
from surfer.gradient import ReverseGradient

operator = X ^ Y ^ Z
ansatz = EfficientSU2(3, reps=1)
values = np.random.random(ansatz.num_parameters)

# classical reverse gradient, scales linearly in number of single-parameter unitaries
rev = ReverseGradient()
rev_value = rev.compute(operator, ansatz, values)
print("Reverse:", rev_value)

# typical execution in Qiskit
expectation = StateFn(operator, is_measurement=True) @ StateFn(ansatz)
qiskit_value = np.real(
    Gradient()
    .convert(expectation)
    .bind_parameters(dict(zip(ansatz.parameters, values)))
    .eval()
)
print("Qiskit:", qiskit_value)

print("(L2) Difference:", np.linalg.norm(rev_value - qiskit_value))
