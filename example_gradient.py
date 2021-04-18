import numpy as np
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import Y, X, Z, Gradient, StateFn
from gradient import ForwardGradient, ReverseGradient

operator = X ^ X ^ X
ansatz = EfficientSU2(3, reps=1)
values = list(range(ansatz.num_parameters))

fwd = ForwardGradient()
rev = ReverseGradient()

print(fwd.compute(operator, ansatz, values))
print(rev.compute(operator, ansatz, values))

expectation = ~StateFn(operator) @ StateFn(ansatz)
print(
    np.real(
        Gradient()
        .convert(expectation)
        .bind_parameters(dict(zip(ansatz.parameters, values)))
        .eval()
    )
)
