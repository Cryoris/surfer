"""Get an EfficientSU2 circuit with the parameter order assumed by Surfer's reverse gradients."""

from qiskit.circuit import ParameterVector
from qiskit.circuit.library import EfficientSU2


def efficient_su2(*args, **kwargs):
    """Get an EfficientSU2 circuit with the parameter order assumed by Surfer's reverse gradients."""
    ansatz = EfficientSU2(*args, **kwargs)
    theta = ParameterVector("th", ansatz.num_parameters)

    reordered = []
    num_qubits = ansatz.num_qubits
    reps = ansatz.reps
    for i in range(reps + 1):
        for j in range(num_qubits):
            reordered.append(theta[2 * j + 2 * num_qubits * i])
        for j in range(num_qubits):
            reordered.append(theta[2 * j + 1 + 2 * num_qubits * i])

    return ansatz.assign_parameters(reordered).decompose()
