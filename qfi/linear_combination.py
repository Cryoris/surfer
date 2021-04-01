import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.opflow import QFI, StateFn


class LinearCombination:
    """Compute the QFI using linear combination of unitaries."""

    def __init__(self):
        pass

    def compute(self,
                circuit: QuantumCircuit,
                values: np.ndarray) -> np.ndarray:
        """Compute the QFI.

        Args:
            circuit: A parameterized quantum circuit of which we compute the QFI.
            values: The parameter values at which the QFI is evaluated.
        """
        qfi = QFI()
        state = StateFn(circuit)
        param_dict = dict(zip(circuit.parameters, values))
        return qfi.convert(state).bind_parameters(param_dict).eval().real


def linear_combination(circuit: QuantumCircuit, values: np.ndarray) -> np.ndarray:
    """TODO"""
    lc = LinearCombination()
    return lc.compute(circuit, values)
