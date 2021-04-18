from typing import Optional
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.opflow import QFI, StateFn, CircuitSampler, ExpectationBase


class LinearCombination:
    """Compute the QFI using linear combination of unitaries."""

    def __init__(
        self,
        sampler: Optional[CircuitSampler] = None,
        expectation: Optional[ExpectationBase] = None,
    ):
        self.sampler = sampler
        self.expectation = expectation

    def compute(self, circuit: QuantumCircuit, values: np.ndarray) -> np.ndarray:
        """Compute the QFI.

        Args:
            circuit: A parameterized quantum circuit of which we compute the QFI.
            values: The parameter values at which the QFI is evaluated.
        """
        state = StateFn(circuit)
        param_dict = dict(zip(circuit.parameters, values))
        qfi = QFI().convert(state)

        if self.expectation is not None:
            qfi = self.expectation.convert(qfi)

        if self.sampler is not None:
            qfi = self.sampler.convert(qfi, params=param_dict)
        else:
            qfi = qfi.bind_parameters(param_dict)

        return qfi.eval().real


def linear_combination(circuit: QuantumCircuit, values: np.ndarray) -> np.ndarray:
    """TODO"""
    lc = LinearCombination()
    return lc.compute(circuit, values)
