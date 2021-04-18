"""Compute the QFI using linear combination of unitaries."""

from typing import Optional
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.opflow import QFI, StateFn, CircuitSampler, ExpectationBase

from .qfi import QFICalculator


class LinearCombination(QFICalculator):
    """Compute the QFI using linear combination of unitaries."""

    def __init__(
        self,
        sampler: Optional[CircuitSampler] = None,
        expectation: Optional[ExpectationBase] = None,
        do_checks: bool = True,
    ):
        super().__init__(do_checks)
        self.sampler = sampler
        self.expectation = expectation

    def compute(self, circuit: QuantumCircuit, values: np.ndarray) -> np.ndarray:
        if self.do_checks:
            self.check_inputs(circuit, values)

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
    lc = LinearCombination()  # pylint: disable=invalid-name
    return lc.compute(circuit, values)
