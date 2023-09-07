"""Compute the QFI using linear combination of unitaries."""

from __future__ import annotations
import numpy as np

from qiskit.primitives import Estimator, BaseEstimator
from qiskit.circuit import QuantumCircuit
from qiskit_algorithms.gradients import LinCombQGT

from .qfi import QFICalculator


class LinearCombination(QFICalculator):
    """Compute the QFI using linear combination of unitaries."""

    def __init__(
        self,
        estimator: BaseEstimator | None = None,
        do_checks: bool = True,
    ):
        super().__init__(do_checks)
        if estimator is None:
            estimator = Estimator()

        self.qgt = LinCombQGT(estimator)

    def compute(self, circuit: QuantumCircuit, values: np.ndarray) -> np.ndarray:
        if self.do_checks:
            self.check_inputs(circuit, values)

        qgt = self.qgt.run([circuit], [values]).result().qgts[0]
        qfi = 4 * np.real(qgt)
        return qfi


def linear_combination(circuit: QuantumCircuit, values: np.ndarray) -> np.ndarray:
    """TODO"""
    lc = LinearCombination()  # pylint: disable=invalid-name
    return lc.compute(circuit, values)
