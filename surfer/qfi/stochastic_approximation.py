"""Stochastic approximation of the Quantum Fisher Approximation."""

from typing import Union, Callable, Tuple
import numpy as np

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.opflow import StateFn

from .qfi import QFICalculator


class StochasticApproximation(QFICalculator):
    """Stochastic approximation of the Quantum Fisher Approximation."""

    def __init__(self, samples: int, perturbation: float, do_checks: bool = True):
        """
        Args:
            The number of samples.
        """
        super().__init__(do_checks)
        self.samples = samples
        self.perturbation = perturbation

    def compute(
        self,
        circuit: Union[
            Callable[[Tuple[np.ndarray, np.ndarray]], float], QuantumCircuit
        ],
        values: np.ndarray,
    ):
        """Compute the QFI.

        Args:
            circuit: A callable to evaluate the fidelity or a parameterized quantum circuit of
                which we compute the QFI.
            values: The parameter values at which the QFI is evaluated.
        """
        # wrap into the fidelity if it's a circuit
        if isinstance(circuit, QuantumCircuit):
            if self.do_checks:
                self.check_inputs(circuit, values)

            fidelity = get_fidelity(circuit)
        else:
            fidelity = circuit

        # set up variables to store averages
        estimate = np.zeros((values.size, values.size))

        # iterate over the directions
        for _ in range(self.samples):
            estimate += self._point_sample(fidelity, values, self.perturbation)

        return estimate / self.samples

    @staticmethod
    def _point_sample(fidelity, values, perturbation):
        delta1 = 1 - 2 * np.random.binomial(1, 0.5, values.size)
        delta2 = 1 - 2 * np.random.binomial(1, 0.5, values.size)
        pert1 = perturbation * delta1
        pert2 = perturbation * delta2

        x_pp = fidelity(values, values + pert1 + pert2)
        x_pm = fidelity(values, values + pert1 - pert2)
        x_mp = fidelity(values, values - pert1 + pert2)
        x_mm = fidelity(values, values - pert1 - pert2)

        # compute the preconditioner point estimate
        diff = x_pp - x_pm - x_mp + x_mm
        diff /= 4 * perturbation ** 2

        rank_one = np.outer(delta1, delta2)
        sample = diff * (rank_one + rank_one.T) / 2

        # factor -0.5 comes from FS metric, factor 4 from QFI magic definitionzz
        return -2 * sample


# pylint: disable=invalid-name
def get_fidelity(circuit: QuantumCircuit):
    """Convenience function to evaluate the fidelity."""
    x = ParameterVector("x", circuit.num_parameters)
    y = ParameterVector("y", circuit.num_parameters)
    parameters = x[:] + y[:]  # list of parameters to bind
    overlap = ~StateFn(circuit.assign_parameters(x)) @ StateFn(
        circuit.assign_parameters(y)
    )

    def fidelity(x_, y_):
        param_dict = dict(zip(parameters, x_.tolist() + y_.tolist()))
        bound = overlap.assign_parameters(param_dict)
        return np.abs(bound.eval()) ** 2

    return fidelity


def stochastic_approximation(
    circuit: QuantumCircuit,
    values: np.ndarray,
    samples: int = 100,
    perturbation: float = 0.01,
) -> np.ndarray:
    """Convenience function to call stochastic approximation."""
    sa = StochasticApproximation(samples, perturbation)
    return sa.compute(circuit, values)
