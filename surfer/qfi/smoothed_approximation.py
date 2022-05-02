"""Stochastic approximation of the Quantum Fisher Approximation."""

import numpy as np

from qiskit.circuit import QuantumCircuit

from .stochastic_approximation import StochasticApproximation

NoneType = type(None)


class SmoothedApproximation:
    """Smoothed stochastic approximation of the Quantum Fisher Approximation."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        values: np.ndarray,
        batch_size: int,
        perturbation: float,
    ):
        """
        Args:
            circuit: The circuit for which to compute the QFI.
            values: The values for which to compute the QFI.
            batch_size: The number of samples per batch.
            perturbation: The perturbation for the finite difference approximation. Values of
                0.1 (lots of noise) to 0.01 (low/no noise) usually work well.
            do_checks: Whether to check the inputs of the ``compute`` method.
        """
        self.circuit = circuit
        self.values = values

        # the object to get a new QFI estimate with the specified batch size
        self.sampler = StochasticApproximation(batch_size, perturbation)
        self.batch_size = batch_size

        # the number of iterations / batch samples taken
        self.num_iterations = 0

        # the current estimate for the current iteration
        self.summed_point_estimate = np.zeros(
            (circuit.num_parameters, circuit.num_parameters)
        )
        self.num_point_batches = 0

        # the smoothed QFI estimate (we start from the identity)
        self.previous_estimate = np.identity(circuit.num_parameters)

    def sample(self):
        """Get a new sample and add it to the smoothed average.

        Args:
            values: The parameter values at which to evaluate the QFI sample.
        """
        sample = self.sampler.compute(self.circuit, self.values)
        self.summed_point_estimate += sample
        self.num_point_batches += 1

    @property
    def point_estimate(self):
        """Get the current point estimate.

        Returns:
            The current point estimate, or a zero matrix if no samples have been taken.
        """
        if self.num_point_batches > 0:
            return self.summed_point_estimate / self.num_point_batches

        n = self.circuit.num_parameters
        return np.zeros((n, n))

    @property
    def smoothed_estimate(self):
        """Get the current smoothed estimate.

        Returns:
            The current smoothed estimate.
        """
        k = self.num_iterations
        return k / (k + 1) * self.previous_estimate + 1 / k * self.point_estimate

    def next_iteration(self, next_values):
        self.values = next_values
        self.point_estimate = None
        self.num_point_batches = 0
        self.previous_estimate = self.smoothed_estimate
        self.num_iterations += 1
