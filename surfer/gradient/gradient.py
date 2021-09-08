"""The Gradient interface.

Avoids using the plain name Gradient since that exists in Qiskit and I want to avoid conflicts.
"""

from abc import ABC, abstractmethod
import qiskit
import qiskit.opflow
import numpy as np


class GradientCalculator(ABC):
    """The Gradient interface."""

    def __init__(self, do_checks: bool = True):
        """
        Args:
            do_checks: Do some sanity checks on the inputs. Can be disabled for performance.
        """
        self.do_checks = do_checks

    @abstractmethod
    def compute(
        self,
        operator: qiskit.opflow.OperatorBase,
        circuit: qiskit.QuantumCircuit,
        values: np.ndarray,
    ) -> np.ndarray:
        """Compute the Gradient for the given circuit.

        The initial state is assumed to be the all-zero state.

        Args:
            operator: The operator for the expectation value.
            circuit: A parameterized unitary circuit preparing the quantum state of which we compute
                the Gradient.
            values: The parameter values.
        """
        raise NotImplementedError

    @staticmethod
    def check_inputs(circuit: qiskit.QuantumCircuit, values: np.ndarray) -> None:
        """Check the circuit and values.

        Args:
            circuit: A parameterized unitary circuit preparing the quantum state of which we compute
                the Gradient.
            values: The parameter values.

        Raises:
            ValueError: If the circuit is invalid (non unitary or gates with more than 1 parameter).
            ValueError: If the number of values doesn't match the parameters.
        """
        _check_circuit_is_unitay(circuit)
        _check_1_parameter_per_gate(circuit)

        # check the number of parameters
        if circuit.num_parameters != values.size:
            raise ValueError(
                f"Mismatching number of parameters ({circuit.num_parameters}) "
                f"and values ({values.size})."
            )


def _check_circuit_is_unitay(circuit):
    try:
        _ = circuit.to_gate()
    except qiskit.circuit.exceptions.CircuitError:
        # pylint: disable=raise-missing-from
        raise ValueError("The circuit is not unitary.")


def _check_1_parameter_per_gate(circuit):
    for inst, _, _ in circuit.data:
        params = inst.params
        if (
            any(isinstance(params, qiskit.circuit.ParameterExpression))
            and len(params) > 1
        ):
            raise ValueError(
                "If a gate is parameterized, it can only have 1 parameter."
            )
