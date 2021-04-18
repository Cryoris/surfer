"""The QFI interface.

Avoids using the plain name QFI since that exists in Qiskit and I want to avoid conflicts.
"""

from abc import ABC, abstractmethod
import qiskit
import numpy as np


class QFICalculator(ABC):
    """The QFI interface."""

    def __init__(self, do_checks: bool = True):
        """
        Args:
            do_checks: Do some sanity checks on the inputs. Can be disabled for performance.
        """
        self.do_checks = do_checks

    @abstractmethod
    def compute(self, circuit: qiskit.QuantumCircuit, values: np.ndarray) -> np.ndarray:
        """Compute the QFI for the given circuit.

        The initial state is assumed to be the all-zero state.

        Args:
            circuit: A parameterized unitary circuit preparing the quantum state of which we compute
                the QFI.
            values: The parameter values.
        """
        raise NotImplementedError

    @staticmethod
    def check_inputs(circuit: qiskit.QuantumCircuit, values: np.ndarray) -> None:
        """Check the circuit and values.

        Args:
            circuit: A parameterized unitary circuit preparing the quantum state of which we compute
                the QFI.
            values: The parameter values.

        Raises:
            ValueError: If the circuit is invalid (non unitary or gates with more than 1 parameter).
            ValueError: If the number of values doesn't match the parameters.
            NotImplementedError: If the circuit has repeated parameters.
        """
        _check_circuit_is_unitay(circuit)
        _check_1_parameter_per_gate(circuit)

        # check the number of parameters
        if circuit.num_parameters != values.size:
            raise ValueError(
                f"Mismatching number of parameters ({circuit.num_parameters}) "
                f"and values ({values.size})."
            )

        _check_no_duplicate_params(circuit)


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
            any(
                isinstance(param, qiskit.circuit.ParameterExpression)
                for param in params
            )
            and len(params) > 1
        ):
            raise ValueError(
                "If a gate is parameterized, it can only have 1 parameter."
            )


def _check_no_duplicate_params(circuit):
    # pylint: disable=protected-access
    for _, gates in circuit._parameter_table.items():
        if len(gates) > 1:
            raise NotImplementedError(
                "The product rule is currently not implemented, parameters must be unique."
            )
