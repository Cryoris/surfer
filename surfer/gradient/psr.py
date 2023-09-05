import numpy as np
from qiskit.quantum_info import StabilizerState, Statevector
from qiskit.converters import circuit_to_dag
from .gradient import GradientCalculator
from ..tools.clifford import Cliffordize, dag_to_clifford
from ..tools.unroll_parameterized_gates import UnrollParameterizedGates


class PSR(GradientCalculator):
    """Parameter shift gradient, supporting Clifford optimization."""

    def __init__(self, clifford=False):
        """
        Args:
            clifford: If ``True``, try to cast the gradient circuits to Cliffords.
        """
        super().__init__(do_checks=False)
        self.clifford = clifford

    def compute(self, operator, circuit, values):
        d = circuit.num_parameters
        shifts = (np.pi / 2 * np.eye(d)).tolist()

        if self.clifford:
            unroller = UnrollParameterizedGates(["ry", "rz"])
            circuit = unroller(circuit)

        values = np.asarray(values)

        plus_circuits = [
            circuit.bind_parameters(values + np.asarray(shift)) for shift in shifts
        ]
        minus_circuits = [
            circuit.bind_parameters(values - np.asarray(shift)) for shift in shifts
        ]

        if self.clifford:
            cliffordizer = Cliffordize()

            def to_stab(circuit):
                dag = circuit_to_dag(circuit, copy_operations=False)
                clifford = dag_to_clifford(cliffordizer.run(dag))
                return StabilizerState(clifford)

            plus_states = list(map(to_stab, plus_circuits))
            minus_states = list(map(to_stab, minus_circuits))

            plus_values = np.array(
                [_ss_expectation(state, operator) for state in plus_states]
            )
            minus_values = np.array(
                [_ss_expectation(state, operator) for state in minus_states]
            )
        else:
            plus_states = [Statevector(circuit) for circuit in plus_circuits]
            minus_states = [Statevector(circuit) for circuit in minus_circuits]

            plus_values = np.array(
                [state.expectation_value(operator) for state in plus_states]
            )
            minus_values = np.array(
                [state.expectation_value(operator) for state in minus_states]
            )

        gradients = (plus_values - minus_values) / 2
        return gradients


def _ss_expectation(state, hamiltonian):
    values = [state.expectation_value(pauli) for pauli in hamiltonian.paulis]
    return np.dot(hamiltonian.coeffs, values)
