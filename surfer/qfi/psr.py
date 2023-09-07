import numpy as np
from qiskit.quantum_info import StabilizerState, Statevector
from qiskit.converters import circuit_to_dag
from .qfi import QFICalculator
from ..tools.clifford import Cliffordize, dag_to_clifford
from ..tools.unroll_parameterized_gates import UnrollParameterizedGates


class ParameterShiftQFI(QFICalculator):
    """Parameter shift gradient, supporting Clifford optimization."""

    def __init__(self, clifford=False):
        """
        Args:
            clifford: If ``True``, try to cast the gradient circuits to Cliffords.
        """
        super().__init__(do_checks=False)
        self.clifford = clifford

    def compute(self, circuit, values):
        d = circuit.num_parameters
        shifts = np.pi / 2 * np.eye(d)

        if self.clifford:
            unroller = UnrollParameterizedGates(["ry", "rz"])
            circuit = unroller(circuit)

        values = np.asarray(values)

        bound = circuit.bind_parameters(values)

        if self.clifford:
            cliffordizer = Cliffordize()
            dag = circuit_to_dag(bound)
            clifford = dag_to_clifford(cliffordizer.run(dag))
            right = StabilizerState(clifford)
        else:
            right = Statevector(bound.inverse())

        directions = {"++": (1, 1), "+-": (1, -1), "-+": (-1, 1), "--": (-1, -1)}

        # TODO optimize and avoid +- evaluations if i==j, where the fidelity is 1
        circuits = {
            label: [
                circuit.bind_parameters(values + c_0 * shifts[i] + c_1 * shifts[j])
                for i in range(d)
                for j in range(d)
            ]
            for label, (c_0, c_1) in directions.items()
        }

        if self.clifford:

            def stab_fidelity(circuit):
                # indicator that shifts cancelled
                if circuit is None:
                    return 1
                dag = circuit_to_dag(circuit, copy_operations=False)
                clifford = dag_to_clifford(cliffordizer.run(dag))
                return _fidelity(right.evolve(clifford.adjoint()))

            values = {
                label: list(map(stab_fidelity, circuits_))
                for label, circuits_ in circuits.items()
            }
        else:
            values = {
                label: [_fidelity(Statevector(circuit)) for circuit in circuits_]
                for label, circuits_ in circuits.items()
            }

        tensor = np.empty((d, d))
        for i in range(d):
            for j in range(i, d):
                ind = i + d * j
                tensor[i, j] = (
                    values["++"][ind]
                    - values["+-"][ind]
                    - values["-+"][ind]
                    + values["--"][ind]
                )

                if i != j:
                    tensor[j, i] = tensor[i, j]

        # qfi = 4 qgt = 4 (-0.5 * tensor / 4) = -0.5 tensor  -- factor 1/4 from PSR
        qfi = -0.5 * tensor
        return qfi


def _fidelity(state):
    return state.probabilities_dict().get("0" * state.num_qubits, 0)
