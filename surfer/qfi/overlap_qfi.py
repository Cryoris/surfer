"""QFI via the overlap method."""

from typing import List, Optional

import numpy as np

from qiskit.circuit import ParameterExpression, QuantumCircuit, Parameter
from qiskit.circuit.library import RYGate, RZGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import AnalysisPass, TransformationPass
from qiskit.quantum_info import StabilizerState, Statevector, Clifford
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
    _append_cx,
    _append_h,
    _append_x,
    _append_y,
    _append_z,
)

from .qfi import QFICalculator
from ..tools.unroll_parameterized_gates import UnrollParameterizedGates
from ..tools.gradient_lookup import gradient_lookup


class OverlapQFI(QFICalculator):
    def __init__(self, clifford=False):
        """
        Args:
            clifford: If True, use Clifford simulation for the circuits. Otherwise
                statevector simulation is used.
        """
        super().__init__()
        self.clifford = clifford
        self.supported_gates = ["rx", "ry", "rz"]

    def compute(
        self,
        circuit: QuantumCircuit,
        values: np.ndarray,
        parameters: Optional[List[Parameter]] = None,
    ) -> np.ndarray:
        # compute QFI with respect to all parameters if none are set
        if parameters is None:
            parameters = circuit.parameters

        unrolled = UnrollParameterizedGates(self.supported_gates)(circuit)

        # convert circuit to Clifford for the phasefix
        if self.clifford:
            data = Clifford(Cliffordize()(unrolled.bind_parameters(values)))
        else:
            data = unrolled.bind_parameters(values)

        # get the derivative data
        values_dict = dict(zip(circuit.parameters, values))
        get_derivatives = Derivatives(values_dict, self.clifford)
        get_derivatives(unrolled)
        derivatives = get_derivatives.property_set["derivatives"]

        qfi = np.zeros((len(parameters), len(parameters)), dtype=complex)

        for i, p_i in enumerate(parameters):
            # TODO maybe set to 1 directly if possible?
            qfi[i, i] += self.compute_curvature(derivatives, p_i, p_i)
            qfi[i, i] += self.compute_phasefix(derivatives, p_i, p_i, data)
            for j_, p_j in enumerate(parameters[i + 1 :]):
                j = i + 1 + j_
                qfi[i, j] += self.compute_curvature(derivatives, p_i, p_j)
                qfi[i, j] += self.compute_phasefix(derivatives, p_i, p_j, data)

        qfi += np.triu(qfi, k=1).T

        return 4 * np.round(np.real(qfi), decimals=10)

    def compute_curvature(self, derivatives, p_i, p_j):
        coeff_i, data_i = derivatives[p_i][0]
        coeff_j, data_j = derivatives[p_j][0]
        data = data_j.compose(inverse(data_i))
        # circuit = circuit_j.compose(circuit_i.inverse())
        # bound = circuit.bind_parameters(values)
        return np.conj(coeff_i) * coeff_j * self.execute(data)

    def compute_phasefix(self, derivatives, p_i, p_j, data):
        coeff_i, data_i = derivatives[p_i][0]
        coeff_j, data_j = derivatives[p_j][0]

        left = data.compose(inverse(data_i))
        right = data_j.compose(inverse(data))

        return np.conj(coeff_i) * coeff_j * self.execute(left) * self.execute(right)

    def execute(self, data):
        if isinstance(data, Clifford):
            state = StabilizerState(data)
        else:
            state = Statevector(data)

        return state.probabilities_dict().get("0" * data.num_qubits, 0)


class Derivatives(AnalysisPass):
    def __init__(
        self, values_dict, clifford, parameters: Optional[List[Parameter]] = None
    ):
        """
        Args:
            parameters: With respect to which parameters to derive. None means all.
        """
        super().__init__()
        self.parameters = parameters
        self.values_dict = values_dict
        self.clifford = clifford
        self.supported_gates = ["rx", "ry", "rz"]

    def run(self, dag: DAGCircuit) -> None:
        # unroll to supported ops

        unrolled = UnrollParameterizedGates(self.supported_gates).run(dag)
        # cliffordizer = Cliffordize()

        derivatives = {}

        for node in unrolled.op_nodes():
            parameters = self.get_parameters(node)

            if len(parameters) > 0:
                parameter = parameters[0]  # RX/Y/Z have only 1 parameter
                gate = node.op
                coeff, derivation = gradient_lookup(gate)[
                    0
                ]  # for RX/Y/Z derivatives are no sums

                derivation_dag = circuit_to_dag(derivation)

                derived = copy_dag(dag)
                derived.substitute_node_with_dag(node, derivation_dag)
                if self.clifford:
                    ref = dag_to_clifford(derived, self.values_dict)
                else:
                    ref = dag_to_circuit(derived).bind_parameters(self.values_dict)

                if parameter in derivatives.keys():
                    derivatives[parameter].append((coeff, ref))
                else:
                    derivatives[parameter] = [(coeff, ref)]

        self.property_set["derivatives"] = derivatives

    @staticmethod
    def get_parameters(node):
        """Check if ``node`` is parameterized. Throws error if not plain parameter.

        Args:
            node: The node to check.

        Returns:
            The parameters in the node's operation.

        Raises:
            RuntimeError: If the node is parameterized with a non-plain parameter
            (like ``-2 * x`` instead of just ``x``).
        """
        parameters = []
        for param in node.op.params:
            if isinstance(param, ParameterExpression):
                if not isinstance(param, Parameter):
                    raise RuntimeError(
                        f"Only plain parameters are supported, not {param}."
                    )
                parameters.append(param)

        return parameters


class Cliffordize(TransformationPass):
    def __init__(self, copy=False):
        super().__init__()
        self.copy = copy

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        if self.copy:
            return self._run_copy(dag)
        return self._run_inplace(dag)

    def _run_copy(self, dag):
        new_dag = empty_like(dag)
        for node in dag.op_nodes():
            if node.op.num_qubits == 1:
                if not is_identity(node.op):
                    new_node = new_dag.apply_operation_back(
                        node.op, node.qargs, node.cargs
                    )
                    try_replace_ry(new_dag, new_node)
            else:
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)

        return new_dag

    def _run_inplace(self, dag):
        for node in dag.op_nodes():
            if node.op.num_qubits > 1:
                continue

            if is_identity(node.op):
                dag.remove_op_node(node)

            try_replace_ry(dag, node)

        return dag


def is_identity(op):
    if isinstance(op, (RYGate, RZGate)):
        return np.isclose(float(op.params[0]), 0)

    return np.allclose(op.to_matrix(), np.identity(2))


def try_replace_ry(dag, node):
    if isinstance(node.op, RYGate):
        angle = float(node.op.params[0])
        # RY(pi/2) = H X
        if np.isclose(angle, np.pi / 2):
            replacement = QuantumCircuit(1)
            replacement.h(0)
            replacement.x(0)
            dag.substitute_node_with_dag(node, circuit_to_dag(replacement))
        elif np.isclose(angle, -np.pi / 2):
            replacement = QuantumCircuit(1)
            replacement.x(0)
            replacement.h(0)
            dag.substitute_node_with_dag(node, circuit_to_dag(replacement))


def empty_like(dag):
    new_dag = DAGCircuit()
    new_dag.name = dag.name
    new_dag.metadata = dag.metadata

    new_dag.add_qubits(dag.qubits)
    new_dag.add_clbits(dag.clbits)

    for qreg in dag.qregs.values():
        new_dag.add_qreg(qreg)
    for creg in dag.cregs.values():
        new_dag.add_creg(creg)

    return new_dag


def copy_dag(dag):
    new_dag = empty_like(dag)
    for node in dag.op_nodes():
        new_dag.apply_operation_back(node.op, node.qargs, node.cargs)
    return new_dag


def dag_to_clifford(dag, value_dict={}):
    clifford = Clifford(np.identity(2 * dag.num_qubits()), validate=False)
    qubit_indices = {bit: idx for idx, bit in enumerate(dag.qubits)}
    for node in dag.topological_op_nodes():
        if node.op.name == "cx":
            control, target = [qubit_indices[qubit] for qubit in node.qargs]
            clifford = _append_cx(clifford, control, target)
            continue
        # if is_identity(node.op, value):
        # continue
        index = qubit_indices[node.qargs[0]]
        if node.op.name == "ry":
            angle = node.op.params[0]
            if isinstance(angle, ParameterExpression):
                angle = value_dict[angle]
            else:
                angle = float(node.op.params[0])

            if np.isclose(angle, 0):
                pass
            elif np.isclose(angle, np.pi / 2):
                clifford = _append_h(clifford, index)
                clifford = _append_x(clifford, index)
            elif np.isclose(angle, -np.pi / 2):
                clifford = _append_x(clifford, index)
                clifford = _append_h(clifford, index)
            else:
                raise NotImplementedError("Invalid angle for RY.")
        elif node.op.name == "rz":
            angle = node.op.params[0]
            if isinstance(angle, ParameterExpression):
                angle = value_dict[angle]
            else:
                angle = float(node.op.params[0])

            if np.isclose(angle, 0):
                pass
            else:
                raise NotImplementedError("Invalid angle for RZ.")
        elif node.op.name == "y":
            clifford = _append_y(clifford, index)
        elif node.op.name == "z":
            clifford = _append_z(clifford, index)
        else:
            raise NotImplementedError(f"Cannot apply {node.op}.")

    return clifford


def inverse(data):
    if isinstance(data, Clifford):
        return _invert_clifford(data)
    if isinstance(data, DAGCircuit):
        return _invert_dag(data)
    return _invert_circuit(data)


def _invert_dag(dag):
    new_dag = empty_like(dag)
    for node in dag.op_nodes():
        new_dag.apply_operation_front(node.op.inverse(), node.qargs, node.cargs)
    return new_dag


def _invert_circuit(circuit):
    return circuit.inverse()


def _invert_clifford(clifford):
    return clifford.adjoint()
