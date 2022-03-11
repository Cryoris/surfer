"""QFI via the overlap method."""

import copy
from typing import List, Optional

import numpy as np

from qiskit.circuit import ParameterExpression, QuantumCircuit, Parameter
from qiskit.circuit.library import RYGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.exceptions import QiskitError
from qiskit.transpiler import AnalysisPass, TransformationPass
from qiskit.quantum_info import StabilizerState, Statevector, Clifford
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
    _append_cx,
    _append_h,
    _append_x,
    _append_y,
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
        if parameters is None:
            parameters = circuit.parameters

        unrolled = UnrollParameterizedGates(self.supported_gates)(circuit)

        values_dict = dict(zip(circuit.parameters, values))
        get_derivatives = Derivatives(values_dict)
        get_derivatives(unrolled)

        derivatives = get_derivatives.property_set["derivatives"]

        qfi = np.zeros((len(parameters), len(parameters)), dtype=complex)

        for i, p_i in enumerate(parameters):
            # maybe set to 1 directly?
            qfi[i, i] += self.compute_curvature(derivatives, p_i, p_i, values)
            # qfi[i, i] += self.compute_phasefix(derivatives, p_i, p_i, unrolled, values)
            for j_, p_j in enumerate(parameters[i + 1 :]):
                j = i + 1 + j_
                qfi[i, j] += self.compute_curvature(derivatives, p_i, p_j, values)
                # qfi[i, j] += self.compute_phasefix(
                #     derivatives, p_i, p_j, unrolled, values
                # )

        qfi += np.triu(qfi, k=1).T

        return 4 * np.round(np.real(qfi), decimals=10)

    def compute_curvature(self, derivatives, p_i, p_j, values_dict):
        coeff_i, circuit_i = derivatives[p_i][0]
        coeff_j, circuit_j = derivatives[p_j][0]
        circuit = circuit_j.compose(circuit_i.inverse())
        bound = circuit.bind_parameters(values)
        return np.conj(coeff_i) * coeff_j * self.execute_circuit(bound)

    def compute_phasefix(self, derivatives, p_i, p_j, circuit, values):
        coeff_i, circuit_i = derivatives[p_i][0]
        coeff_j, circuit_j = derivatives[p_j][0]

        left = circuit.compose(circuit_i.inverse())
        right = circuit_j.compose(circuit.inverse())

        return (
            np.conj(coeff_i)
            * coeff_j
            * self.execute_circuit(left.bind_parameters(values))
            * self.execute_circuit(right.bind_parameters(values))
        )

    def execute_circuit(self, circuit):
        if self.clifford:

            # cliffordized = Cliffordize(copy=True)(circuit)
            cliffordized = dag_to_clifford(circuit)
            try:
                state = StabilizerState(cliffordized)
                # state = StabilizerState(circuit)
            except QiskitError as exc:
                raise ValueError("Cannot convert circuit to clifford") from exc
        else:
            state = Statevector(circuit)

        return state.probabilities_dict().get("0" * circuit.num_qubits, 0)


class Derivatives(AnalysisPass):
    def __init__(self, values_dict, parameters: Optional[List[Parameter]] = None):
        """
        Args:
            parameters: With respect to which parameters to derive. None means all.
        """
        super().__init__()
        self.parameters = parameters
        self.values_dict = values_dict
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
                # derived.substitute_node(node, derivation)
                # derived = dag_to_circuit(derived)

                if parameter in derivatives.keys():
                    derivatives[parameter].append((coeff, derived))
                else:
                    derivatives[parameter] = [(coeff, derived)]

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
    if isinstance(op, RYGate):
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
    for node in dag.op_nodes():
        if node.op.name == "cx":
            _append_cx(clifford, node.qargs[0].index, node.qargs[1].index)
            continue
        if is_identity(node.op):
            continue
        if node.op.name == "ry":
            angle = node.op.params[0]
            if isinstance(angle, ParameterExpression):
                angle = value_dict[angle]
            else:
                angle = float(node.op.params[0])

            if np.isclose(angle, np.pi / 2):
                _append_h(clifford, node.qargs[0].index)
                _append_x(clifford, node.qargs[0].index)
            elif np.isclose(angle, -np.pi / 2):
                _append_x(clifford, node.qargs[0].index)
                _append_h(clifford, node.qargs[0].index)
            else:
                raise NotImplementedError("Invalid angle for RY.")
        elif node.op.name == "y":
            _append_y(clifford, node.qargs[0].index)

    return clifford
