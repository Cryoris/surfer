import numpy as np

from qiskit.circuit import QuantumCircuit, ParameterExpression
from qiskit.circuit.library import RYGate, RZGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import Clifford
from qiskit.transpiler import TransformationPass
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
    _append_cx,
    _append_h,
    _append_x,
    _append_y,
    _append_z,
)


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
            else:
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


def dag_to_clifford(dag, value_dict={}):
    clifford = Clifford(np.identity(2 * dag.num_qubits()), validate=False)
    qubit_indices = {bit: idx for idx, bit in enumerate(dag.qubits)}
    has_action = {idx: False for idx in range(dag.num_qubits())}
    for node in dag.topological_op_nodes():
        if node.op.name == "cx":
            control, target = [qubit_indices[qubit] for qubit in node.qargs]
            if has_action[control]:
                clifford = _append_cx(clifford, control, target)
                has_action[target] = True
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
                has_action[index] = True
            elif np.isclose(angle, -np.pi / 2):
                clifford = _append_x(clifford, index)
                clifford = _append_h(clifford, index)
                has_action[index] = True
            elif np.isclose(angle, np.pi):
                clifford = _append_z(clifford, index)
                clifford = _append_x(clifford, index)
                has_action[index] = True
            elif np.isclose(angle, -np.pi):
                clifford = _append_x(clifford, index)
                clifford = _append_z(clifford, index)
                has_action[index] = True
            else:
                raise ValueError(f"Invalid angle for RY: {angle}")
        elif node.op.name == "rz":
            angle = node.op.params[0]
            if isinstance(angle, ParameterExpression):
                angle = value_dict[angle]
            else:
                angle = float(node.op.params[0])

            if np.isclose(angle, 0):
                pass
            else:
                raise ValueError(f"Invalid angle for RZ: {angle}")

            # not setting an action, as this does not active CX gates

        elif node.op.name == "x":
            clifford = _append_x(clifford, index)
            has_action[index] = True
        elif node.op.name == "y":
            clifford = _append_y(clifford, index)
            has_action[index] = True
        elif node.op.name == "z":
            if has_action[index]:
                clifford = _append_z(clifford, index)
        elif node.op.name == "h":
            clifford = _append_h(clifford, index)
            has_action[index] = True
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
