from typing import List
from qiskit.circuit import Instruction, ParameterExpression
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler import TransformationPass


class UnrollParameterizedGates(TransformationPass):
    """Unroll parameterized gates until a supported basis set is reached.

    As this pass is called upon the gradient calculation it only unrolls in favor of performance,
    instead of attempting a basis translation.
    """

    def __init__(self, supported_gates: List[str]) -> None:
        """
        Args:
            supported_gates: A list of suppported basis gates specified as string.
        """
        super().__init__()
        self.supported_gates = supported_gates

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the transpiler pass.

        Args:
            dag: The DAG circuit in which the parameterized gates should be unrolled.

        Returns:
            A DAG where the parameterized gates have been unrolled.

        Raises:
            ValueError: If the circuit cannot be unrolled.
        """
        for node in dag.op_nodes():
            # check whether it is parameterized and we need to decompose it
            if _is_parameterized(node.op) and (
                node.op.name not in self.supported_gates
            ):
                # replace the node with it's decomposition
                definition = node.op.definition

                # recurse to unroll further parameterized blocks
                unrolled = self.run(circuit_to_dag(definition))

                # replace with fully unrolled dag
                dag.substitute_node_with_dag(node, unrolled)

        return dag


def _is_parameterized(op: Instruction) -> bool:
    return any(
        isinstance(param, ParameterExpression) and len(param.parameters) > 0
        for param in op.params
    )
