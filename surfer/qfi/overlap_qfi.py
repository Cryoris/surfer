import numpy as np
import copy
from typing import List, Optional
from qiskit.circuit import ParameterExpression, QuantumCircuit, Parameter
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import AnalysisPass
from qiskit.quantum_info import StabilizerState, Statevector

from .qfi import QFICalculator
from ..tools.unroll_parameterized_gates import UnrollParameterizedGates
from ..tools.gradient_lookup import gradient_lookup


class Overlap(QFICalculator):
    def __init__(self, clifford=False):
        """
        Args:
            clifford: If True, use Clifford simulation for the circuits. Otherwise
                statevector simulation is used.
        """
        super().__init__()
        self.clifford = clifford

    def compute(
        self,
        circuit: QuantumCircuit,
        values: np.ndarray,
        parameters: Optional[List[Parameter]] = None,
    ) -> np.ndarray:
        if parameters is None:
            parameters = circuit.parameters

        get_derivatives = Derivatives()
        get_derivatives(circuit)

        derivatives = get_derivatives.property_set["derivatives"]

        qfi = np.zeros((len(parameters), len(parameters)), dtype=complex)

        for i, p_i in enumerate(parameters):
            for j, p_j in enumerate(parameters):
                qfi[i, j] += self.compute_curvature(derivatives, p_i, p_j, values)
                qfi[i, j] += self.compute_phasefix(
                    derivatives, p_i, p_j, circuit, values
                )

        return 4 * np.round(np.real(qfi), decimals=10)

        for key, derivative in derivatives.property_set["derivatives"].items():
            print(key)
            for coeff, dag in derivative:
                print(coeff)
                print(dag)
            print()

    def compute_curvature(self, derivatives, p_i, p_j, values):
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
            state = StabilizerState(circuit)
        else:
            state = Statevector(circuit)

        print(circuit)
        print(state)
        return state.probabilities_dict().get("0" * circuit.num_qubits, 0)


class Derivatives(AnalysisPass):
    def __init__(self, parameters: Optional[List[Parameter]] = None):
        """
        Args:
            parameters: With respect to which parameters to derive. None means all.
        """
        super().__init__()
        self.parameters = parameters
        self.supported_gates = ["rx", "ry", "rz"]

    def run(self, dag: DAGCircuit) -> None:
        # unroll to supported ops

        unrolled = UnrollParameterizedGates(self.supported_gates).run(dag)

        derivatives = {}

        for node in unrolled.op_nodes():
            parameters = self.get_parameters(node)

            if len(parameters) > 0:
                parameter = parameters[0]  # RX/Y/Z have only 1 parameter
                gate = node.op
                coeff, derivation = gradient_lookup(gate)[
                    0
                ]  # for RX/Y/Z derivatives are no sums

                derived = copy.deepcopy(dag)
                derived.substitute_node_with_dag(node, circuit_to_dag(derivation))
                derived = dag_to_circuit(derived)

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
