import unittest
from ddt import ddt, data
import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import EfficientSU2, RealAmplitudes
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Clifford

from surfer.qfi import ReverseQFI, OverlapQFI
from surfer.qfi.overlap_qfi import dag_to_clifford, Cliffordize


@ddt
class TestOverlapQFI(unittest.TestCase):
    """Tests for the reverse gradient calculation."""

    def reference(self, circuit, values):
        return ReverseQFI().compute(circuit, values)

    def test_simple(self):
        """Test a simple 1-qubit circuit."""

        circuit = RealAmplitudes(2, reps=1)
        qfi = OverlapQFI().compute(circuit, np.zeros(circuit.num_parameters), None)
        print(qfi)

    @data("0")  # , "+")
    def test_simple_cliff(self, initial_parameters):
        """Test a simple 1-qubit circuit."""

        circuit = RealAmplitudes(50, reps=1)
        if initial_parameters == "0":
            values = np.zeros(circuit.num_parameters)
        else:
            values = np.zeros(circuit.num_parameters)
            for i in range(circuit.num_qubits):
                values[~i] = np.pi / 2

        qfi = OverlapQFI(clifford=True).compute(circuit, values, None)
        # print(qfi)
        # print(self.reference(circuit, values))
        # self.assertTrue(np.allclose(qfi, self.reference(circuit, values)))

    def test_dag_to_clifford(self):
        circuit = RealAmplitudes(2, reps=1).bind_parameters(np.zeros(4))
        dag = circuit_to_dag(circuit.decompose())
        clifford = dag_to_clifford(dag)
        print(clifford)
        print(Clifford(Cliffordize()(circuit.decompose())))


if __name__ == "__main__":
    unittest.main()
