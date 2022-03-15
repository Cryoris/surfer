import unittest
from ddt import ddt, data
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Clifford

from surfer.qfi import ReverseQFI, OverlapQFI
from surfer.qfi.overlap_qfi import dag_to_clifford, Cliffordize


@ddt
class TestOverlapQFI(unittest.TestCase):
    """Tests for the reverse gradient calculation."""

    def reference(self, circuit, values):
        return ReverseQFI(phase_fix=True).compute(circuit, values)

    def test_simple(self):
        """Test a simple 1-qubit circuit."""

        circuit = RealAmplitudes(2, reps=1)
        values = np.zeros(circuit.num_parameters)
        qfi = OverlapQFI().compute(circuit, values, None)
        self.assertTrue(np.allclose(qfi, self.reference(circuit, values)))

    @data("0", "+")  # , "+")
    def test_simple_cliff(self, initial_parameters="0"):
        """Test a simple 1-qubit circuit."""

        circuit = RealAmplitudes(3, reps=2)
        if initial_parameters == "0":
            values = np.zeros(circuit.num_parameters)
        else:
            values = np.zeros(circuit.num_parameters)
            for i in range(circuit.num_qubits):
                values[~i] = np.pi / 2

        qfi = OverlapQFI(clifford=True).compute(circuit, values, None)
        self.assertTrue(np.allclose(qfi, self.reference(circuit, values)))

    def test_dag_to_clifford(self):
        circuit = QuantumCircuit(2)
        circuit.rz(0, 0)
        circuit.ry(0, 1)
        circuit.y(0)
        circuit.z(1)
        circuit.cx(0, 1)
        circuit.rz(0, 0)
        circuit.ry(0, 1)
        dag = circuit_to_dag(circuit)
        clifford = dag_to_clifford(dag)
        ref = Clifford(Cliffordize()(circuit))

        self.assertEqual(clifford, ref)


if __name__ == "__main__":
    unittest.main()
