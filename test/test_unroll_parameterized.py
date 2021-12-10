import unittest

from qiskit.circuit import QuantumCircuit, ParameterVector
from surfer.tools.unroll_parameterized_gates import UnrollParameterizedGates


class TestUnrollParameterized(unittest.TestCase):
    """Test the pass to unroll parameterized gates."""

    def setUp(self):
        super().setUp()
        self.supported_gates = ["rx", "ry", "rz", "cp", "crx", "cry", "crz"]

    def test_only_parameterized_is_unrolled(self):
        """Test only parameterized gates are unrolled."""
        x = ParameterVector("x", 4)
        block1 = QuantumCircuit(1)
        block1.rx(x[0], 0)

        sub_block = QuantumCircuit(2)
        sub_block.cx(0, 1)
        sub_block.rz(x[2], 0)

        block2 = QuantumCircuit(2)
        block2.ry(x[1], 0)
        block2.append(sub_block.to_gate(), [0, 1])

        block3 = QuantumCircuit(3)
        block3.ccx(0, 1, 2)

        circuit = QuantumCircuit(3)
        circuit.append(block1.to_gate(), [1])
        circuit.append(block2.to_gate(), [0, 1])
        circuit.append(block3.to_gate(), [0, 1, 2])
        circuit.cry(x[3], 0, 2)

        unroller = UnrollParameterizedGates(self.supported_gates)
        unrolled = unroller(circuit)

        expected = QuantumCircuit(3)
        expected.rx(x[0], 1)
        expected.ry(x[1], 0)
        expected.cx(0, 1)
        expected.rz(x[2], 0)
        expected.append(block3.to_gate(), [0, 1, 2])
        expected.cry(x[3], 0, 2)

        self.assertEqual(unrolled, expected)
