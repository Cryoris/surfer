import unittest
import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import EfficientSU2, RealAmplitudes
from surfer.qfi.overlap_qfi import Overlap


class TestOverlapQFI(unittest.TestCase):
    """Tests for the reverse gradient calculation."""

    def test_simple(self):
        """Test a simple 1-qubit circuit."""

        circuit = RealAmplitudes(2, reps=1)
        qfi = Overlap().compute(circuit, np.zeros(circuit.num_parameters), None)
        print(qfi)


if __name__ == "__main__":
    unittest.main()
