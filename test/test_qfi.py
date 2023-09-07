import unittest
from ddt import ddt, data
import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import EfficientSU2, RealAmplitudes
from surfer.qfi import ReverseQFI, OverlapQFI, ParameterShiftQFI, LinearCombination

_ALL = [ReverseQFI, OverlapQFI, ParameterShiftQFI]


@ddt
class TestQFI(unittest.TestCase):
    """Tests for the reverse gradient calculation."""

    @data(*_ALL)
    def test_simple(self, qfi_cls):
        """Test a simple 1-qubit circuit."""
        x = ParameterVector("x", 2)
        circuit = QuantumCircuit(1)
        circuit.rx(x[0], 0)
        circuit.ry(0.5 * x[1], 0)

        values = [0.2, 0.9]

        expect = np.array([[1.0, 0.0], [0.0, 0.9605305 / 4]])

        qfi = qfi_cls().compute(circuit, values)
        self.assertTrue(np.allclose(qfi, expect))

    @data(*_ALL)
    def test_phasefix(self, qfi_cls):
        """Test with a non-trivial phasefix."""
        x = ParameterVector("x", 2)
        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit.rz(x[0], 0)
        circuit.rx(x[1], 0)

        values = np.array([np.pi / 4, 0.1])

        qfi = qfi_cls().compute(circuit, values)
        # using Qiskit as reference calculation
        ref = LinearCombination().compute(circuit, values)

        self.assertTrue(np.allclose(qfi, ref))

    @data(*_ALL)
    def test_large(self, qfi_cls):
        """Test a larger system. This might take a bit."""
        circuit = EfficientSU2(num_qubits=4, reps=3, entanglement="full")
        new_params = ParameterVector("x", 8)
        flattened = 4 * list(new_params[:])
        rebound = circuit.assign_parameters(flattened)

        np.random.seed(2)
        values = np.random.random(rebound.num_parameters)

        qfi = qfi_cls().compute(rebound, values)
        # using Qiskit as reference calculation
        ref = LinearCombination().compute(rebound, values)

        self.assertTrue(np.allclose(qfi, ref))

    @data(*_ALL)
    def test_correctly_sorted(self, qfi_cls):
        """Test the QFI is correctly sorted."""
        x = ParameterVector("x", 2)
        circuit = QuantumCircuit(1)
        circuit.rx(x[1], 0)
        circuit.ry(x[0], 0)

        values = np.array([0.2, 0.2])

        expect = np.array([[0.9605305, 0], [0.0, 1]])

        qfi = qfi_cls().compute(circuit, values)
        self.assertTrue(np.allclose(qfi, expect))

    @data(*_ALL)
    def test_shared_parameters(self, qfi_cls):
        """Test the product rule is applied correctly."""
        x = ParameterVector("x", 2)
        circuit = QuantumCircuit(2)
        circuit.h([0, 1])
        circuit.rx(x[0], 0)
        circuit.ry(x[1], 1)
        circuit.cz(0, 1)
        circuit.ry(x[0], 0)
        circuit.cry(x[1], 1, 0)

        values = np.array([0.1, 0.2])

        qfi = qfi_cls().compute(circuit, values)
        # using Qiskit as reference calculation
        ref = LinearCombination().compute(circuit, values)

        self.assertTrue(np.allclose(qfi, ref))

    @data(*_ALL)
    def test_decompose(self, qfi_cls):
        """Test the QFI can decompose into a supported basis gate set."""
        x = ParameterVector("x", 2)
        inner = QuantumCircuit(1)
        inner.h(0)
        inner.rx(x[0], 0)
        inner.ry(x[1], 0)
        inner.sxdg(0)

        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit.append(inner.to_gate(), [0])

        values = np.array([0.2, 0.9])
        expect = np.array([[1.0, 0.0], [0.0, 0.9605305]])

        qfi = qfi_cls().compute(circuit, values)
        self.assertTrue(np.allclose(qfi, expect))

    @data(*_ALL)
    def test_partially_bound_circuit(self, qfi_cls):
        """Test a QFI calculation with a circuit that also has bound parameters."""
        x = ParameterVector("x", 2)
        circuit = QuantumCircuit(1)
        circuit.rz(0, 0)
        circuit.rx(x[0], 0)
        circuit.ry(x[1], 0)

        values = np.array([0.2, 0.9])
        expect = np.array([[1.0, 0.0], [0.0, 0.9605305]])

        qfi = qfi_cls().compute(circuit, values)
        self.assertTrue(np.allclose(qfi, expect))

    def test_individual_derivative(self):
        """Test the calculation of a individual derivatives, not the entire QFI."""
        x = ParameterVector("x", 4)
        circuit = QuantumCircuit(2)
        circuit.rx(x[0], 0)
        circuit.rx(x[1], 1)
        circuit.ry(x[2], 0)
        circuit.ry(x[3], 1)

        values = np.array([0.2, 0.3, 0.9, 1.4])
        expect = np.array([[1.0, 0.0], [0.0, 0.9605305]])

        qfi = ReverseQFI().compute(circuit, values, parameters=[x[0], x[2]])
        self.assertTrue(np.allclose(qfi, expect))


@ddt
class TestCliffordQFI(unittest.TestCase):
    """Test cases for circuits that are Clifford."""

    def setUp(self):
        super().setUp()

        qc1 = RealAmplitudes(2, reps=1)
        qc1.h(qc1.qubits)

        self.case1 = (qc1, np.zeros(qc1.num_parameters))

    @data(OverlapQFI, ParameterShiftQFI)
    def test_simple(self, qfi_cls):
        """Test a simple case."""
        circuit, values = self.case1
        qfi = qfi_cls(clifford=True).compute(circuit, values)

        expect = ReverseQFI().compute(circuit, values)
        self.assertTrue(np.allclose(qfi, expect))

    @data(OverlapQFI, ParameterShiftQFI)
    def test_fails(self, qfi_cls):
        circuit = RealAmplitudes(1, reps=0)
        values = [0.1]  # not an angle where RY is Clifford

        with self.assertRaises(ValueError):
            _ = qfi_cls(clifford=True).compute(circuit, values)


if __name__ == "__main__":
    unittest.main()
