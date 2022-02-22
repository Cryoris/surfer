import unittest
import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.opflow import X, Z, Y
from surfer.qfi import ReverseQFI


class TestReverseQFI(unittest.TestCase):
    """Tests for the reverse gradient calculation."""

    def test_simple(self):
        """Test a simple 1-qubit circuit."""
        x = ParameterVector("x", 2)
        circuit = QuantumCircuit(1)
        circuit.rx(x[0], 0)
        circuit.ry(x[1], 0)

        values = np.array([0.2, 0.9])

        expect = np.array([[1.0, 0.0], [0.0, 0.9605305]])

        qfi = ReverseQFI().compute(circuit, values)
        self.assertTrue(np.allclose(qfi, expect))

    # def test_vector_correctly_sorted(self):
    #     """Test the gradient vector is correctly sorted."""
    #     x = ParameterVector("x", 4)
    #     circuit = QuantumCircuit(2)
    #     circuit.h([0, 1])
    #     circuit.ry(x[3], 0)
    #     circuit.ry(x[1], 1)
    #     circuit.cx(0, 1)
    #     circuit.ry(x[0], 0)
    #     circuit.ry(x[2], 1)

    #     values = [-np.pi / 2, np.pi, np.pi / 4, np.pi / 2]

    #     expected = [-1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]

    #     observable = Z ^ Z

    #     grad = ReverseGradient().compute(observable, circuit, values)

    #     self.assertTrue(np.allclose(grad, expected))

    # def test_shared_parameters(self):
    #     """Test the product rule is applied correctly."""
    #     x = ParameterVector("x", 2)
    #     circuit = QuantumCircuit(2)
    #     circuit.h([0, 1])
    #     circuit.rx(x[0], 0)
    #     circuit.ry(x[1], 1)
    #     circuit.cz(0, 1)
    #     circuit.ry(x[0], 0)
    #     circuit.cry(x[1], 1, 0)

    #     values = [-np.pi / 2, np.pi, np.pi / 2]

    #     expected = [-1, -0.5]

    #     observable = X ^ X

    #     grad = ReverseGradient().compute(observable, circuit, values)

    #     # from qiskit.opflow import Gradient, StateFn

    #     # exp = ~StateFn(observable) @ StateFn(circuit)
    #     # print(
    #     #     Gradient()
    #     #     .convert(exp)
    #     #     .bind_parameters(dict(zip(circuit.parameters, values)))
    #     #     .eval()
    #     # )

    #     self.assertTrue(np.allclose(grad, expected))

    def test_decompose(self):
        """Test the gradient can decompose into a supported basis gate set."""
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

        qfi = ReverseQFI().compute(circuit, values)
        self.assertTrue(np.allclose(qfi, expect))

    def test_partially_bound_circuit(self):
        """Test a gradient calculation with a circuit that also has bound parameters."""
        x = ParameterVector("x", 2)
        circuit = QuantumCircuit(1)
        circuit.rz(0, 0)
        circuit.rx(x[0], 0)
        circuit.ry(x[1], 0)

        values = np.array([0.2, 0.9])
        expect = np.array([[1.0, 0.0], [0.0, 0.9605305]])

        qfi = ReverseQFI().compute(circuit, values)
        self.assertTrue(np.allclose(qfi, expect))

    def test_single_derivative(self):
        """Test the calculation of a single derivative, not the entire gradient."""
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
