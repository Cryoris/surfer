import unittest
import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.opflow import X, Z, Y, I
from surfer.gradient import ReverseGradient


class TestReverseGradient(unittest.TestCase):
    """Tests for the reverse gradient calculation."""

    def test_1q_circuit(self):
        """Test a simple 1-qubit circuit."""
        x = ParameterVector("x", 1)
        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit.ry(x[0], 0)

        values = [0.2]
        exact = -np.sin(values[0])

        observable = X

        grad = ReverseGradient().compute(observable, circuit, values)
        self.assertAlmostEqual(grad[0], exact)

    def test_vector_correctly_sorted(self):
        """Test the gradient vector is correctly sorted."""
        x = ParameterVector("x", 4)
        circuit = QuantumCircuit(2)
        circuit.h([0, 1])
        circuit.ry(x[3], 0)
        circuit.ry(x[1], 1)
        circuit.cx(0, 1)
        circuit.ry(x[0], 0)
        circuit.ry(x[2], 1)

        values = [-np.pi / 2, np.pi, np.pi / 4, np.pi / 2]

        expected = [-1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]

        observable = Z ^ Z

        grad = ReverseGradient().compute(observable, circuit, values)

        self.assertTrue(np.allclose(grad, expected))

    def test_shared_parameters(self):
        """Test the product rule is applied correctly."""
        x = ParameterVector("x", 2)
        circuit = QuantumCircuit(2)
        circuit.h([0, 1])
        circuit.rx(x[0], 0)
        circuit.ry(x[1], 1)
        circuit.cz(0, 1)
        circuit.ry(x[0], 0)
        circuit.cry(x[1], 1, 0)

        values = [-np.pi / 2, np.pi, np.pi / 2]

        expected = [-1, -0.5]

        observable = X ^ X

        grad = ReverseGradient().compute(observable, circuit, values)

        # from qiskit.opflow import Gradient, StateFn

        # exp = ~StateFn(observable) @ StateFn(circuit)
        # print(
        #     Gradient()
        #     .convert(exp)
        #     .bind_parameters(dict(zip(circuit.parameters, values)))
        #     .eval()
        # )

        self.assertTrue(np.allclose(grad, expected))

    def test_decompose(self):
        """Test the gradient can decompose into a supported basis gate set."""
        x = ParameterVector("x", 1)
        inner = QuantumCircuit(1)
        inner.rx(x[0], 0)
        inner.sxdg(0)

        circuit = QuantumCircuit(1)
        circuit.sx(0)
        circuit.append(inner.to_gate(), [0])

        values = [np.pi]

        expected = [1]

        observable = X + Y

        grad = ReverseGradient().compute(observable, circuit, values)
        self.assertTrue(np.allclose(grad, expected))

    def test_partially_bound_circuit(self):
        """Test a gradient calculation with a circuit that also has bound parameters."""
        x = ParameterVector("x", 1)
        circuit = QuantumCircuit(1)
        circuit.ry(np.pi / 2, 0)  # bound parameter
        circuit.ry(x[0], 0)  # free parameter
        values = [1]

        observable = X

        exact = -np.sin(values[0])

        grad = ReverseGradient().compute(observable, circuit, values)

        self.assertEqual(len(grad), 1)
        self.assertAlmostEqual(grad[0], exact)

    def test_single_derivative(self):
        """Test the calculation of a single derivative, not the entire gradient."""
        x = ParameterVector("x", 4)
        circuit = QuantumCircuit(2)
        circuit.h([0, 1])
        circuit.ry(x[3], 0)
        circuit.ry(x[1], 1)
        circuit.cx(0, 1)
        circuit.ry(x[0], 0)
        circuit.ry(x[2], 1)

        values = [-np.pi / 2, np.pi, np.pi / 4, np.pi / 2]

        expected = 1 / np.sqrt(2)

        observable = Z ^ Z

        grad = ReverseGradient().compute(observable, circuit, values, parameters=x[3])
        self.assertEqual(len(grad), 1)
        self.assertAlmostEqual(grad[0], expected)

    def test_partial(self):
        x = ParameterVector("x", 2)
        circuit = QuantumCircuit(2)
        # circuit.rx(x[0], 0)
        # circuit.rx(x[1], 1)
        circuit.rx(x[0], 0)
        circuit.rzz(x[1], 0, 1)

        values = [0, 0]
        observable = (X ^ I) + (I ^ X) + 0.25 * (Z ^ Z)
        expect = -0.5j

        grad = ReverseGradient(partial_gradient=True).compute(
            observable, circuit, values
        )
        print(grad)

        from qiskit.primitives import Estimator
        from qiskit.algorithms.gradients.lin_comb_estimator_gradient import (
            LinCombEstimatorGradient,
            DerivativeType,
        )

        lcu = LinCombEstimatorGradient(Estimator(), DerivativeType.COMPLEX)
        res = lcu.run([circuit], [observable.primitive], [values]).result()
        print(res.gradients[0] / 2)


if __name__ == "__main__":
    unittest.main()
