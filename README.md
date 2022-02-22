# Surfer 🏄

Tools to ride the waves of loss landscapes!

This package includes different methods to compute gradients of expecation values and the Quantum Fisher Information (QFI) of states. It is based on Qiskit.

## Gradient methods

Gradient methods are located in `surfer.gradient` and derive the `GradientCalculator` interface. Each method has a custom constructor but offers a
```python
compute(operator: qiskit.opflow.OperatorBase, circuit: qiskit.QuantumCircuit, values: numpy.ndarray)
```
method to compute the gradients.

See ``examples/gradient.py`` for an example.

**Supported methods**
* `ForwardGradient`: Statevector-based, scales quadratically in the number of parameters.
* `ReverseGradient`: Statevector-based, scales linearly in the number of parameters, see also [arXiv:2009.02823](https://arxiv.org/abs/2009.02823).

## QFI methods

QFI methods are located in `surfer.qfi` and derive the `QFICalculator` interface. Each method has a custom constructor but offers a
```python
compute(circuit: qiskit.QuantumCircuit, values: numpy.ndarray)
```
method to compute the QFI of the state that is obtained by applying `circuit` to the all zero state.

See ``examples/qfi.py`` for an example.

**Supported methods**
* `LinearCombination`: Proxy for Qiskit's QFI calculation based on the linear combination of unitaries.
* `ReverseQFI`: Statevector-based, scales quadratically in the number of parameters, see also [arXiv:2011.02991](https://arxiv.org/abs/2011.02991).
* `StochasticApproximation`: Monte Carlo-style stochastic approximation of the QFI, based on [arXiv:2103.09232](https://arxiv.org/abs/2103.09232).