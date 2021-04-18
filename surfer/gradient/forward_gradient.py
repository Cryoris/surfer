"""A class to compute gradients of expectation values."""

from functools import reduce
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector

from surfer.tools.split_circuit import split
from surfer.tools.gradient_lookup import analytic_gradient
from surfer.tools.bind import bind
from surfer.tools.accumulate_product_rule import accumulate_product_rule


class ForwardGradient:
    def compute(self, operator, ansatz, values):
        """
        Args:
            operator (OperatorBase): The operator in the expectation value.
            ansatz (QuantumCircuit): The ansatz in the expecation value.
            state_in (Statevector): The initial, unparameterized state, upon which the ansatz acts.
            target_parameters (List[Parameter]): The parameters with respect to which to derive.
                If None, the derivative for all parameters is computed (also bound parameters!).
        """
        unitaries, paramlist = split(ansatz, return_parameters=True)
        parameter_binds = dict(zip(ansatz.parameters, values))

        num_parameters = len(unitaries)

        ansatz = bind(ansatz, parameter_binds)

        bound_unitaries = bind(unitaries, parameter_binds)

        # lam = reduce(lambda x, y: x.evolve(y), ulist, self.state_in).evolve(self.operator)
        zero = Statevector.from_int(0, (2,) * ansatz.num_qubits)
        lam = Statevector(ansatz).evolve(operator)

        grads = []
        for j in range(num_parameters):
            grad = 0

            deriv = analytic_gradient(unitaries[j], paramlist[j][0])
            for _, gate in deriv:
                bind(gate, parameter_binds, inplace=True)

            for coeff, gate in deriv:
                dj_unitaries = (
                    bound_unitaries[: max(0, j)]
                    + [gate]
                    + bound_unitaries[min(num_parameters, j + 1) :]
                )
                phi = reduce(lambda x, y: x.evolve(y), dj_unitaries, zero)
                grad += coeff * lam.conjugate().data.dot(phi.data)
            grads += [2 * grad.real]

        accumulated, unique_params = accumulate_product_rule(paramlist, grads)
        return accumulated
