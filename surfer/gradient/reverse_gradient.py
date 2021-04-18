"""A class to compute gradients of expectation values."""

from qiskit.quantum_info import Statevector

from surfer.tools.split_circuit import split
from surfer.tools.gradient_lookup import analytic_gradient
from surfer.tools.bind import bind
from surfer.tools.accumulate_product_rule import accumulate_product_rule


class ReverseGradient:
    """A class to compute gradients of expectation values."""

    def compute(self, operator, ansatz, values):
        """
        Args:
            operator (OperatorBase): The operator in the expectation value.
            ansatz (QuantumCircuit): The ansatz in the expecation value.
            target_parameters (List[Parameter]): The parameters with respect to which to derive.
                If None, the derivative for all parameters is computed (also bound parameters!).
        """
        unitaries, paramlist = split(ansatz, return_parameters=True)
        parameter_binds = dict(zip(ansatz.parameters, values))

        num_parameters = len(unitaries)

        ansatz = bind(ansatz, parameter_binds)

        bound_unitaries = bind(unitaries, parameter_binds)

        phi = Statevector(ansatz)
        lam = phi.evolve(operator)
        grads = []
        for j in reversed(range(num_parameters)):
            uj = unitaries[j]

            deriv = analytic_gradient(uj, paramlist[j][0])
            for _, gate in deriv:
                bind(gate, parameter_binds, inplace=True)

            uj_dagger = bind(uj, parameter_binds).inverse()

            phi = phi.evolve(uj_dagger)

            # TODO use projection
            grad = (
                2
                * sum(
                    coeff * lam.conjugate().data.dot(phi.evolve(gate).data)
                    for coeff, gate in deriv
                ).real
            )
            grads += [grad]

            if j > 0:
                lam = lam.evolve(uj_dagger)

        accumulated, _ = accumulate_product_rule(paramlist, list(reversed(grads)))
        return accumulated
