"""A class to compute gradients of expectation values."""

from qiskit.quantum_info import Statevector

from surfer.tools.split_circuit import split
from surfer.tools.gradient_lookup import analytic_gradient
from surfer.tools.bind import bind
from surfer.tools.accumulate_product_rule import accumulate_product_rule

from .gradient import GradientCalculator


class ReverseGradient(GradientCalculator):
    """Reverse mode gradient calculation, scaling linearly in the number of parameters."""

    def __init__(self, partial_gradient=False, do_checks=True):
        """
        Args:
            partial_gradient (bool): If True, evaluate <psi|H|grad psi> instead of
                grad <psi|H|psi>.
        """
        super().__init__(do_checks)
        self._partial_gradient = partial_gradient

    def compute(self, operator, circuit, values):
        unitaries, paramlist = split(circuit, return_parameters=True)
        parameter_binds = dict(zip(circuit.parameters, values))

        num_parameters = len(unitaries)

        circuit = bind(circuit, parameter_binds)

        phi = Statevector(circuit)
        lam = phi.evolve(operator)
        grads = []
        for j in reversed(range(num_parameters)):
            uj = unitaries[j]  # pylint: disable=invalid-name

            deriv = analytic_gradient(uj, paramlist[j][0])
            for _, gate in deriv:
                bind(gate, parameter_binds, inplace=True)

            uj_dagger = bind(uj, parameter_binds).inverse()

            phi = phi.evolve(uj_dagger)

            # TODO use projection
            grad = sum(
                coeff * lam.conjugate().data.dot(phi.evolve(gate).data)
                for coeff, gate in deriv
            )
            if self._partial_gradient:
                grads += [grad]
            else:
                grads += [2 * grad.real]

            if j > 0:
                lam = lam.evolve(uj_dagger)

        accumulated, _ = accumulate_product_rule(
            paramlist, list(reversed(grads)))
        return accumulated
