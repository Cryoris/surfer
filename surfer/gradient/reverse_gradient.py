"""A class to compute gradients of expectation values."""

from qiskit.quantum_info import Statevector

from surfer.tools.split_circuit import split
from surfer.tools.gradient_lookup import analytic_gradient
from surfer.tools.bind import bind
from surfer.tools.accumulate_product_rule import accumulate_product_rule
from surfer.tools.unroll_parameterized_gates import UnrollParameterizedGates

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

        supported_parameterized_gates = [
            "rx",
            "ry",
            "rz",
            "cp",
            "crx",
            "cry",
            "crz",
        ]
        self.unroller = UnrollParameterizedGates(supported_parameterized_gates)

    def compute(self, operator, circuit, values, parameters=None):
        # try unrolling to a supported basis
        circuit = self.unroller(circuit)

        if parameters is None:
            parameters = "free"

        original_parameter_order = circuit.parameters
        unitaries, paramlist = split(
            circuit, parameters=parameters, return_parameters=True
        )
        parameter_binds = dict(zip(circuit.parameters, values))

        num_parameters = len(unitaries)

        circuit = bind(circuit, parameter_binds)

        phi = Statevector(circuit)
        lam = phi.evolve(operator)

        # store gradients in a dictionary to return them in the correct order
        grads = {param: 0 for param in original_parameter_order}

        for j in reversed(range(num_parameters)):
            uj = unitaries[j]  # pylint: disable=invalid-name

            # we currently only support gates with a single parameter,
            # as soon as we support multiple parameters per gate we have to
            # iterate over all parameters in the parameter list
            paramj = paramlist[j][0]

            deriv = analytic_gradient(uj, paramj)
            for _, gate in deriv:
                bind(gate, parameter_binds, inplace=True)

            uj_dagger = bind(uj, parameter_binds).inverse()

            phi = phi.evolve(uj_dagger)

            grad = sum(
                coeff * lam.conjugate().data.dot(phi.evolve(gate).data)
                for coeff, gate in deriv
            )
            if self._partial_gradient:
                grads[paramj] += grad
            else:
                grads[paramj] += 2 * grad.real

            if j > 0:
                lam = lam.evolve(uj_dagger)

        return list(grads.values())
