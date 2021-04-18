"""A class to compute gradients of expectation values."""

from functools import reduce
from qiskit.quantum_info import Statevector

from surfer.tools.split_circuit import split
from surfer.tools.gradient_lookup import analytic_gradient
from surfer.tools.bind import bind
from surfer.tools.accumulate_product_rule import accumulate_product_rule

from .gradient import GradientCalculator


class ForwardGradient(GradientCalculator):
    """Standard forward gradient calculation, scaling quadratically in the number of parameters."""

    # pylint: disable=too-many-locals
    def compute(self, operator, circuit, values):
        unitaries, paramlist = split(circuit, return_parameters=True)
        parameter_binds = dict(zip(circuit.parameters, values))

        num_parameters = len(unitaries)

        circuit = bind(circuit, parameter_binds)

        bound_unitaries = bind(unitaries, parameter_binds)

        # lam = reduce(lambda x, y: x.evolve(y), ulist, self.state_in).evolve(self.operator)
        zero = Statevector.from_int(0, (2,) * circuit.num_qubits)
        lam = Statevector(circuit).evolve(operator)

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

        accumulated, _ = accumulate_product_rule(paramlist, grads)
        return accumulated
