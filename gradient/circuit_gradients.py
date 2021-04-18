"""A class to compute gradients of expectation values."""

from functools import reduce
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from .split_circuit import split
from .gradient_lookup import analytic_gradient


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

        ansatz = _bind(ansatz, parameter_binds)

        bound_unitaries = _bind(unitaries, parameter_binds)

        phi = Statevector(ansatz)
        lam = phi.evolve(operator)
        grads = []
        for j in reversed(range(num_parameters)):
            uj = unitaries[j]

            deriv = analytic_gradient(uj, paramlist[j][0])
            for _, gate in deriv:
                _bind(gate, parameter_binds, inplace=True)

            uj_dagger = _bind(uj, parameter_binds).inverse()

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

        accumulated, unique_params = accumulate_product_rule(
            paramlist, list(reversed(grads))
        )
        return accumulated


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
        print(paramlist)
        print(values)
        parameter_binds = dict(zip(ansatz.parameters, values))

        num_parameters = len(unitaries)

        ansatz = _bind(ansatz, parameter_binds)

        bound_unitaries = _bind(unitaries, parameter_binds)

        # lam = reduce(lambda x, y: x.evolve(y), ulist, self.state_in).evolve(self.operator)
        zero = Statevector.from_int(0, (2,) * ansatz.num_qubits)
        lam = Statevector(ansatz).evolve(operator)

        grads = []
        for j in range(num_parameters):
            grad = 0

            deriv = analytic_gradient(unitaries[j], paramlist[j][0])
            for _, gate in deriv:
                _bind(gate, parameter_binds, inplace=True)

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


def accumulate_product_rule(paramlist, gradients):
    grads = {}
    for paramlist, grad in zip(paramlist, gradients):
        # all our gates only have one single parameter
        param = paramlist[0]
        grads[param] = grads.get(param, 0) + grad

    return list(grads.values()), list(grads.keys())


# pylint: disable=inconsistent-return-statements
def _bind(circuits, parameter_binds, inplace=False):
    if not isinstance(circuits, list):
        existing_parameter_binds = {p: parameter_binds[p] for p in circuits.parameters}
        return circuits.assign_parameters(existing_parameter_binds, inplace=inplace)

    bound = []
    for circuit in circuits:
        existing_parameter_binds = {p: parameter_binds[p] for p in circuit.parameters}
        bound.append(
            circuit.assign_parameters(existing_parameter_binds, inplace=inplace)
        )

    if not inplace:
        return bound
