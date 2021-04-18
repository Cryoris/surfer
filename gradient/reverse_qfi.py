"""A class to compute gradients of expectation values."""

import numpy as np
from qiskit.quantum_info import Statevector, Operator
from .split_circuit import split
from .gradient_lookup import analytic_gradient
from .circuit_gradients import _bind


class ReverseQFI:
    """A class to compute gradients of expectation values."""

    def compute(self, ansatz, values):
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

        phase_fixes = np.zeros(num_parameters, dtype=complex)
        lis = np.zeros((num_parameters, num_parameters), dtype=complex)

        chi = Statevector(bound_unitaries[0])
        psi = chi.copy()
        phi = Statevector.from_int(0, (2,) * ansatz.num_qubits)

        deriv = analytic_gradient(unitaries[0], paramlist[0][0])
        for _, gate in deriv:
            _bind(gate, parameter_binds, inplace=True)

        grad_coeffs = [coeff for coeff, _ in deriv]
        grad_states = [phi.evolve(gate) for _, gate in deriv]

        # TODO compute the phis once
        phase_fixes[0] = sum(
            c_i * chi.conjugate().data.dot(state_i.data)
            for c_i, state_i in zip(grad_coeffs, grad_states)
        )
        lis[0, 0] = sum(
            sum(
                np.conj(c_i) * c_j * state_i.conjugate().data.dot(state_j.data)
                for c_i, state_i in zip(grad_coeffs, grad_states)
            )
            for c_j, state_j in zip(grad_coeffs, grad_states)
        )

        for j in range(1, num_parameters):
            lam = psi.copy()
            phi = psi.copy()

            # get d_j U_j
            uj = unitaries[j]
            deriv = analytic_gradient(uj, paramlist[j][0])

            for _, gate in deriv:
                _bind(gate, parameter_binds, inplace=True)

            # compute |phi> (in general it's a sum of states and coeffs)
            grad_coeffs = [coeff for coeff, _ in deriv]
            grad_states = [phi.evolve(gate.decompose()) for _, gate in deriv]

            # compute L_{j, j}
            lis[j, j] = sum(
                sum(
                    np.conj(c_i) * c_j * state_i.conjugate().data.dot(state_j.data)
                    for c_i, state_i in zip(grad_coeffs, grad_states)
                )
                for c_j, state_j in zip(grad_coeffs, grad_states)
            )

            for i in reversed(range(j)):
                # apply U_{i + 1}_dg
                uip_inv = bound_unitaries[i + 1].inverse()
                grad_states = [state.evolve(uip_inv) for state in grad_states]

                lam = lam.evolve(bound_unitaries[i].inverse())

                # get d_i U_i
                ui = unitaries[i]
                deriv = analytic_gradient(ui, paramlist[i][0])
                for _, gate in deriv:
                    _bind(gate, parameter_binds, inplace=True)

                # compute |phi> (in general it's a sum of states and coeffs)
                grad_coeffs_mu = [coeff for coeff, _ in deriv]
                grad_states_mu = [lam.evolve(gate) for _, gate in deriv]

                # compute L_{i, j}
                lis[i, j] = sum(
                    sum(
                        np.conj(c_i) * c_j * state_i.conjugate().data.dot(state_j.data)
                        for c_i, state_i in zip(grad_coeffs_mu, grad_states_mu)
                    )
                    for c_j, state_j in zip(grad_coeffs, grad_states)
                )

            phase_fixes[j] = sum(
                chi.conjugate().data.dot(c_i * (state_i.data))
                for c_i, state_i in zip(grad_coeffs, grad_states)
            )
            psi = psi.evolve(bound_unitaries[j])

        # accumulated, unique_params = accumulate_product_rule(paramlist, list(reversed(grads)))
        # return accumulated

        # stack quantum geometric tensor together
        qgt = np.zeros((num_parameters, num_parameters), dtype=complex)
        for i in range(num_parameters):
            for j in range(num_parameters):
                if i <= j:
                    qgt[i, j] = lis[i, j] - np.conj(phase_fixes[i]) * phase_fixes[j]
                else:
                    qgt[i, j] = (
                        np.conj(lis[j, i]) - np.conj(phase_fixes[i]) * phase_fixes[j]
                    )

        return 4 * np.real(qgt)
