"""A class to compute gradients of expectation values."""

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector

from surfer.tools.split_circuit import split
from surfer.tools.gradient_lookup import analytic_gradient
from surfer.tools.bind import bind

from .qfi import QFICalculator


class ReverseQFI(QFICalculator):
    """A class to compute gradients of expectation values."""

    def __init__(self, do_checks: bool = True, phase_fix: bool = True):
        """
        Args:
            do_checks: Do some sanity checks on the inputs. Can be disabled for performance.
            phase_fix: Whether or not to include the phase fix.
        """
        super().__init__(do_checks)
        self.phase_fix = phase_fix

    # pylint: disable=too-many-locals
    def compute(self, circuit: QuantumCircuit, values: np.ndarray):
        if self.do_checks:
            self.check_inputs(circuit, values)

        unitaries, paramlist = split(circuit, return_parameters=True, parameters="free")
        parameter_binds = dict(zip(circuit.parameters, values))

        num_parameters = len(unitaries)

        circuit = bind(circuit, parameter_binds)

        bound_unitaries = bind(unitaries, parameter_binds)

        phase_fixes = np.zeros(num_parameters, dtype=complex)
        lis = np.zeros((num_parameters, num_parameters), dtype=complex)

        chi = Statevector(bound_unitaries[0])
        psi = chi.copy()
        phi = Statevector.from_int(0, (2,) * circuit.num_qubits)

        deriv = analytic_gradient(unitaries[0], paramlist[0][0])
        for _, gate in deriv:
            bind(gate, parameter_binds, inplace=True)

        grad_coeffs = [coeff for coeff, _ in deriv]
        grad_states = [phi.evolve(gate) for _, gate in deriv]

        if self.phase_fix:
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
            uj = unitaries[j]  # pylint: disable=invalid-name
            deriv = analytic_gradient(uj, paramlist[j][0])

            for _, gate in deriv:
                bind(gate, parameter_binds, inplace=True)

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
                ui = unitaries[i]  # pylint: disable=invalid-name
                deriv = analytic_gradient(ui, paramlist[i][0])
                for _, gate in deriv:
                    bind(gate, parameter_binds, inplace=True)

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

            if self.phase_fix:
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
