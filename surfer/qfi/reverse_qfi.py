"""A class to compute gradients of expectation values."""

from typing import List, Optional
import numpy as np
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import Statevector

from surfer.tools.unroll_parameterized_gates import UnrollParameterizedGates
from surfer.tools.split_circuit import split
from surfer.tools.gradient_lookup import analytic_gradient
from surfer.tools.bind import bind

from .qfi import QFICalculator


class ReverseQFI(QFICalculator):
    """A class to compute gradients of expectation values."""

    supported_parameterized_gates = [
        "rx",
        "ry",
        "rz",
        "cp",
        "crx",
        "cry",
        "crz",
    ]

    def __init__(self, do_checks: bool = True, phase_fix: bool = True):
        """
        Args:
            do_checks: Do some sanity checks on the inputs. Can be disabled for performance.
            phase_fix: Whether or not to include the phase fix.
        """
        super().__init__(do_checks)
        self.phase_fix = phase_fix

        self.unroller = UnrollParameterizedGates(
            ReverseQFI.supported_parameterized_gates
        )

    # pylint: disable=too-many-locals
    def compute(
        self,
        circuit: QuantumCircuit,
        values: np.ndarray,
        parameters: Optional[List[Parameter]] = None,
    ):
        if self.do_checks:
            self.check_inputs(circuit, values)

        # cast to array, if values are a list
        if not isinstance(values, np.ndarray):
            values = np.asarray(values)

        circuit = self.unroller(circuit)

        if parameters is None:
            parameters = "free"
            original_parameter_order = circuit.parameters
            num_parameters = circuit.num_parameters
        else:
            if not isinstance(parameters, list):
                parameters = [parameters]
            num_parameters = len(parameters)
            original_parameter_order = [
                param for param in circuit.parameters if param in parameters
            ]

        unitaries, paramlist = split(
            circuit, parameters=parameters, return_parameters=True
        )
        parameter_binds = dict(zip(circuit.parameters, values))

        num_unitaries = len(unitaries)

        circuit = bind(circuit, parameter_binds)

        bound_unitaries = bind(unitaries, parameter_binds)

        # phase_fixes = np.zeros(num_unitaries, dtype=complex)
        phase_fixes = {param: 0j for param in original_parameter_order}
        lis = {
            (p_i, p_j): 0j
            for p_i in original_parameter_order
            for p_j in original_parameter_order
        }
        # lis = np.zeros((num_unitaries, num_unitaries), dtype=complex)

        chi = Statevector(bound_unitaries[0])
        psi = chi.copy()
        phi = Statevector.from_int(0, (2,) * circuit.num_qubits)

        deriv = analytic_gradient(unitaries[0], paramlist[0][0])
        for _, gate in deriv:
            bind(gate, parameter_binds, inplace=True)

        grad_coeffs = [coeff for coeff, _ in deriv]
        grad_states = [phi.evolve(gate) for _, gate in deriv]

        p0 = paramlist[0][0]
        if self.phase_fix:
            phase_fixes[p0] = _phasefix_term(chi, grad_coeffs, grad_states)

        lis[(p0, p0)] = _l_term(grad_coeffs, grad_states, grad_coeffs, grad_states)

        for j in range(1, num_unitaries):
            p_j = paramlist[j][0]
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
            lis[(p_j, p_j)] += _l_term(
                grad_coeffs, grad_states, grad_coeffs, grad_states
            )

            for i in reversed(range(j)):
                p_i = paramlist[i][0]

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
                lis[(p_i, p_j)] += _l_term(
                    grad_coeffs_mu, grad_states_mu, grad_coeffs, grad_states
                )

            if self.phase_fix:
                phase_fixes[p_j] += _phasefix_term(chi, grad_coeffs, grad_states)

            psi = psi.evolve(bound_unitaries[j])

        # stack quantum geometric tensor together
        qgt = np.zeros((num_parameters, num_parameters), dtype=complex)
        for i in range(num_parameters):
            p_i = original_parameter_order[i]
            for j in range(num_parameters):
                p_j = original_parameter_order[j]
                if i <= j:
                    qgt[i, j] = (
                        lis[(p_i, p_j)] - np.conj(phase_fixes[p_i]) * phase_fixes[p_j]
                    )
                else:
                    qgt[i, j] = (
                        np.conj(lis[(p_j, p_i)])
                        - np.conj(phase_fixes[p_i]) * phase_fixes[p_j]
                    )

        return 4 * np.real(qgt)


def _l_term(coeffs_i, states_i, coeffs_j, states_j):
    return sum(
        sum(
            np.conj(c_i) * c_j * state_i.conjugate().data.dot(state_j.data)
            for c_i, state_i in zip(coeffs_i, states_i)
        )
        for c_j, state_j in zip(coeffs_j, states_j)
    )


def _phasefix_term(chi, coeffs, states):
    return sum(
        c_i * chi.conjugate().data.dot(state_i.data)
        for c_i, state_i in zip(coeffs, states)
    )
