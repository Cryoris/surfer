# pylint: disable=inconsistent-return-statements
def bind(circuits, parameter_binds, inplace=False):
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
