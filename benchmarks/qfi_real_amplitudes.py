"""Runtime comparison for QFI on the Real Amplitudes circuit."""

import numpy as np
import time

from qiskit.providers.aer import QasmSimulator
from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import AerPauliExpectation, CircuitSampler

from surfer.qfi import LinearCombination, ReverseQFI

num_qubits = 4
reps = 2 ** np.arange(6)
avg = 5

lin_comb_plain = LinearCombination()
lin_comb_aerpauli = LinearCombination(
    CircuitSampler(QasmSimulator()), AerPauliExpectation())
reverse_qfi = ReverseQFI()

methods = [
    lin_comb_plain,
    lin_comb_aerpauli,
    reverse_qfi
]

labels = [
    "Matmult QFI",
    "AerPauli QFI",
    "Reverse QFI"
]

# store all runtimes in a dictionary
times = {label: {'means': [], 'std': []} for label in labels}

circuit = RealAmplitudes(num_qubits)

try:
    for rep in reps:
        print()
        print('Rep:', rep)
        print('-' * 50)
        circuit.reps = rep
        values = np.random.random(circuit.num_parameters)

        for label, method in zip(labels, methods):
            runtimes = []
            print('Method:', label)
            for _ in range(avg):
                start = time.time()
                _ = method.compute(circuit, values)
                runtimes.append(time.time() - start)

            times[label]['means'].append(np.mean(runtimes))
            times[label]['std'].append(np.std(runtimes))

except KeyboardInterrupt:
    print('Interrupted! Dumping now.')

print(times)
np.save('test', times)
