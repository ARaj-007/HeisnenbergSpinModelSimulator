from qiskit import QuantumCircuit
import numpy as np

def random_circuit_ansatz(num_qubits, depth):
    circuit = QuantumCircuit(num_qubits)
    for _ in range(depth):
        for qubit in range(num_qubits):
            rand_gate = np.random.choice(['h', 'rx', 'ry', 'rz'])
            rand_angle = np.random.uniform(0, 2 * np.pi)
            if rand_gate == 'h':
                circuit.h(qubit)
            elif rand_gate == 'rx':
                circuit.rx(rand_angle, qubit)
            elif rand_gate == 'ry':
                circuit.ry(rand_angle, qubit)
            elif rand_gate == 'rz':
                circuit.rz(rand_angle, qubit)
        for qubit1 in range(num_qubits - 1):
            for qubit2 in range(qubit1 + 1, num_qubits):
                if np.random.rand() < 0.5:
                    circuit.cx(qubit1, qubit2)
    return circuit
