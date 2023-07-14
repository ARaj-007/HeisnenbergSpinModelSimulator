import numpy as np
from scipy.optimize import minimize
from heisenberg_spin import heisenberg_hamiltonian
from qiskit import transpile, execute

def vqe_algorithm(ansatz, backend, num_shots=1000, optimizer='COBYLA', max_evals=100):
    def objective(params):
        circuit = ansatz.copy()
        circuit = transpile(circuit, backend)
        circuit = circuit.bind_parameters(params)
        return measure_energy(circuit, backend, num_shots)

    initial_params = np.random.uniform(0, 2 * np.pi, ansatz.num_parameters)
    result = minimize(objective, initial_params, method=optimizer, options={'maxiter': max_evals})
    return result.fun, result.x

def measure_energy(circuit, backend, num_shots):
    meas_circuits = [circuit.measure_all()]
    job = execute(meas_circuits, backend, shots=num_shots)
    result = job.result()
    counts = result.get_counts(0)
    energy = 0
    for state, count in counts.items():
        state_vector = np.array([int(bit) for bit in state[::-1]])
        energy += count * np.dot(state_vector, np.dot(heisenberg_hamiltonian(), state_vector))
    return energy / num_shots
