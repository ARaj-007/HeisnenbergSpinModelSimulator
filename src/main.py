'''
 The main script where we test out all the functions and the VQE algorithm
'''
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from .heisenberg_spin import *
from .random_ansatz import *
from .vqe import *
import matplotlib.pyplot as plt
def exact_solver(qubit_op, problem):
    """
    Solves the quantum problem exactly using the NumPyMinimumEigensolver that will be
    used as reference for the results of the custom VQE

    Args:
        qubit_op (WeightedPauliOperator): The qubit operator representing the quantum problem.
        problem (ElectronicStructureProblem): The electronic structure problem containing
                                              information about the molecule and the Hamiltonian.

    Returns:
        dict: A dictionary containing the result of the exact solver.

    Raises:
        AlgorithmError: If an error occurs during the computation.
    """
    # Use NumPyMinimumEigensolver to compute the minimum eigenvalue
    sol = NumPyMinimumEigensolver().compute_minimum_eigenvalue(qubit_op)
    if sol is None:
        raise ValueError("Solver did not find a valid solution.")
    
    result = problem.interpret(sol)
    return result

result = exact_solver(qubit_op, problem)
print(result)

E_FCI = result.total_energies[0].real
print(E_FCI)


# Without 2 qubit gates
depths = np.arange(1, 30, 1)
times = []
vqe_energies = []
for dep in depths:
    var_form =  random_circuit_ansatz(qubit_op.num_qubits,dep, False)#No 2 qubit gates
    optimal_params, final_energy, total_time = my_vqe(var_form, qubit_op)
    vqe_result = final_energy + repulsion_energy #Adding the shift
    vqe_energies.append(vqe_result)
    times.append(total_time)
    print(
        f"Depth: {np.round(dep, 2)}",
        f"VQE Result: {vqe_result:.5f}",
        f"Time: {total_time:.3f} seconds",
    )

print("Accuracy: Depth(without_2-qubit_gates) vs VQE Result")
plt.plot(depths, vqe_energies, label="VQE Result")
plt.plot(depths, [E_FCI] * len(depths), label="Ground State Energy")
plt.xlabel("Depth")
plt.ylabel("Energy")
plt.legend()

# Set y-axis range from -1.3 to -1 and add ticks at steps of 0.01
plt.ylim(-1.3, -1)
y_ticks = [i * 0.01 for i in range(-130, -99)]  # Assuming a range of -1.3 to -1
plt.yticks(y_ticks)

plt.show()

print("Depth(without_2-qubit_gates) vs Time")
plt.plot(depths, times, label="Time")
plt.xlabel("Depth")
plt.ylabel("Time (seconds)")
plt.show()
# With 2 qubit gates

depths_2 = np.arange(1, 30, 1)
times_2 = []
vqe_energies_2 = []
for dep in depths_2:
    var_form =  random_circuit_ansatz(qubit_op.num_qubits,dep, True)#setting inclusion of two-qubit gates to true
    print(var_form)
    optimal_params, final_energy, total_time = my_vqe(var_form, qubit_op)
    vqe_result = final_energy + repulsion_energy #Adding the shift
    vqe_energies_2.append(vqe_result)
    times_2.append(total_time)
    print(
        f"Depth: {np.round(dep, 2)}",
        f"VQE Result: {vqe_result:.5f}",
        f"Time: {total_time:.3f} seconds",
    )

print("Accuracy: Depth(with_random_2-qubit_gates) vs VQE Result")
plt.plot(depths_2, vqe_energies_2, 'o-', label="VQE Result", markersize=5)
plt.plot(depths_2, [E_FCI] * len(depths_2), '--', label="Ground State Energy")
plt.xlabel("Depth")
plt.ylabel("Energy")
plt.legend()

# Set y-axis range from -1.3 to -1 and add ticks at steps of 0.01
plt.ylim(-1.3, -1)
y_ticks = [round(-1.3 + i * 0.01, 2) for i in range(31)]  # Assuming a range of -1.3 to -1
plt.yticks(y_ticks)

plt.grid()  # Add grid lines for better visibility of data points
plt.show()

print("Depth(with_random_2-qubit_gates) vs Time")
plt.plot(depths_2, times_2, label="Time")
plt.xlabel("Depth")
plt.ylabel("Time (seconds)")
plt.show()

# Using EfficientSU2 ansatz and the VQE algorithm from Qiskit
# We will also make use of Estimator to calculate the expectation value
from qiskit_aer.primitives import Estimator
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.circuit.library import EfficientSU2

depths_3 = np.arange(1, 30, 1)
times_3 = []
vqe_energies_3 = []
noiseless_estimator = Estimator(approximation=True)
optimizer = COBYLA(maxiter=100, tol = 1e-4)
for dep in depths_3:
    var_form = EfficientSU2(qubit_op.num_qubits, entanglement="linear", reps = dep)
    start_time = time.time()
    vqe = VQE(noiseless_estimator, var_form, optimizer)
    vqe_calc = vqe.compute_minimum_eigenvalue(qubit_op)
    end_time = time.time()
    vqe_result = problem.interpret(vqe_calc).total_energies[0].real
    vqe_energies_3.append(vqe_result)
    total_time = end_time - start_time
    times_3.append(total_time)
    print(f"Depth {dep}: VQE_Result = {vqe_result}: Time {total_time}")

print("Accuracy: Depth(EfficientSU2) vs VQE Result")
plt.plot(depths_3, vqe_energies_3, 'o-', label="VQE Result", markersize=5)
plt.plot(depths_3, [E_FCI] * len(depths_3), '--', label="Ground State Energy")
plt.xlabel("Depth")
plt.ylabel("Energy")
plt.legend()

# Set y-axis range from -1.3 to -1 and add ticks at steps of 0.01
plt.ylim(-1.3, -1)
y_ticks = [round(-1.3 + i * 0.01, 2) for i in range(31)]  # Assuming a range of -1.3 to -1
plt.yticks(y_ticks)

plt.grid()  # Add grid lines for better visibility of data points
plt.show()

print("Depth(EfficientSU2) vs Time")
plt.plot(depths_3, times_3, label="Time")
plt.xlabel("Depth")
plt.ylabel("Time (seconds)")
plt.show()
