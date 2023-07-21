'''
Variational Quantum Eigensolver (VQE) algorithm to find the ground state energy
and the function to calculate the expectation value of a quantum circuit with respect to a given Hamiltonian.
Here we are using statevector simulator backend to calculate the expectation value.
'''

from qiskit import Aer, execute
import numpy as np
from qiskit.exceptions import QiskitError
from .random_ansatz import assign_parameters, random_params
from qiskit.algorithms.optimizers import SPSA, SLSQP, COBYLA
import time

def calculate_expectation(circuit, params, hamiltonian):
    """
    Calculate the expectation value of a quantum circuit with respect to a given Hamiltonian.

    Args:
        circuit (QuantumCircuit): The quantum circuit to evaluate.
        params (numpy.ndarray): An array of parameter values to bind to the circuit.
        hamiltonian matrix : The Hamiltonian operator.

    Returns:
        float: The expectation value of the circuit with respect to the Hamiltonian.

    Notes:
        This function calculates the expectation value of a quantum circuit with respect to
        a given Hamiltonian. The circuit is parameterized using Parameter objects, and the
        provided parameter values are bound to the circuit before evaluation.

        The function simulates the quantum circuit on the statevector simulator backend
        and computes the expectation value using the statevector representation of the circuit's
        final state. The expectation value is calculated as the inner product between the
        statevector and the Hamiltonian operator.

        The function returns the total expected energy.

    Raises:
        QiskitError: If the circuit evaluation job fails or the Hamiltonian is invalid.
    """
    backend = Aer.get_backend('statevector_simulator')

    # Bind the provided parameter values to the circuit
    bound_circuit = circuit.bind_parameters({p: val for p, val in zip(circuit.parameters, params.flatten())})

    try:
        # Execute the bound circuit on the statevector simulator backend
        job = execute(bound_circuit, backend)
        result = job.result()
        statevector = result.get_statevector()
    except Exception as exc:
        raise QiskitError("Failed to evaluate the circuit on the statevector simulator!") from exc

    # Calculate the expectation value as the inner product between statevector and Hamiltonian
    expectation = np.vdot(statevector, np.dot(hamiltonian, statevector))
    return expectation.real


def my_vqe(ansatz_circuit, qubit_op, max_iterations=100, tolerance=1e-4):
    """
    Perform the Variational Quantum Eigensolver (VQE) algorithm to find the ground state energy
    of a given Hamiltonian using a parameterized quantum circuit (ansatz).

    Args:
        ansatz_circuit (QuantumCircuit): The parameterized quantum circuit (ansatz) to prepare the quantum state.
        qubit_op (PauliSumOp | SparsePauliOp | BaseOperator): The Hamiltonian as a qubit operator for which
            the ground state energy is to be calculated.
        max_iterations (int, optional): The maximum number of iterations for the optimizer (default is 100).
        tolerance (float, optional): The convergence tolerance for the optimizer (default is 1e-4).

    Returns:
        tuple: A tuple containing the following elements:
            - numpy.ndarray: The optimal parameters for the ansatz circuit.
            - float: The final ground state energy obtained after VQE optimization.
            - float: The time taken for the VQE optimization.

    Note:
        The objective function of the COBYLA optimizer is defined internally to update the circuit,
        print intermediate energies in steps of 5, and return the intermediate energy during optimization.

    Example:
        num_qubits = 4
        depth = 3
        ansatz_circuit = random_circuit_ansatz(num_qubits, depth)
        params, final_energy, optimizer_time = my_vqe(ansatz_circuit, hamiltonian)
        print(f"Optimal parameters: {params}")
        print(f"Final ground state energy: {final_energy}")
        print(f"Optimizer time: {optimizer_time} seconds")
    """
    
    #Random initialization of the circuit parameters
    num_params = len(ansatz_circuit.parameters)
    params = random_params(num_params)
    circuit = assign_parameters(ansatz_circuit, params)
    hamiltonian = qubit_op.to_matrix()
    
    previous_energy = float('inf')

    optimizer = COBYLA(maxiter=max_iterations, tol=tolerance)

    # The objective function for the optimizer
    eval_count = 0
    def objective_function(p):
        nonlocal eval_count
        updated_circuit = assign_parameters(ansatz_circuit, p)
        intermediate_energy = calculate_expectation(updated_circuit, p, hamiltonian)
        eval_count += 1
        if eval_count%5 == 0:
            print(f"Iteration {eval_count}: Energy = {intermediate_energy}")
        return intermediate_energy

    start_time = time.time()
    result = optimizer.minimize(objective_function, x0=params)
    end_time = time.time()

    optimal_params = result.x
    final_energy = calculate_expectation(ansatz_circuit, optimal_params, hamiltonian)
    optimizer_time = end_time - start_time

    return params, final_energy, optimizer_time
