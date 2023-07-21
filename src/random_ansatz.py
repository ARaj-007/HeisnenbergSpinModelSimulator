import numpy as np
from qiskit import QuantumCircuit, Parameter

def random_circuit_ansatz(num_qubits, depth, include_two_qubit_gates=True):
    """
    Generate a random quantum circuit ansatz.

    Args:
        num_qubits (int): The number of qubits in the circuit.
        depth (int): The depth of the circuit, i.e., the number of layers of gates.
        include_two_qubit_gates (bool): If True, include two-qubit gates in the circuit.

    Returns:
        QuantumCircuit: A randomly generated quantum circuit ansatz.

    Notes:
        This function generates a random quantum circuit ansatz with the specified number
        of qubits and depth. The circuit can include single-qubit gates (Hadamard, RX, RY, and RZ),
        and optionally, two-qubit gates (CX gates).

        The circuit is parameterized using Parameter objects, allowing for later optimization
        of the circuit's parameters.

        If include_two_qubit_gates is True, the function randomly adds CX gates between pairs
        of qubits with a 50% probability.

        The generated quantum circuit is returned for further use in VQE or other quantum algorithms.
    """
    circuit = QuantumCircuit(num_qubits)

    # Create parameter objects for the circuit parameters
    circuit_params = [[Parameter(f"theta_{d}_{q}") for q in range(num_qubits)] for d in range(depth)]

    for d in range(depth):
        for q in range(num_qubits):
            rand_gate = np.random.choice(['h', 'rx', 'ry', 'rz'])
            rand_param = circuit_params[d][q]  # Get the corresponding parameter object

            if rand_gate == 'h':
                circuit.h(q)
            elif rand_gate == 'rx':
                circuit.rx(rand_param, q)
            elif rand_gate == 'ry':
                circuit.ry(rand_param, q)
            elif rand_gate == 'rz':
                circuit.rz(rand_param, q)
        if include_two_qubit_gates:
            for qubit1 in range(num_qubits - 1):
                for qubit2 in range(qubit1 + 1, num_qubits):
                    if np.random.rand() < 0.5:
                        circuit.cx(qubit1, qubit2)

    return circuit

# Example usage:
num_qubits = 3  # Number of qubits in the circuit
depth = 5  # Depth of the circuit

ansatz_circuit = random_circuit_ansatz(num_qubits, depth, include_two_qubit_gates=True)
print("Ansatz Circuit:")
print(ansatz_circuit)
print("Number of parameters: ")
print(ansatz_circuit.num_parameters)
print("\nAnsatz Parameters:")
print(ansatz_circuit.parameters)

# Helper function to assign parameters to the circuit 
def assign_parameters(circuit, params):
    # Get the list of circuit parameters
    circuit_params = circuit.parameters
    
    # Check if the number of circuit parameters matches the number of elements in params
    if len(circuit_params) != len(params):
        raise ValueError("Number of circuit parameters does not match the length of the input array.")
    
    # Assign the parameters to the circuit by binding them
    for param, value in zip(circuit_params, params):
        circuit.bind_parameters({param: value})
    
    return circuit

# Function to generate random parameters in specified bounds
def random_params(num_params):
    # Define the range for the random parameters (e.g., 0 to 2*pi)
    lower_bound = 0
    upper_bound = 2 * np.pi
    
    # Generate random parameters using uniform distribution
    params = np.random.uniform(lower_bound, upper_bound, num_params)
    
    return params

