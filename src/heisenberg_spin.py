'''
Molecular Hydrogen (H2) molecule using the ParityMapper
Defining the qubit operator for the H2 molecule
'''

from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.mappers import ParityMapper
qiskit_nature.settings.use_pauli_sum_op = False
from qiskit_nature.second_q.drivers import PySCFDriver


# Defining the molecular information including atomic symbols, coordinates, multiplicity, and charge
molecule = MoleculeInfo(
    # Coordinates in Angstrom
    symbols=["H", "H"],  # Atomic symbols of the two hydrogen atoms
    coords=([0.0, 0.0, -0.3625], [0.0, 0.0, 0.3625]),  # Coordinates of the two hydrogen atoms
    multiplicity=1,  # Multiplicity of the molecule (1 for singlet)
    charge=0,  # Charge of the molecule (neutral charge)
)

# Create a PySCF driver to compute the molecular integrals
driver = PySCFDriver.from_molecule(molecule)

# Run the driver to obtain the electronic structure problem
problem = driver.run()

# Extract the second quantized operators representing the Hamiltonian
second_q_ops = problem.second_q_ops()

# Get the number of spatial orbitals and the number of particles in the system
num_spatial_orbitals = problem.num_spatial_orbitals
num_particles = problem.num_particles

# We are using ParityMapper to map the electronic structure problem to qubits, 
mapper = ParityMapper(num_particles=num_particles)


hamiltonian = second_q_ops[0]

# Performing two-qubit reduction
qubit_op = mapper.map(hamiltonian)
print(qubit_op)

repulsion_energy = problem.nuclear_repulsion_energy# we will use this later to adjust the total energy
print(repulsion_energy)

Hamiltonian_matrix = qubit_op.to_matrix()
print(Hamiltonian_matrix)