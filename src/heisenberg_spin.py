import numpy as np

def heisenberg_hamiltonian():
    sx = np.array([[0, 1], [1, 0]])
    sy = np.array([[0, -1j], [1j, 0]])
    sz = np.array([[1, 0], [0, -1]])
    h = np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)
    return h

def calculate_eigenvalues(matrix):
    eigenvalues, _ = np.linalg.eig(matrix)
    return eigenvalues

def calculate_eigenvectors(matrix):
    _, eigenvectors = np.linalg.eig(matrix)
    return eigenvectors

# Additional utility functions or classes related to the Heisenberg spin model
