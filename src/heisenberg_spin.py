import numpy as np 

def heisenberg_hamiltonian():
    sx = np.array([[0, 1], [1, 0]])
    sy = np.array([[0, -1j], [1j, 0]])
    sz = np.array([[1, 0], [0, -1]])
    h = np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)
    return h