
# VQE (Variational Quantum Eigensolver) Project

![Alt Text](imgs/QuantumComputing.jpg)
This repository contains the implementation of a custom made Variational Quantum Eigensolver (VQE) for finding the ground state energy of a given Hamiltonian by simulating the Heisenberg Spin Model using VQE with Random Circuit Ansatz. The VQE algorithm is a hybrid quantum-classical algorithm that leverages the power of quantum computers to solve optimization problems.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Installation and Usage](#installation-and-usage)
- [Project Work Description](#project-work-description)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Problem Statement
- Calculating the ground state energy of the Heisenberg spin model using a custom made Variational Quantum Eigensolver. 
- The goal is to implement a VQE algorithm with a random circuit ansatz and study the accuracy obtained and time required for varying ansatz depth, as well as the impact of including or excluding two-qubit gates.

## Installation and Usage
To run the code and try out the Quantum VQE algorithm, you can either download the notebook "SimulationUsingRandomCircuitAnsatz.ipynb" and run it on a platform like IBM QuantumLab or if you prefer to run on a local machine in the form of a python script the follow the steps:

### Installation
1. Clone this repository to your local machine:
```bash
git clone https://github.com/ARaj-007/HeisnenbergSpinModelSimulator.git
```
2. Navigate to the project directory:
```bash
cd HeisnenbergSpinModelSimulator 
```
3. Install the required Python packages. You can use `pip` to install the dependencies from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```
4. Navigate to the src folder and run the main script. Feel free to checkout all the src files

You can also run the notebook given locally also but it is preferred to run it on the IBM Quantum Lab server.
