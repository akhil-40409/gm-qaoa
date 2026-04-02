import pytest
import jax.numpy as jnp
from src.grover import GroverAlgorithm
import pennylane as qml

def test_grover_initialization():
    grover = GroverAlgorithm(num_qubits=3)
    assert grover.num_qubits == 3
    assert grover.wires == [0, 1, 2]
    assert grover.N == 8

def test_optimal_iterations():
    grover = GroverAlgorithm(num_qubits=3)
    # N=8, M=1 -> sqrt(8/1)*pi/4 = 2.22 -> floor 2
    assert grover.compute_optimal_iterations(num_targets=1) == 2
    # N=8, M=2 -> sqrt(8/2)*pi/4 = 1.57 -> floor 1
    assert grover.compute_optimal_iterations(num_targets=2) == 1

def test_grover_execution_single_target():
    num_qubits = 2
    target = "11"
    grover = GroverAlgorithm(num_qubits=num_qubits)
    qnode = grover.get_qnode(interface="jax")
    
    probs = qnode([target])
    
    # Decoded index 11 is 3
    assert jnp.argmax(probs) == 3
    assert probs[3] > 0.9  # Should find target with high prob

def test_grover_execution_multi_target():
    num_qubits = 3
    targets = ["000", "111"]
    grover = GroverAlgorithm(num_qubits=num_qubits)
    qnode = grover.get_qnode(interface="jax")
    
    probs = qnode(targets)
    
    # Indices 0 and 7 should be highest
    assert jnp.argmax(probs) in [0, 7]
    assert probs[0] > 0.4
    assert probs[7] > 0.4
    assert probs[1] < 0.1
