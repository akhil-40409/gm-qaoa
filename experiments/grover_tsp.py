import sys
import os
import numpy as np
import pennylane as qml
import timeit
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.grover import GroverAlgorithm
from src.tsp import TSPInstance

def run_tsp_grover(num_cities, device_name="default.qubit", interface="jax"):
    """Execute Grover search on a TSP instance and return metrics."""
    # 1. Setup TSP
    tsp = TSPInstance(num_cities=num_cities, seed=42)
    best_tour, min_cost = tsp.get_optimal_tour()
    
    # 2. Find target indices (optimal tours)
    target_indices = []
    for i, tour in enumerate(tsp.all_tours):
        if np.isclose(tsp.calculate_cost(tour), min_cost):
            target_indices.append(i)
            
    # 3. Setup Grover
    num_feasible = len(tsp.all_tours)
    num_qubits = int(np.ceil(np.log2(num_feasible)))
    grover = GroverAlgorithm(num_qubits=num_qubits)
    
    # Convert indices to bitstrings for the oracle
    target_bitstrings = [format(idx, f'0{num_qubits}b') for idx in target_indices]
    
    # 4. Device and QNode
    dev = qml.device(device_name, wires=num_qubits)
    qnode = grover.get_qnode(device=dev, interface=interface)
    
    # 5. Measure Execution Time
    def _execute():
        return qnode(target_bitstrings)
    
    runtimes = timeit.repeat(_execute, number=1, repeat=5)
    mean_time = np.mean(runtimes)
    
    # Final execution to get probabilities
    probs = _execute()
    
    # 6. Specs for Circuit Depth
    specs = qml.specs(qnode)(target_bitstrings)
    depth = specs['resources'].depth
    
    # Success probability (sum of probabilities of all optimal tours)
    # We convert probs to numpy array for indexing
    probs_np = np.array(probs)
    success_prob = np.sum([probs_np[idx] for idx in target_indices])
    
    return {
        "device": device_name,
        "mean_time": mean_time,
        "success_prob": float(success_prob),
        "depth": depth,
        "num_qubits": num_qubits,
        "valid_indices_count": num_feasible
    }

def main():
    print("="*50)
    print(" Grover Search TSP Baseline Benchmark ")
    print("="*50)
    
    for n in [3, 4]:
        print(f"\n--- TSP with {n} Cities ---")
        devices = ["default.qubit", "lightning.qubit"]
        results = []
        for dev in devices:
            try:
                res = run_tsp_grover(n, dev)
                results.append(res)
                print(f"Device: {dev:<15} | Time: {res['mean_time']:.4e}s | Success: {res['success_prob']:.4f} | Depth: {res['depth']}")
            except Exception as e:
                print(f"Device {dev} failed: {e}")

if __name__ == "__main__":
    main()
