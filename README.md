# GM-QAOA for Traveling Salesperson Problem

Implementation and benchmarking of Grover-mixer QAOA (GM-QAOA) for the Traveling Salesperson Problem using PennyLane backends (`default.qubit`, `lightning.qubit`, and `catalyst`).

This project implements the algorithms described in the paper:
**[Grover-mixer QAOA: A quantum algorithm for constrained optimization](https://arxiv.org/abs/2005.00941)** (Bärtschi and Eidenbenz, 2020).

## Features
- **Grover-Mixer QAOA**: Implementation of search-space restricted mixers for constrained optimization.
- **TSP Encoding**: Efficient encoding of the Traveling Salesperson Problem (TSP) for quantum circuits.
- **Backends**: Execution and benchmarking across `default.qubit`, `lightning.qubit`, and JIT compilation via `catalyst`.

## Directory Structure
- `src/`: Core implementation
  - `grover.py`: GM-QAOA mixer and circuit implementation
  - `tsp.py`: TSP problem encoding and cost function
- `experiments/`: Research notebooks and benchmarking scripts
  - `grover_tsp.py`: Main benchmarking script
  - `grover_tsp.ipynb`: Interactive demonstration of TSP on small instances
- `tests/`: Unit tests for implementation validation

## Getting Started

### Dependencies
This project uses `uv` for dependency management.
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Running Experiments
To run the TSP benchmarking script:
```bash
python experiments/grover_tsp.py
```

### Running Tests
To run the test suite:
```bash
pytest
```
