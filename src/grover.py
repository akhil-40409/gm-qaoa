import pennylane as qml
import jax.numpy as jnp
import numpy as np
from typing import List, Union, Any


class GroverAlgorithm:
    """A modular implementation of Grover's Algorithm.

    This class provides tools to construct, iterate, and execute Grover's
    search algorithm with support for multiple target states and JAX integration.
    """

    def __init__(self, num_qubits: int):
        """Initialize the Grover Algorithm.

        Args:
            num_qubits: The number of qubits in the system.
        """
        self.num_qubits = num_qubits
        self.wires = list(range(num_qubits))
        self.N = 2**num_qubits

    def initial_state(self) -> None:
        """Prepare the system in an equal superposition of all states."""
        for wire in self.wires:
            qml.Hadamard(wires=wire)

    def oracle(
        self, target_states: List[Union[str, List[int], jnp.ndarray]]
    ) -> None:
        """Apply the oracle to flip the sign of the target states.

        Args:
            target_states: A list of target states to search for.
                           Can be bitstrings ("101"), lists ([1, 0, 1]),
                           or JAX/NumPy arrays.
        """
        for target in target_states:
            # Convert string bitstrings to arrays if necessary
            if isinstance(target, str):
                target = jnp.array([int(b) for b in target])
            qml.FlipSign(target, wires=self.wires)

    def diffuser(self) -> None:
        """Apply the Grover Diffusion operator (reflection about the mean)."""
        qml.templates.GroverOperator(wires=self.wires)

    def compute_optimal_iterations(self, num_targets: int) -> int:
        """Calculate the optimal number of Grover iterations.

        Args:
            num_targets: The number of states marked by the oracle.

        Returns:
            The optimal number of iterations as an integer.
        """
        if num_targets == 0:
            return 0
        # Optimal iterations formula: floor(pi/4 * sqrt(N/M))
        return jnp.floor(jnp.sqrt(self.N / num_targets) * jnp.pi / 4).astype(
            int
        )

    def circuit(
        self, target_states: List[Union[str, List[int], jnp.ndarray]]
    ) -> None:
        """Construct the complete Grover circuit.

        Args:
            target_states: The list of states the oracle should mark.
        """
        num_targets = len(target_states)
        iterations = self.compute_optimal_iterations(num_targets)

        # 1. State Preparation
        self.initial_state()

        # 2. Grover Iterations
        for _ in range(iterations):
            self.oracle(target_states)
            self.diffuser()

    def get_qnode(
        self, device: Any = None, interface: str = "jax"
    ) -> qml.QNode:
        """Create a PennyLane QNode for this Grover instance.

        Args:
            device: The PennyLane device to run on. Defaults to 'default.qubit'.
            interface: The interface to use (e.g., 'jax', 'autograd').

        Returns:
            A compiled QNode that takes 'target_states' as input.
        """
        if device is None:
            device = qml.device("default.qubit", wires=self.wires)

        @qml.qnode(device, interface=interface)
        def grover_qnode(target_states):
            self.circuit(target_states)
            return qml.probs(wires=self.wires)

        return grover_qnode
