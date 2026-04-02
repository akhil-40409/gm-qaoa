import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import permutations
from typing import List, Tuple, Optional


class TSPInstance:
    """A class to represent and solve a Traveling Salesperson Problem (TSP) instance.

    This class provides tools for calculating tour costs, finding optimal tours,
    and visualizing the TSP graph.
    """

    def __init__(self, num_cities: int, distance_matrix: Optional[np.ndarray] = None, seed: int = 42):
        """Initialize the TSP instance.

        Args:
            num_cities: The number of cities in the TSP.
            distance_matrix: Optional symmetric distance matrix. If None, one is generated randomly.
            seed: Random seed for distance matrix generation.
        """
        self.num_cities = num_cities
        if distance_matrix is not None:
            self.distance_matrix = distance_matrix
        else:
            np.random.seed(seed)
            # Generate random coordinates for cities
            coords = np.random.rand(num_cities, 2)
            self.coords = coords
            # Compute Euclidean distances
            dist_matrix = np.zeros((num_cities, num_cities))
            for i in range(num_cities):
                for j in range(i + 1, num_cities):
                    dist = np.linalg.norm(coords[i] - coords[j])
                    dist_matrix[i, j] = dist_matrix[j, i] = dist
            self.distance_matrix = dist_matrix

        # Generate all possible tours (fixing city 0 as the starting point)
        self.all_tours = self._generate_all_tours()

    def _generate_all_tours(self) -> List[Tuple[int, ...]]:
        """Generate all possible tours fixing city 0 as the start/end point."""
        other_cities = list(range(1, self.num_cities))
        perms = sorted(list(permutations(other_cities)))
        tours = [tuple([0] + list(p)) for p in perms]
        return tours

    def calculate_cost(self, tour: Tuple[int, ...]) -> float:
        """Calculate the total distance of a given tour.

        Args:
            tour: A tuple of city indices representing the tour.

        Returns:
            The total path distance.
        """
        cost = 0.0
        for i in range(len(tour) - 1):
            cost += self.distance_matrix[tour[i], tour[i + 1]]
        # Return to start
        cost += self.distance_matrix[tour[-1], tour[0]]
        return cost

    def get_optimal_tour(self) -> Tuple[Tuple[int, ...], float]:
        """Find the optimal tour and its cost using brute force.

        Returns:
            A tuple of (optimal_tour, minimum_cost).
        """
        min_cost = float("inf")
        best_tour = None
        for tour in self.all_tours:
            cost = self.calculate_cost(tour)
            if cost < min_cost:
                min_cost = cost
                best_tour = tour
        return best_tour, min_cost

    def get_feasible_states(self) -> List[Tuple[int, ...]]:
        """Return the list of all feasible tour states (constrained subset)."""
        return self.all_tours

    def visualize(self, tour: Optional[Tuple[int, ...]] = None, title: str = "TSP Instance"):
        """Visualize the TSP instance as a graph.

        Args:
            tour: Optional tour to highlight with edges.
            title: Title for the plot.
        """
        G = nx.Graph()
        # Add nodes
        for i in range(self.num_cities):
            pos = self.coords[i] if hasattr(self, "coords") else (np.cos(2*np.pi*i/self.num_cities), np.sin(2*np.pi*i/self.num_cities))
            G.add_node(i, pos=pos)

        pos = nx.get_node_attributes(G, "pos")
        
        plt.figure(figsize=(8, 6))
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color="skyblue")
        nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")

        # Draw all edges with low alpha
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                nx.draw_networkx_edges(G, pos, edgelist=[(i, j)], width=1.0, alpha=0.2, edge_color="gray")

        # Highlight tour edges if provided
        if tour:
            tour_edges = [(tour[i], tour[i+1]) for i in range(len(tour)-1)]
            tour_edges.append((tour[-1], tour[0]))
            nx.draw_networkx_edges(G, pos, edgelist=tour_edges, width=2.5, alpha=0.8, edge_color="orange")

        plt.title(title)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    # Quick test
    tsp = TSPInstance(num_cities=3)
    best_tour, min_cost = tsp.get_optimal_tour()
    print(f"Number of possible tours: {len(tsp.all_tours)}")
    print(f"Optimal tour: {best_tour} with cost {min_cost:.4f}")
    # tsp.visualize(tour=best_tour)
