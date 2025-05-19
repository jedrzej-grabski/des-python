import numpy as np
import matplotlib.pyplot as plt
from des import DESOptimizer
from des.config import DESConfig
from des.utils.boundary_handlers import BoundaryHandlerType
from des.utils.benchmark_functions import (
    Sphere,
    Rastrigin,
)


def run_optimization_example():
    """Run a simple optimization example."""

    # Problem dimension
    dimensions = 20

    # Define bounds
    lower_bounds = -5.12
    upper_bounds = 5.12

    # initial_point = np.zeros(dimensions)
    # initial_point = np.full(dimensions, 3.0, dtype=float)
    initial_point = np.random.uniform(lower_bounds, upper_bounds, size=dimensions)

    # Create a configuration object with custom settings
    config = DESConfig(dimensions=dimensions)
    # Set core parameters
    config.budget = 10000
    config.population_size = 4 * dimensions

    # Enable diagnostics for visualization
    config.with_convergence_diagnostics()

    print("Starting DES optimization...")
    print(f"Dimensions: {dimensions}")
    print(f"Budget: {config.budget}")
    print(f"Population size: {config.population_size}")

    # Create and run optimizer
    optimizer = DESOptimizer(
        func=Rastrigin(dimensions=dimensions),
        initial_point=initial_point,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        config=config,
        boundary_strategy=BoundaryHandlerType.BOUNCE_BACK,
    )

    result = optimizer.optimize()

    # Print results
    print("\nOptimization completed:")
    print(f"Best fitness: {result.best_fitness:.20f}")
    print(f"Function evaluations: {result.evaluations}")
    print(f"Message: {result.message}")

    # Plot convergence curve if diagnostics were enabled
    if result.diagnostic.bestVal is not None:
        plt.figure(figsize=(10, 6))
        plt.semilogy(result.diagnostic.bestVal)
        plt.grid(True)
        plt.xlabel("Iteration")
        plt.ylabel("Best Fitness (log scale)")
        plt.title("Convergence Curve")
        plt.tight_layout()
        plt.savefig("convergence_curve.png")


if __name__ == "__main__":
    run_optimization_example()
