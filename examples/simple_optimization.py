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

    # Initial point (center of search space)
    # initial_point = np.zeros(dimensions)
    initial_point = np.full(dimensions, 3.0, dtype=float)

    # Define bounds
    lower_bounds = -5.12
    upper_bounds = 5.12

    # Create a configuration object with custom settings
    config = DESConfig(dimensions=dimensions)
    # Set core parameters
    config.budget = 80000
    config.population_size = 4 * dimensions
    config.tol = 1e-12

    # Enable diagnostics for visualization
    config.with_convergence_diagnostics()

    # Alternative fluent API:
    # config = DESConfig().with_custom_budget(100000).with_population_size(4 * dimensions).with_tolerance(1e-8).with_convergence_diagnostics()

    print("Starting DES optimization on Rastrigin function...")
    print(f"Dimensions: {dimensions}")
    print(f"Budget: {config.budget}")
    print(f"Population size: {config.population_size}")
    print(f"Tolerance: {config.tol}")

    # Create and run optimizer
    optimizer = DESOptimizer(
        func=Sphere(dimensions=dimensions),
        initial_point=initial_point,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        config=config,
        boundary_strategy=BoundaryHandlerType.BOUNCE_BACK,
    )

    result = optimizer.optimize()

    # Print results
    print("\nOptimization completed:")
    print(f"Best fitness: {result.best_fitness}")
    print(f"Function evaluations: {result.evaluations}")
    print(
        f"Convergence status: {'Converged' if result.convergence == 0 else 'Maximum iterations reached'}"
    )
    print(f"Message: {result.message}")

    # Plot convergence curve if diagnostics were enabled
    if "bestVal" in result.diagnostic:
        plt.figure(figsize=(10, 6))
        plt.semilogy(result.diagnostic["bestVal"])
        plt.grid(True)
        plt.xlabel("Iteration")
        plt.ylabel("Best Fitness (log scale)")
        plt.title("Convergence Curve")
        plt.tight_layout()
        plt.savefig("convergence_curve.png")


if __name__ == "__main__":
    run_optimization_example()
