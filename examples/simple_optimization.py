import numpy as np
import matplotlib.pyplot as plt
from des import DESOptimizer
from des.config import DESConfig
from des.utils.boundary_handlers import BoundaryHandlerType
from des.utils.benchmark_functions import Sphere, Rastrigin, Rosenbrock, CEC17Function
from des.utils.initial_point_generator import (
    InitialPointGenerator,
    InitialPointGeneratorType,
)

import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning, module="opfunu")


def run_optimization_example():
    """Run a simple optimization example."""

    dimensions = 8

    # opt_func = CEC17Function(dimensions=dimensions, function_id=3)
    opt_func = Sphere(dimensions=dimensions)

    # Define bounds
    lower_bounds = -5.12
    upper_bounds = 5.12

    initial_point_generator = InitialPointGenerator(
        strategy=InitialPointGeneratorType.NORMAL,
        dimensions=dimensions,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    initial_point = initial_point_generator.generate()
    # Create a configuration object with custom settings
    config = DESConfig(dimensions=dimensions)

    # Set core parameters
    config.budget = 1000 * dimensions
    config.population_size = 4 * dimensions

    # Enable diagnostics for visualization
    config.with_convergence_diagnostics()
    config.enable_all_diagnostics()

    print("Starting DES optimization...")
    print(f"Dimensions: {dimensions}")
    print(f"Budget: {config.budget}")
    print(f"Population size: {config.population_size}")
    print(f"Initial point value: {opt_func(initial_point):.20f}")

    # Create and run optimizer
    optimizer = DESOptimizer(
        func=opt_func,
        initial_point=initial_point,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        config=config,
        boundary_strategy=BoundaryHandlerType.CLAMP,
    )

    result = optimizer.optimize()

    # Print results
    print("\nOptimization completed:")
    print(f"Best fitness: {result.best_fitness:.20f}")
    print(f"Function evaluations: {result.evaluations}")
    print(f"Message: {result.message}")
    # print(len(result.diagnostic.bestVal))
    # print(result.diagnostic.bestVal)
    # print(result.diagnostic.mean[:200])
    # Plot convergence curve if diagnostics were enabled
    if result.diagnostic.bestVal is not None:
        plt.figure(figsize=(10, 6))
        plt.semilogy(result.diagnostic.bestVal)
        plt.grid(True)
        plt.xlabel("Iteration")
        plt.ylabel("Best Fitness (log scale)")
        plt.title("Convergence Curve")
        plt.tight_layout()
        plt.savefig("graphs/convergence_curve.png")

    if result.diagnostic.mean is not None:
        plt.figure(figsize=(10, 6))
        plt.semilogy(result.diagnostic.mean)
        plt.grid(True)
        plt.xlabel("Iteration")
        plt.ylabel("Mean fitness value (log scale)")
        plt.title("Mean fitness")
        plt.tight_layout()
        plt.savefig("graphs/fitness_curve.png")


if __name__ == "__main__":
    run_optimization_example()
