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
from des.utils.des_plotter import DESPlotter

import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning, module="opfunu")


def run_optimization_example():
    """Run a simple optimization example."""

    dimensions = 50

    # opt_func = CEC17Function(dimensions=dimensions, function_id=3)
    opt_func = Sphere(dimensions=dimensions)

    # Define bounds
    lower_bounds = -50.12
    upper_bounds = 50.12

    initial_point_generator = InitialPointGenerator(
        strategy=InitialPointGeneratorType.UNIFORM,
        dimensions=dimensions,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    initial_point = initial_point_generator.generate()

    # Create a configuration object with custom settings
    config = DESConfig(dimensions=dimensions)

    # Set core parameters
    config.budget = 10000 * dimensions
    config.population_size = 4 * dimensions

    # Enable diagnostics for visualization
    config.with_convergence_diagnostics()
    config.enable_all_diagnostics()

    print("Starting DES optimization...")
    print(f"Dimensions: {dimensions}")
    print(f"Budget: {config.budget}")
    print(f"Population size: {config.population_size}")
    print(f"Initial point value: {opt_func(initial_point):.20f}")

    # print(f"Inital point {initial_point}")
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

    plotter = DESPlotter(log_data=result.diagnostic)
    plotter.plot_convergence_curve()
    for i in range(100):
        plotter.plot_population_2d(i)


if __name__ == "__main__":
    run_optimization_example()
