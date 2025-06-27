import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import the factory and register algorithms
from src import AlgorithmFactory
import src.algorithms  # This triggers algorithm registration

# Import specific components directly
from src.des.config import DESConfig
from src.utils.boundary_handlers import BoundaryHandlerType
from src.utils.benchmark_functions import Sphere
from src.utils.initial_point_generator import (
    InitialPointGenerator,
    InitialPointGeneratorType,
)
from src.plotting.multi_algorithm_plotter import MultiAlgorithmPlotter

import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning, module="opfunu")

# Set matplotlib to non-interactive backend for headless environments
plt.ioff()  # Turn off interactive mode
plt.switch_backend("Agg")  # Use non-interactive backend


def run_optimization_example():
    """Run a simple optimization example using the new architecture."""

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

    # Create optimizer using the factory pattern
    optimizer = AlgorithmFactory.create_optimizer(
        algorithm_name="DES",
        func=opt_func,
        initial_point=initial_point,
        config=config,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        boundary_strategy=BoundaryHandlerType.CLAMP,
    )

    # Run optimization
    result = optimizer.optimize()

    # Print results
    print("\nOptimization completed:")
    print(f"Best fitness: {result.best_fitness:.20f}")
    print(f"Function evaluations: {result.evaluations}")
    print(f"Message: {result.message}")
    print(f"Algorithm: {result.algorithm_name}")

    # For DES-specific attributes, you can safely access them like this:
    if hasattr(result.diagnostic, "Ft"):
        des_logs = result.diagnostic  # Type: BaseLogData, but has Ft attribute
        if des_logs.Ft:  # type: ignore  # or use cast
            print(f"Final Ft value: {des_logs.Ft[-1]}")

    # Create output directory for plots
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    print(f"\nSaving plots to: {output_dir.absolute()}")

    # Create plotter using the new multi-algorithm plotter
    plotter = MultiAlgorithmPlotter()

    # Plot algorithm-specific metrics (automatically detects DES)
    metrics_path = output_dir / "des_metrics.png"
    fig_metrics = plotter.plot_algorithm_specific_metrics(
        result, "DES", save_path=metrics_path
    )
    plt.close(fig_metrics)  # Close figure to free memory
    print(f"Saved DES metrics plot to: {metrics_path}")

    # Plot convergence curve
    results_dict = {"DES": result}
    convergence_path = output_dir / "convergence_comparison.png"
    fig_convergence = plotter.plot_convergence_comparison(
        results_dict,
        save_path=convergence_path,
        title="DES Convergence on Sphere Function",
    )
    plt.close(fig_convergence)  # Close figure to free memory
    print(f"Saved convergence plot to: {convergence_path}")

    # Plot Ft parameter evolution (DES-specific) - type-safe check
    if hasattr(result.diagnostic, "Ft") and getattr(result.diagnostic, "Ft"):
        ft_path = output_dir / "ft_evolution.png"
        fig_ft = plotter.plot_parameter_evolution(
            results_dict, "Ft", save_path=ft_path, title="DES Ft Parameter Evolution"
        )
        plt.close(fig_ft)  # Close figure to free memory
        print(f"Saved Ft evolution plot to: {ft_path}")

    # Optional: Create population evolution plots for 2D visualization
    if dimensions == 2:
        population_path = output_dir / "population_evolution_2d.png"
        plot_population_evolution_2d(result.diagnostic, save_path=population_path)
        print(f"Saved population evolution plot to: {population_path}")

    print(f"\nAll plots saved successfully to: {output_dir.absolute()}")


def plot_population_evolution_2d(log_data, save_path=None):
    """Plot population evolution for 2D problems."""
    if not (hasattr(log_data, "population") and log_data.population):
        print("No population data available for 2D plotting")
        return

    # Plot every 10th iteration to avoid too many plots
    iterations_to_plot = range(
        0, len(log_data.population), max(1, len(log_data.population) // 10)
    )

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, iteration in enumerate(iterations_to_plot[:10]):
        if i >= len(axes):
            break

        population = log_data.population[iteration]
        best_fitness = (
            log_data.best_fitness[iteration] if log_data.best_fitness else None
        )

        axes[i].scatter(population[:, 0], population[:, 1], alpha=0.6, s=20)
        axes[i].set_title(
            f"Iteration {iteration}\nBest: {best_fitness:.2e}"
            if best_fitness
            else f"Iteration {iteration}"
        )
        axes[i].set_xlabel("x1")
        axes[i].set_ylabel("x2")
        axes[i].grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(iterations_to_plot), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Population Evolution (2D)", fontsize=16)
    plt.tight_layout()

    # Save the plot instead of showing it
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved population evolution plot to: {save_path}")

    plt.close(fig)  # Close figure to free memory


if __name__ == "__main__":
    run_optimization_example()
