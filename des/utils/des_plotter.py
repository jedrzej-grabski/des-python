import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numpy.typing import NDArray

from des.logging.loggers import DiagnosticLogger, LogData


class DESPlotter:
    """
    Plotter class for visualizing DES optimization diagnostics.
    """

    def __init__(
        self,
        log_data: LogData,
        output_dir: str = "graphs",
        figsize: tuple[int, int] = (10, 6),
        dpi: int = 150,
    ) -> None:
        """
        Initialize the DES plotter.

        Args:
            log_data: LogData object or DiagnosticLogger containing diagnostic information
            output_dir: Directory to save plots
            figsize: Figure size (width, height)
            dpi: DPI for saved figures
        """

        self.log_data = log_data

        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = dpi

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_convergence_curve(
        self,
        save: bool = True,
        filename: str = "convergence_curve.png",
        log_scale: bool = True,
        show_grid: bool = True,
        title: str | None = None,
    ) -> Figure | None:
        """
        Plot the convergence curve showing best fitness over iterations.

        Args:
            save: Whether to save the plot
            filename: Filename for saved plot
            log_scale: Whether to use log scale for y-axis
            show_grid: Whether to show grid
            title: Custom title for the plot

        Returns:
            Figure object if plot was created, None otherwise
        """
        if not self.log_data.bestVal:
            print("No best fitness data available for convergence plot")
            return None

        fig, ax = plt.subplots(figsize=self.figsize)

        if log_scale:
            ax.semilogy(self.log_data.bestVal)
            ylabel = "Best Fitness (log scale)"
        else:
            ax.plot(self.log_data.bestVal)
            ylabel = "Best Fitness"

        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(title or "Convergence Curve")

        if show_grid:
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(self.output_dir, filename), dpi=self.dpi)

        return fig

    def plot_fitness_statistics(
        self,
        save: bool = True,
        filename: str = "fitness_statistics.png",
        log_scale: bool = True,
        show_grid: bool = True,
        title: str | None = None,
    ) -> Figure | None:
        """
        Plot fitness statistics (best, worst, mean) over iterations.

        Args:
            save: Whether to save the plot
            filename: Filename for saved plot
            log_scale: Whether to use log scale for y-axis
            show_grid: Whether to show grid
            title: Custom title for the plot

        Returns:
            Figure object if plot was created, None otherwise
        """
        data_available = any(
            [self.log_data.bestVal, self.log_data.worstVal, self.log_data.mean]
        )

        if not data_available:
            print("No fitness statistics data available")
            return None

        fig, ax = plt.subplots(figsize=self.figsize)

        plot_func = ax.semilogy if log_scale else ax.plot
        ylabel_suffix = " (log scale)" if log_scale else ""

        if self.log_data.bestVal:
            plot_func(self.log_data.bestVal, label="Best Fitness", linewidth=2)

        if self.log_data.worstVal:
            plot_func(self.log_data.worstVal, label="Worst Fitness", linewidth=2)

        if self.log_data.mean:
            plot_func(self.log_data.mean, label="Mean Fitness", linewidth=2)

        ax.set_xlabel("Iteration")
        ax.set_ylabel(f"Fitness{ylabel_suffix}")
        ax.set_title(title or "Fitness Statistics")
        ax.legend()

        if show_grid:
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(self.output_dir, filename), dpi=self.dpi)

        return fig

    def plot_population_diversity(
        self,
        save: bool = True,
        filename: str = "population_diversity.png",
        show_grid: bool = True,
        title: str | None = None,
    ) -> Figure | None:
        """
        Plot population diversity using eigenvalues of covariance matrix.

        Args:
            save: Whether to save the plot
            filename: Filename for saved plot
            show_grid: Whether to show grid
            title: Custom title for the plot

        Returns:
            Figure object if plot was created, None otherwise
        """
        if not self.log_data.eigen:
            print("No eigenvalue data available for diversity plot")
            return None

        fig, ax = plt.subplots(figsize=self.figsize)

        # Convert eigenvalue data to array
        eigen_array = np.array(self.log_data.eigen)

        # Plot each eigenvalue as a separate line
        for i in range(eigen_array.shape[1]):
            ax.semilogy(eigen_array[:, i], label=f"Eigenvalue {i+1}")

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Eigenvalues (log scale)")
        ax.set_title(title or "Population Diversity (Eigenvalues)")
        ax.legend()

        if show_grid:
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(self.output_dir, filename), dpi=self.dpi)

        return fig

    def plot_ft_evolution(
        self,
        save: bool = True,
        filename: str = "ft_evolution.png",
        show_grid: bool = True,
        title: str | None = None,
    ) -> Figure | None:
        """
        Plot Ft parameter evolution over iterations.

        Args:
            save: Whether to save the plot
            filename: Filename for saved plot
            show_grid: Whether to show grid
            title: Custom title for the plot

        Returns:
            Figure object if plot was created, None otherwise
        """
        if not self.log_data.Ft:
            print("No Ft data available")
            return None

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(self.log_data.Ft, linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Ft Value")
        ax.set_title(title or "Ft Parameter Evolution")

        if show_grid:
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(self.output_dir, filename), dpi=self.dpi)

        return fig

    def plot_population_2d(
        self,
        iteration: int = -1,
        dimensions: tuple[int, int] = (0, 1),
        save: bool = True,
        filename_prefix: str = "population_2d",
        show_mean: bool = True,
        show_grid: bool = True,
        title: str | None = None,
        show_function: bool = True,
        contour_levels: int = 20,
        function_alpha: float = 0.6,
    ) -> Figure | None:
        """
        Plot 2D visualization of population at a specific iteration.

        Args:
            iteration: Iteration to plot (-1 for last iteration)
            dimensions: Tuple of dimension indices to plot
            save: Whether to save the plot
            filename_prefix: Filename prefix for saved plot
            show_mean: Whether to show mean point
            show_grid: Whether to show grid
            title: Custom title for the plot
            show_function: Whether to show the optimization function as contours
            contour_levels: Number of contour levels for function visualization
            function_alpha: Alpha (transparency) for function contours

        Returns:
            Figure object if plot was created, None otherwise
        """
        if not self.log_data.pop:
            print("No population data available")
            return None

        if iteration == -1:
            iteration = len(self.log_data.pop) - 1

        if iteration >= len(self.log_data.pop):
            print(f"Iteration {iteration} not available")
            return None

        filename = filename_prefix + f"_{iteration}.png"

        population = self.log_data.pop[iteration]

        dim1, dim2 = dimensions

        if dim1 >= population.shape[1] or dim2 >= population.shape[1]:
            print(f"Dimensions {dimensions} not available in population data")
            return None

        fig, ax = plt.subplots(figsize=self.figsize)

        if show_function and self.log_data.opt_func and self.log_data.bounds:
            lower_bounds, upper_bounds = self.log_data.bounds

            # Create a grid for the contour plot
            x_min, x_max = lower_bounds[dim1], upper_bounds[dim1]
            y_min, y_max = lower_bounds[dim2], upper_bounds[dim2]

            # Extend the grid slightly beyond bounds for better visualization
            margin_x = (x_max - x_min) * 0.1
            margin_y = (y_max - y_min) * 0.1

            x_grid = np.linspace(x_min - margin_x, x_max + margin_x, 100)
            y_grid = np.linspace(y_min - margin_y, y_max + margin_y, 100)
            X, Y = np.meshgrid(x_grid, y_grid)

            # Create points for function evaluation
            grid_points = np.zeros((X.size, population.shape[1]))

            # Set the two dimensions we're plotting
            grid_points[:, dim1] = X.ravel()
            grid_points[:, dim2] = Y.ravel()

            # For other dimensions, use the mean from the current population
            for i in range(population.shape[1]):
                if i != dim1 and i != dim2:
                    grid_points[:, i] = np.mean(population[:, i])

            # Evaluate the function
            Z_values = np.array(
                [self.log_data.opt_func(point) for point in grid_points]
            )
            Z = Z_values.reshape(X.shape)

            # Create contour plot
            contour = ax.contour(
                X,
                Y,
                Z,
                levels=contour_levels,
                alpha=function_alpha,
                colors="gray",
                linewidths=0.5,
            )
            contourf = ax.contourf(
                X,
                Y,
                Z,
                levels=contour_levels,
                alpha=function_alpha / 2,
                cmap="viridis",
            )

            # Add colorbar
            plt.colorbar(contourf, ax=ax, label="Function Value")

            # except Exception as e:
            # print(f"Warning: Could not plot function contours: {e}")

        # Plot population points
        ax.scatter(
            population[:, dim1],
            population[:, dim2],
            alpha=0.8,
            s=50,
            label="Population",
            color="red",
            edgecolor="darkred",
            zorder=5,
        )

        # Plot mean point if available and requested
        if (
            show_mean
            and self.log_data.meanCords
            and iteration < len(self.log_data.meanCords)
        ):
            mean_coords = self.log_data.meanCords[iteration]
            ax.scatter(
                mean_coords[dim1],
                mean_coords[dim2],
                color="blue",
                s=150,
                marker="x",
                linewidth=4,
                label="Mean",
                zorder=6,
            )

        ax.set_xlabel(f"Dimension {dim1}")
        ax.set_ylabel(f"Dimension {dim2}")
        ax.set_title(title or f"Population at Iteration {iteration}")
        ax.legend()

        if show_grid:
            ax.grid(True, alpha=0.3, zorder=1)

        if save:
            plt.savefig(os.path.join(self.output_dir, filename), dpi=self.dpi)

        plt.close()

        return fig

    def plot_all_diagnostics(
        self,
        save: bool = True,
        filename_prefix: str = "diagnostic",
        log_scale: bool = True,
    ) -> list[Figure]:
        """
        Generate all available diagnostic plots.

        Args:
            save: Whether to save the plots
            filename_prefix: Prefix for saved plot filenames
            log_scale: Whether to use log scale where applicable

        Returns:
            List of created Figure objects
        """
        figures = []

        # Convergence curve
        fig = self.plot_convergence_curve(
            save=save,
            filename=f"{filename_prefix}_convergence.png",
            log_scale=log_scale,
        )
        if fig:
            figures.append(fig)

        # Fitness statistics
        fig = self.plot_fitness_statistics(
            save=save,
            filename=f"{filename_prefix}_fitness_stats.png",
            log_scale=log_scale,
        )
        if fig:
            figures.append(fig)

        # Population diversity
        fig = self.plot_population_diversity(
            save=save, filename=f"{filename_prefix}_diversity.png"
        )
        if fig:
            figures.append(fig)

        # Ft evolution
        fig = self.plot_ft_evolution(
            save=save, filename=f"{filename_prefix}_ft_evolution.png"
        )
        if fig:
            figures.append(fig)

        # 2D population (if available)
        fig = self.plot_population_2d(
            save=save, filename=f"{filename_prefix}_population_2d.png"
        )
        if fig:
            figures.append(fig)

        return figures
