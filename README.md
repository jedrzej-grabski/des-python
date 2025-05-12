### des-python

# DES Algorithm

A Python implementation of the Differential Evolution Strategy algorithm.

## Overview

This package provides an object-oriented implementation of the DES optimization algorithm, which is a variant of Differential Evolution with adaptive parameter control.

## Installation

```bash
pip install -e .
```

## Usage

```python
import numpy as np
from des.des_optimizer import DESOptimizer

# Define objective function
def sphere_function(x):
    return np.sum(x**2)

# Initialize optimizer
optimizer = DESOptimizer(
    func=sphere_function,
    initial_point=np.zeros(10),  # 10-dimensional problem
    lower_bounds=-10,            # All dimensions have same lower bound
    upper_bounds=10,             # All dimensions have same upper bound
    budget=10000,                # Maximum number of function evaluations
)

# Run optimization
result = optimizer.optimize()

# Print results
print(f"Best solution: {result.best_solution}")
print(f"Best fitness: {result.best_fitness}")
print(f"Function evaluations: {result.evaluations}")
print(f"Optimization status: {result.message}")
```

## Features

- Object-oriented design for easy extension
- Adaptive parameter control based on the history of successful steps
- Configurable constraints handling
- Detailed logging and diagnostics
- Ready-to-use examples and benchmark functions

## License

MIT
