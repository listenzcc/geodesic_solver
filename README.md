# Geodesic Solver on Parametric Surfaces (Perlin Noise Demo)

This project implements a Python tool to calculate and visualize **geodesics**—the shortest paths—on curved 2D surfaces embedded in 3D space.

The demonstration uses a procedurally generated terrain (Perlin noise) as the surface defined by . It applies concepts from Riemannian geometry to find the path of minimum distance between two points, comparing it against a naive "straight line" projection (gradient descent-style approach).

*(Note: The above image is a placeholder representation of the script's output. Running the script will generate a similar interactive matplotlib figure.)*

## Overview

In Euclidean space, the shortest path between two points is a straight line. On a curved surface, like a hill or a valley, the shortest path is called a **geodesic**.

A geodesic path balances two competing factors:

1. Minimizing the direct Euclidean distance between points.
2. Avoiding areas with steep gradients where "movement costs" (due to change in elevation) are high.

This codebase provides a robust `PerlinSurfaceGeodesic` class that calculates the necessary differential geometry components—specifically the metric tensor and Christoffel symbols—to define movement on any surface defined by a function  and its gradient.

## Features

* **Differential Geometry Engine**: Calculates the Riemannian Metric Tensor () and Christoffel Symbols () for any differentiable parametric surface.
* **Multiple Geodesic Solvers**: Implements three distinct numerical methods for finding geodesics:
1. **Variational/Energy Minimization (Default, Fast):** Treats the path as a spline and optimizes control points to minimize total path energy. Stable and fast.
2. **Shooting Method (ODE Solver):** Solves the second-order geodesic differential equation as an initial value problem, optimizing the initial firing angle to hit the target.
3. **Dijkstra on Riemannian Grid:** Discretizes the surface and finds the global optimum on the grid using graph search with edge weights derived from the local metric tensor.


* **Procedural Terrain Generation**: Includes customizable Perlin-style noise generators to create interesting test surfaces.
* **Comparison Baseline**: Includes a naive "Gradient Descent" pathfinder that tries to walk directly toward the target projected on the XY plane, serving as a baseline to show how much shorter real geodesics are.
* **Rich Visualization**: Generates a 3-panel matplotlib figure showing the 3D view, top-down contour view, and an elevation profile comparison.

## Mathematical Background

The core idea is that distance on a curved surface is measured differently than on a flat plane. The squared infinitesimal distance  on the surface  is given by:



This defines the **Metric Tensor** .

A particle traveling along a geodesic experiences zero acceleration *tangent to the surface*. This leads to the **Geodesic Differential Equation**, which the code solves numerically:



Where  are the **Christoffel symbols**, derived from derivatives of the metric tensor, representing the "curvature forces" acting on the path.

## Requirements

* Python 3.x
* `numpy`
* `scipy`
* `matplotlib`
* `tqdm` (for progress bars)

Install dependencies via pip:

```bash
pip install numpy scipy matplotlib tqdm

```

## Usage

The script is self-contained. Simply run it with Python:

```bash
python your_script_name.py

```

By default, it will:

1. Generate a random Perlin noise surface.
2. Define start point A near (0.2, 0.2) and end point B near (0.8, 0.8).
3. Calculate the geodesic using the **fast variational method** (`geodesic_on_perlin_fast`).
4. Calculate a comparison path using the naive gradient descent approach.
5. Print path length statistics to the console.
6. Display the comparison visualization.

### Switching Solvers

To use different solving methods, modify the `if __name__ == "__main__":` block in the code. Comment out the default solver and uncomment the one you wish to use:

```python
    # --- In the main block ---

    # 1. Shooting Method (ODE solver - hardest to converge, theoretically most accurate)
    # geodesic_path = geodesic_solver.geodesic_on_perlin(A, B, optimize=True)

    # 2. Variational Method (Fast spline optimization - DEFAULT)
    geodesic_path = geodesic_solver.geodesic_on_perlin_fast(A, B, 100)

    # 3. Dijkstra Graph Search (Grid based global optimum, resolution dependent)
    # geodesic_path = geodesic_solver.geodesic_on_perlin_dijkstra(A, B, grid_size=100)

```

## Custom Surfaces

You can use this solver on *any* surface, not just Perlin noise. You just need to provide the function and its gradient to the `PerlinSurfaceGeodesic` constructor.

```python
# Example: A simple bowl shape z = x^2 + y^2
def bowl_func(x, y):
    return x**2 + y**2

def bowl_grad(x, y):
    # df/dx = 2x, df/dy = 2y
    return 2*x, 2*y

# Initialize solver with custom functions
geodesic_solver = PerlinSurfaceGeodesic(
    perlin_func=bowl_func,
    grad_func=bowl_grad
)

# ... define A and B and solve as usual ...

```
