import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline

# Define the true function
def true_function(x):
    return 1/(1+x*x)

# Generate data points and interpolate
def interpolate_and_plot(n_points):
    # Generate evenly spaced nodes and true function values
    x_nodes = np.linspace(-1, 1, n_points)
    y_nodes = true_function(x_nodes)
    
    # Dense points for comparison
    x_dense = np.linspace(-1, 1, 1000)
    y_true_dense = true_function(x_dense)
    
    # Linear interpolation
    linear_interp = interp1d(x_nodes, y_nodes, kind='linear')
    y_linear_dense = linear_interp(x_dense)
    
    # Quadratic interpolation (2nd degree spline with `kind='quadratic'`)
    quadratic_interp = interp1d(x_nodes, y_nodes, kind='quadratic')
    y_quadratic_dense = quadratic_interp(x_dense)
    
    # Cubic spline interpolation
    cubic_spline = CubicSpline(x_nodes, y_nodes)
    y_cubic_dense = cubic_spline(x_dense)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(x_dense, y_true_dense, 'k-', label='True Function $e^x$', linewidth=2)
    plt.plot(x_dense, y_linear_dense, 'r--', label='Linear Interpolation')
    plt.plot(x_dense, y_quadratic_dense, 'g-.', label='Quadratic Interpolation')
    plt.plot(x_dense, y_cubic_dense, 'b-', label='Cubic Spline Interpolation')
    plt.plot(x_nodes, y_nodes, 'ko', label='Data Points')
    plt.title(f'Interpolation Comparison with n={n_points} Points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot the absolute errors
    plt.figure(figsize=(10, 6))
    plt.plot(x_dense, np.abs(y_linear_dense-y_true_dense), 'r--', label='Linear Error')
    plt.plot(x_dense, np.abs(y_quadratic_dense- y_true_dense), 'g-.', label='Quadratic Error')
    plt.plot(x_dense, np.abs(y_cubic_dense - y_true_dense), 'b-', label='Cubic Spline Error')
    plt.title(f'Error Analysis with n={n_points} Points')
    plt.xlabel('x')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()

# Interpolate and plot for n=10, 20, 40 points
for n in [10, 20, 40]:
    interpolate_and_plot(n)
