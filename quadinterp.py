import numpy as np
import matplotlib.pyplot as plt

# Define the range and function
x_min, x_max = -1, 1
f = np.exp

# Generate 10 equally spaced nodes in the range [-1, 1]
nodes = np.linspace(x_min, x_max, 10)
values = f(nodes)  # Function values at the nodes

# Define a piecewise quadratic interpolation function
def piecewise_quadratic_interpolation(x, nodes, values):
    
    n = len(nodes)
    result = np.zeros_like(x, dtype=float)
    
    for i in range(n - 2):  # Create quadratic polynomials for each pair of consecutive nodes
        # Extract the current segment's nodes and values
        x0, x1, x2 = nodes[i], nodes[i + 1], nodes[i + 2]
        y0, y1, y2 = values[i], values[i + 1], values[i + 2]
        
        # Define the coefficients of the quadratic polynomial
        a = (y2 - (x2 - x0) / (x1 - x0) * y1 + (x2 - x1) / (x1 - x0) * y0) / ((x2 - x0) * (x2 - x1))
        b = (y1 - y0) / (x1 - x0) - a * (x1 + x0)
        c = y0 - a * x0**2 - b * x0
        
        # Evaluate the polynomial for x in [x0, x2]
        seg = (x >= x0) & (x <= x2)
        result[seg] = a * x[seg]**2 + b * x[seg] + c
    
    return result

# Points for evaluation
x_eval = np.linspace(x_min, x_max, 500)

# Perform the piecewise quadratic interpolation
interpolated_values = piecewise_quadratic_interpolation(x_eval, nodes, values)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x_eval, np.exp(x_eval), label='e^x (Original)', color='blue')
plt.plot(x_eval, interpolated_values, label='Piecewise Quadratic Interpolation', color='red', linestyle='--')
plt.scatter(nodes, values, color='black', label='Interpolation Nodes')
plt.legend()
plt.title('Piecewise Quadratic Interpolation of $e^x$')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()
