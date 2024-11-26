import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def linear_interpolator(x,x_vals,y_vals):

        # Handle extrapolation (before the first point or after the last point)
        if x < x_vals[0]:
            i = 0  # Extrapolate using the first segment
        elif x > x_vals[-1]:
            i = len(x_vals) - 2  # Extrapolate using the last segment
        else:
            # Find the interval [x_i, x_{i+1}] where x lies
            i = next(j for j in range(len(x_vals) - 1) if x_vals[j] <= x <= x_vals[j + 1])

        # Calculate the slope and intercept for the segment [x_i, x_{i+1}]
        x_i, x_next = x_vals[i], x_vals[i + 1]
        y_i, y_next = y_vals[i], y_vals[i + 1]
        m = (y_next - y_i) / (x_next - x_i)  # Slope
        b = y_i - m * x_i  # Intercept

        # Return the value of the linear function y = m*x + b
        return m*x+b
y=[0]*10
# Plotting the result
x_smooth = np.linspace(-1,1,10)
y_smooth = np.exp(x_smooth)
x=np.linspace(-1,1,10)
for i in range (0,10):
     y[i]=linear_interpolator(x[i],x_smooth,y_smooth)

plt.figure(figsize=(8, 6))
plt.plot(x_smooth,y_smooth, 'o', label='Data points', markersize=8, color='red')
plt.plot(x,y, '-', label='Piecewise linear interpolation', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Piecewise Linear Interpolation')
plt.legend()
plt.grid(True)
plt.show()
