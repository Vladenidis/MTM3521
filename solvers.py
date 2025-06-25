import numpy as np

def euler_method(f, t_span, y0, h):
    t_start, t_end = t_span
    num_steps = int((t_end - t_start) / h)
    t_values = np.linspace(t_start, t_end, num_steps + 1)
    y_values = np.zeros(num_steps + 1)
    y_values[0] = y0

    for i in range(num_steps):
        y_values[i+1] = y_values[i] + h * f(t_values[i], y_values[i])

    return t_values, y_values

def linear_interpolation(x_points, y_points, x_new):
    for i in range(len(x_points) - 1):
        if x_points[i] <= x_new <= x_points[i+1]:
            x0, y0 = x_points[i], y_points[i]
            x1, y1 = x_points[i+1], y_points[i+1]
            
            y_new = y0 + (y1 - y0) * (x_new - x0) / (x1 - x0)
            return y_new
    raise ValueError("x_new is outside the range of x_points")