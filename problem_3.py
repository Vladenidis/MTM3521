import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from solvers import euler_method, linear_interpolation

f = lambda t, y: (2/t)*y + t**2 * np.exp(t)
y_true_func = lambda t: t**2 * (np.exp(t) - np.e)
t_span = (1, 2)
y0 = 0

def solve_part_a():
    print("(a)")
    h = 0.25
    t_approx, y_approx = euler_method(f, t_span, y0, h)
    y_true_vals = y_true_func(t_approx)
    errors = np.abs(y_true_vals - y_approx)
    
    results = pd.DataFrame({
        "t_i": t_approx,
        "y_i (Euler)": y_approx,
        "y(t_i) (Verdadeiro)": y_true_vals,
        "|y(t_i) - y_i|": errors
    })
    print(results)
    return t_approx, y_approx

def solve_part_b(t_approx, y_approx):
    print("\n(b)")
    interp_points = [1.04, 1.55, 1.97]
    for t_new in interp_points:
        y_interp = linear_interpolation(t_approx, y_approx, t_new)
        y_true_val = y_true_func(t_new)
        error = np.abs(y_true_val - y_interp)
        print(f"t = {t_new}:")
        print(f"  y Interpolado  = {y_interp:.6f}")
        print(f"  y Verdadeiro   = {y_true_val:.6f}")
        print(f"  Erro           = {error:.6f}")

def solve_part_c():
    print("\n(c)")

    L = 2
    
    t = 2
    M = 2*(np.exp(t) - np.e) + 4*t*np.exp(t) + t**2*np.exp(t)
    
    b, a = t_span[1], t_span[0]
    error_target = 0.1
    
    h_required = error_target * 2 * L / (M * (np.exp(L * (b-a)) - 1))
    print(f"Para garantir erro <= {error_target}, tamanho do passo necessário é h <= {h_required:.6f}")

def solve_convergence_analysis():
    plt.figure(figsize=(10, 6))
    
    t_true = np.linspace(t_span[0], t_span[1], 500)
    plt.plot(t_true, y_true_func(t_true), 'r-', linewidth=2.5, label='Solução Verdadeira')


    for j in [1, 2, 3]:
        N = 10**j
        h = (t_span[1] - t_span[0]) / N
        t_approx, y_approx = euler_method(f, t_span, y0, h)
        plt.plot(t_approx, y_approx, '--', label=f"Aproximação de Euler (N={N})")

    plt.title("Convergência do método de Euler quando N cresce")
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    t_approx_a, y_approx_a = solve_part_a()
    solve_part_b(t_approx_a, y_approx_a)
    solve_part_c()
    solve_convergence_analysis()