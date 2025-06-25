import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def solve_problem_1():
    t = sp.Symbol('t')
    
    f_expr = (sp.Symbol('y') + 1) / 3
    t0 = 0
    y0 = 0

    y_k = [sp.Integer(y0)]
    for k in range(1, 7):
        integral = sp.integrate(f_expr.subs(sp.Symbol('y'), y_k[k-1]), (t, t0, t))
        y_k.append(y0 + integral)

    print("Aproximações de Picard:")
    for i, y in enumerate(y_k):
        print(f"y_{i}(t) = {y}")

    y_true_expr = sp.exp(t/3) - 1
    print(f"\nSolução Verdadeira: y(t) = {y_true_expr}")

    y_2_func = sp.lambdify(t, y_k[2], 'numpy')
    y_4_func = sp.lambdify(t, y_k[4], 'numpy')
    y_6_func = sp.lambdify(t, y_k[6], 'numpy')
    y_true_func = sp.lambdify(t, y_true_expr, 'numpy')

    t_vals = np.linspace(0, 2, 200)
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, y_true_func(t_vals), 'r-', linewidth=2, label='Solução Verdadeira: $e^{t/3}-1$')
    plt.plot(t_vals, y_2_func(t_vals), 'g--', label='Aproximação de Picard $y_2(t)$')
    plt.plot(t_vals, y_4_func(t_vals), 'b-.', label='Aproximação de Picard $y_4(t)$')
    plt.plot(t_vals, y_6_func(t_vals), 'm:', label='Aproximação de Picard $y_6(t)$')
    
    plt.title("Aproximações de Picard vs. Solução Verdadeira")
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    solve_problem_1()