import numpy as np
import matplotlib.pyplot as plt
from solvers import euler_method

def solve_pvi_a():
    f = lambda t, y: t * np.exp(3*t) - 2*y
    y_true = lambda t: (1/5)*t*np.exp(3*t) - (1/25)*np.exp(3*t) + (1/25)*np.exp(-2*t)
    t_span = (0, 1)
    y0 = 0
    h = 0.05
    
    t_approx, y_approx = euler_method(f, t_span, y0, h)
    t_true = np.linspace(t_span[0], t_span[1], 200)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t_approx, y_approx, 'bo-', label="Aproximação de Euler (h=0.05)")
    plt.plot(t_true, y_true(t_true), 'r-', label="Solução Verdadeira")
    plt.title("PVI (a): $y' = te^{3t} - 2y$")
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.legend()
    plt.grid(True)

    actual_error = np.abs(y_true(t_approx) - y_approx)
    
    L = 2

    M = np.abs((11*1+6)*np.exp(3*1) + 8*y_true(1))
    error_bound = (h * M / (2 * L)) * (np.exp(L * (t_approx - t_span[0])) - 1)
    
    plt.subplot(1, 2, 2)
    plt.plot(t_approx, actual_error, 'g-o', label='Erro Verdadeiro')
    plt.plot(t_approx, error_bound, 'k--', label='Limitante do Erro Teórico')
    plt.title("Análise de Erro para PVI (a)")
    plt.xlabel('t')
    plt.ylabel('Erro Absoluto')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def solve_pvi_b():
    f = lambda t, y: (1 + t) / (1 + y)
    y_true = lambda t: np.sqrt(t**2 + 2*t + 6) - 1
    t_span = (1, 2)
    y0 = 2
    h = 0.05

    t_approx, y_approx = euler_method(f, t_span, y0, h)
    t_true = np.linspace(t_span[0], t_span[1], 200)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t_approx, y_approx, 'bo-', label="Aproximação de Euler (h=0.05)")
    plt.plot(t_true, y_true(t_true), 'r-', label="Solução Verdadeira")
    plt.title("PVI (b): $y' = (1+t)/(1+y)$")
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.legend()
    plt.grid(True)
    
    actual_error = np.abs(y_true(t_approx) - y_approx)
    
    L = (1 + 2) / (1 + 2)**2 
    
    plt.subplot(1, 2, 2)
    plt.plot(t_approx, actual_error, 'g-o', label='Erro Verdadeiro')
    plt.title("Análise de Erro para PVI (b)")
    plt.xlabel('t')
    plt.ylabel('Erro Absoluto')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    solve_pvi_a()
    solve_pvi_b()