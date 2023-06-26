import numpy as np
from scipy.integrate import solve_ivp

def rhs(t, variables, k1, k2, c, f, v):
    x = variables[:3]
    y = variables[3:12].reshape((3, 3))
    z = variables[12:].reshape((3, 3))

    I = np.identity(3)
    n = np.cross(f, v)
    v_norm = np.linalg.norm(v)
    n_norm = np.linalg.norm(n)
    norm_sq = 1 + v_norm**2
    normn_sq = 1 + n_norm**2

    dx_dt = -k1 * np.dot((I - np.outer(x, x)), np.dot(y, x)) - k2 * np.dot((I - np.outer(x, x)), np.dot(z, x))
    dy_dt = -c * y + np.outer(v, v) / norm_sq
    dz_dt = -c * z + np.outer(n, n) / normn_sq

    return np.concatenate((dx_dt, dy_dt.flatten(), dz_dt.flatten()))

def solve_coupled_equations(k1, k2, c, f, v):
    x0 = np.array([0.5770625, 0.5770625, 0.5770625])
    y0 = np.zeros((3, 3))
    z0 = np.zeros((3, 3))

    initial_values = np.concatenate((x0, y0.flatten(), z0.flatten())) # Carefully enter the initial values, the scipy solve_ivp only takes flatten arrays.
    t_span = (0, 100)  # Adjust the time span as needed

    solution = solve_ivp(
        fun=lambda t, variables: rhs(t, variables, k1, k2, c, f, v),
        t_span=t_span,
        y0=initial_values,
        method='RK45',
        dense_output=True
    )

    return solution

# Example usage
k1 = 10
k2 = 15
c = 5
f = np.array([1, 1, -1])
v = np.array([1, 1, 0])

solution = solve_coupled_equations(k1, k2, c, f, v)

# Extract the solution
t = np.linspace(0, 100, 10000)  # Time points for evaluation
solution_vals = solution.sol(t)

# Extract individual variables
x_solution = solution_vals[:3]
y_solution = solution_vals[3:12].reshape((3, 3, len(t)))
z_solution = solution_vals[12:].reshape((3, 3, len(t)))

# Print the solution at different time points , printing only x here.
for i in range(len(t)):
    print(f"Time: {t[i]}")
    print("x:", x_solution[:, i])
