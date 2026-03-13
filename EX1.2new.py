import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

T = 1.0

H = np.array([[0.1, 0.0], [0.0, 0.2]])
M = np.eye(2)
C = np.eye(2)
D = np.eye(2)
R = np.eye(2)
sigma = 0.3 * np.eye(2)

x0 = np.array([1.0, 1.0])


class LQR:

    def __init__(self, H, M, C, D, R, sigma, T):
        self.H = H
        self.M = M
        self.C = C
        self.D = D
        self.R = R
        self.sigma = sigma
        self.T = T
        self.Dinv = np.linalg.inv(D) 

    # Riccati ODE solver (backward Euler)
    def solve_riccati(self, t_grid):
        
        N = len(t_grid)
        dt = t_grid[1] - t_grid[0]

        S = np.zeros((N, 2, 2))
        S[-1] = self.R

        for i in reversed(range(N - 1)):
            S_next = S[i + 1]

            drift = (
                -2 * self.H.T @ S_next
                + S_next @ self.M @ self.Dinv @ self.M.T @ S_next
                - self.C
            )

            S[i] = S_next - dt * drift

        self.S = S
        self.t_grid = t_grid

    def get_S(self, t):
        idx = np.argmin(np.abs(self.t_grid - t))
        return self.S[idx]

    # value function
    def value_function(self, t, x):
        S = self.get_S(t)
        return x.T @ S @ x

    # optimal control
    def optimal_control(self, t, x):
        S = self.get_S(t)
        return -self.Dinv @ self.M.T @ S @ x

def simulate_LQR(lqr, x0, N, MC):
    T = lqr.T
    dt = T / N
    d = len(x0)

    cost = np.zeros(MC)

    for m in range(MC):
        X = x0.copy()
        running_cost = 0

        for n in range(N):
            t = n * dt

            u = lqr.optimal_control(t, X)

            drift = lqr.H @ X + lqr.M @ u

            dW = np.sqrt(dt) * np.random.randn(d)

            X = X + drift * dt + lqr.sigma @ dW

            running_cost += (
                X.T @ lqr.C @ X +
                u.T @ lqr.D @ u
            ) * dt

        terminal_cost = X.T @ lqr.R @ X
        cost[m] = running_cost + terminal_cost

    return np.mean(cost)


# create lqr
lqr = LQR(H, M, C, D, R, sigma, T)

t_grid = np.linspace(0, T, 5000)
lqr.solve_riccati(t_grid)

true_value = lqr.value_function(0, x0)

print("Value function:", true_value)

# EXPERIMENT 1
# vary time steps
MC = int(1e5)

time_steps = [1, 10, 50, 100, 500, 1000, 5000]

errors_time = []

for N in time_steps:
    print(f"Running time steps experiment: N={N}")
    est = simulate_LQR(lqr, x0, N, MC)
    error = abs(est - true_value)
    errors_time.append(error)

plt.figure()
plt.loglog(time_steps, errors_time, marker='o')
plt.xlabel("time steps")
plt.ylabel("error")
plt.title("Error vs time discretisation")
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.show()

# EXPERIMENT 2
# vary MC samples

N = 5000

MC_samples = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]

errors_MC = []

for MC in MC_samples:
    print(f"Running MC samples experiment: MC={MC}")
    est = simulate_LQR(lqr, x0, N, MC)
    error = abs(est - true_value)
    errors_MC.append(error)

plt.figure()
plt.loglog(MC_samples, errors_MC, marker='o')
plt.xlabel("MC samples")
plt.ylabel("error")
plt.title("Error vs Monte Carlo samples")
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.show()