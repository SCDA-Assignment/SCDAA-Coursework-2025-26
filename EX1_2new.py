import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)

T = 1.0

H = torch.tensor([[0.1, 0.0], [0.0, 0.2]])
M = torch.eye(2)
C = torch.eye(2)
D = torch.eye(2)
R = torch.eye(2)
sigma = 0.3 * torch.eye(2)

x0 = torch.tensor([1.0, 1.0])


class LQR:

    def __init__(self, H, M, C, D, R, sigma, T):

        self.H = H
        self.M = M
        self.C = C
        self.D = D
        self.R = R
        self.sigma = sigma
        self.T = T
        self.Dinv = torch.linalg.inv(D)

    # Riccati solver
    def solve_riccati(self, t_grid):

        N = len(t_grid)
        dt = t_grid[1] - t_grid[0]

        S = torch.zeros(N, 2, 2)
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

        idx = torch.argmin(torch.abs(self.t_grid - t))
        return self.S[idx]

    # value function
    def value_function(self, t, x):

        S = self.get_S(t)

        value = x @ S @ x

        idx = torch.argmin(torch.abs(self.t_grid - t))

        trace_vals = torch.stack([
            torch.trace(self.sigma @ self.sigma.T @ S_i)
            for S_i in self.S[idx:]
        ])

        dt = self.t_grid[1] - self.t_grid[0]

        integral = torch.sum((trace_vals[:-1] + trace_vals[1:]) * 0.5 * dt)

        return value + integral

    # control
    def optimal_control(self, t, X):

        S = self.get_S(t)

        return -(X @ S.T @ self.M @ self.Dinv.T)


# ===== 向量化 Monte Carlo =====

def simulate_LQR(lqr, x0, N, MC):

    T = lqr.T
    dt = T / N
    d = len(x0)

    # MC paths
    X = x0.repeat(MC, 1)

    running_cost = torch.zeros(MC)

    for n in range(N):

        t = n * dt

        S = lqr.get_S(torch.tensor(t))

        # optimal control
        u = -(X @ S.T)

        drift = X @ lqr.H.T + u @ lqr.M.T

        dW = torch.sqrt(torch.tensor(dt)) * torch.randn(MC, d)

        running_cost += (
            torch.sum((X @ lqr.C) * X, dim=1)
            + torch.sum((u @ lqr.D) * u, dim=1)
        ) * dt

        X = X + drift * dt + dW @ lqr.sigma.T

    terminal_cost = torch.sum((X @ lqr.R) * X, dim=1)

    cost = running_cost + terminal_cost

    return torch.mean(cost)


# ===== Create LQR =====

lqr = LQR(H, M, C, D, R, sigma, T)

t_grid = torch.linspace(0, T, 5000)

lqr.solve_riccati(t_grid)

true_value = lqr.value_function(torch.tensor(0.0), x0)

print("Value function:", true_value.item())


# ======================
# Experiment 1
# ======================

MC = 100000

time_steps = [1, 10, 50, 100, 500, 1000, 5000]

errors_time = []

for N in time_steps:

    print(f"Running time steps experiment: N={N}")

    est = simulate_LQR(lqr, x0, N, MC)

    error = torch.abs(est - true_value)

    errors_time.append(error.item())

plt.figure()

plt.loglog(time_steps, errors_time, marker='o')

plt.xlabel("time steps")
plt.ylabel("error")

plt.title("Error vs time discretisation")

plt.grid(True)

plt.show()


# ======================
# Experiment 2
# ======================

N = 5000

MC_samples = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]

errors_MC = []

for MC in MC_samples:

    print(f"Running MC samples experiment: MC={MC}")

    est = simulate_LQR(lqr, x0, N, MC)

    error = torch.abs(est - true_value)

    errors_MC.append(error.item())

plt.figure()

plt.loglog(MC_samples, errors_MC, marker='o')

plt.xlabel("MC samples")
plt.ylabel("error")

plt.title("Error vs Monte Carlo samples")

plt.grid(True)

plt.show()