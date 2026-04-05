import torch
class LQR:

    # (i) Initialise the class with LQR model parameters
    def __init__(self, H, M, C, D, R, sigma, T):

        self.H = H.float()
        self.M = M.float()
        self.C = C.float()
        self.D = D.float()
        self.R = R.float()
        self.sigma = sigma.float()
        self.T = T

        # Precompute D^{-1}
        self.Dinv = torch.linalg.inv(self.D)

    # (ii) Solve the Riccati ODE on a given time grid
    def solve_riccati(self, t_grid):

        if not torch.is_tensor(t_grid):
            t_grid = torch.tensor(t_grid, dtype=torch.float32)
        else:
            t_grid = t_grid.float()

        N = len(t_grid)
        dt = t_grid[1] - t_grid[0]

        # Store S(t)
        S = torch.zeros(N, 2, 2, dtype=torch.float32)
        S[-1] = self.R   # Terminal condition: S(T) = R

        # Backward Euler scheme for Riccati ODE
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

        # Compute g(t) = ∫_t^T tr(σσ^T S(r)) dr
        sigma_sigma_T = self.sigma @ self.sigma.T
        trace_vals = torch.stack([
            torch.trace(sigma_sigma_T @ S_i) for S_i in self.S
        ])

        g = torch.zeros(N, dtype=torch.float32)
        for i in range(N - 2, -1, -1):
            g[i] = g[i + 1] + 0.5 * (trace_vals[i] + trace_vals[i + 1]) * dt

        self.g = g

    # Helper function: retrieve S(t) for given t
    def get_S(self, t):

        if t.dim() == 0:
            idx = torch.argmin(torch.abs(self.t_grid - t))
            return self.S[idx]

        diff = torch.abs(self.t_grid.unsqueeze(1) - t.unsqueeze(0))
        idx = torch.argmin(diff, dim=0)
        return self.S[idx]

    # Helper function: retrieve g(t) for given t
    def get_g(self, t):

        if t.dim() == 0:
            idx = torch.argmin(torch.abs(self.t_grid - t))
            return self.g[idx]

        diff = torch.abs(self.t_grid.unsqueeze(1) - t.unsqueeze(0))
        idx = torch.argmin(diff, dim=0)
        return self.g[idx]
    
    # (iii) Compute the value function v(t, x)
    # Input:
    # t: (batch,)
    # x: (batch, 1, 2)
    # Output:
    # (batch, 1)
    def value_function(self, t, x):

        S = self.get_S(t)              # (batch, 2, 2)
        g = self.get_g(t)              # (batch,)

        xT = x.transpose(1, 2)         # (batch, 2, 1)

        value = torch.bmm(torch.bmm(x, S), xT)   # (batch, 1, 1)
        value = value.squeeze(-1)                # (batch, 1)

        return value + g.unsqueeze(1)
   
    # (iv) Compute the optimal control a(t, x)
    # Input:
    # t: (batch,)
    # X: (batch, 1, 2)
    # Output:
    # (batch, 2)
    def optimal_control(self, t, X):

        S = self.get_S(t)

        # Single sample case: X shape = (2,)
        if X.dim() == 1:
            return -(self.Dinv @ self.M.T @ S @ X)

        # Batch case
        xT = X.transpose(1, 2)   # (batch, 2, 1)

        A = (self.Dinv @ self.M.T).unsqueeze(0).expand(S.size(0), -1, -1)
        SX = torch.bmm(S, xT)
        u = -torch.bmm(A, SX)

        return u.squeeze(-1)     # (batch, 2)