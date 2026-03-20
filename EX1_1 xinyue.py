import torch


class LQR:

    def __init__(self, H, M, C, D, R, sigma, T, device="cpu"):

        self.device = device

        # 转成 torch tensor
        self.H = torch.tensor(H, dtype=torch.float32, device=device)
        self.M = torch.tensor(M, dtype=torch.float32, device=device)
        self.C = torch.tensor(C, dtype=torch.float32, device=device)
        self.D = torch.tensor(D, dtype=torch.float32, device=device)
        self.R = torch.tensor(R, dtype=torch.float32, device=device)
        self.sigma = torch.tensor(sigma, dtype=torch.float32, device=device)

        self.T = T

        self.Dinv = torch.linalg.inv(self.D)


    # -----------------------------------------------------
    # Riccati ODE (Euler backward for simplicity)
    # -----------------------------------------------------
    def solve_riccati(self, time_grid):

        time_grid = torch.tensor(time_grid, dtype=torch.float32, device=self.device)

        N = len(time_grid)
        dt = time_grid[1] - time_grid[0]

        S = torch.zeros((N, 2, 2), device=self.device)
        S[-1] = self.R   # terminal condition

        # backward iteration
        for i in reversed(range(N - 1)):
            S_next = S[i + 1]

            dS = (
                -2 * self.H.T @ S_next
                + S_next @ self.M @ self.Dinv @ self.M.T @ S_next
                - self.C
            )

            S[i] = S_next - dt * dS   # backward Euler

        self.time_grid = time_grid
        self.S_grid = S


    # -----------------------------------------------------
    # 插值 S(t) （简单 nearest + index）
    # -----------------------------------------------------
    def get_S(self, t):

        # t: (batch,)
        idx = torch.bucketize(t, self.time_grid) - 1
        idx = torch.clamp(idx, 0, len(self.time_grid) - 1)

        return self.S_grid[idx]   # (batch,2,2)


    # -----------------------------------------------------
    # Value function
    # 输入:
    # t: (batch,)
    # x: (batch,1,2)
    # 输出:
    # (batch,1)
    # -----------------------------------------------------
    def value(self, t, x):

        S = self.get_S(t)   # (batch,2,2)

        xT = x.transpose(1, 2)   # (batch,2,1)

        val = torch.bmm(torch.bmm(x, S), xT)  # (batch,1,1)

        return val.squeeze(-1)   # (batch,1)


    # -----------------------------------------------------
    # Optimal control
    # 输出:
    # (batch,2)
    # -----------------------------------------------------
    def control(self, t, x):

        S = self.get_S(t)   # (batch,2,2)

        xT = x.transpose(1, 2)   # (batch,2,1)

        # (batch,2,1)
        u = - torch.bmm(
            self.Dinv @ self.M.T @ S,
            xT
        )

        return u.squeeze(-1)   # (batch,2)