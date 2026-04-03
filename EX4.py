import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# DEVICE / SEED
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

# =========================================================
# 1. FFN: used for control a(t,x)
# =========================================================
class FFN(nn.Module):
    def __init__(self, sizes, activation=nn.ReLU, output_activation=nn.Identity, batch_norm=False):
        super().__init__()
        layers = [nn.BatchNorm1d(sizes[0])] if batch_norm else []
        for j in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[j], sizes[j + 1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(sizes[j + 1], affine=True))
            if j < len(sizes) - 2:
                layers.append(activation())
            else:
                layers.append(output_activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# =========================================================
# 2. Net: used for value v(t,x)
# input dim = 3 = (t, x1, x2), output dim = 1
# =========================================================
class Net(nn.Module):
    def __init__(self, n_layer, n_hidden, dim):
        super().__init__()
        self.input_layer = nn.Linear(dim, n_hidden)
        self.hidden_layers = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for _ in range(n_layer)])
        self.output_layer = nn.Linear(n_hidden, 1)

    def act(self, x):
        return x * torch.sigmoid(x)  # swish

    def forward(self, x):
        o = self.act(self.input_layer(x))
        for layer in self.hidden_layers:
            o = self.act(layer(o))
        return self.output_layer(o)

# =========================================================
# 3. Actor wrapper around FFN
# =========================================================
class Actor(nn.Module):
    def __init__(self, hidden_sizes=(64, 64)):
        super().__init__()
        sizes = [3] + list(hidden_sizes) + [2]
        self.net = FFN(
            sizes=sizes,
            activation=nn.Tanh,
            output_activation=nn.Identity,
            batch_norm=False
        )

    def forward(self, tx):
        return self.net(tx)

# =========================================================
# 4. AUTOGRAD UTILS
# =========================================================
def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True
    )[0]

def diffusion_term_from_tx(v, tx, sigma):
    """
    return 0.5 * tr(sigma sigma^T Hess_x v)
    tx = [t, x1, x2]
    """
    dv_dtx = torch.autograd.grad(
        v, tx,
        grad_outputs=torch.ones_like(v),
        create_graph=True,
        retain_graph=True
    )[0]

    vx = dv_dtx[:, 1:3]
    h_rows = []
    for i in range(2):
        grad_i = vx[:, i:i+1]
        second = torch.autograd.grad(
            grad_i, tx,
            grad_outputs=torch.ones_like(grad_i),
            create_graph=True,
            retain_graph=True
        )[0][:, 1:3]
        h_rows.append(second)

    Hx = torch.stack(h_rows, dim=1)   # (batch, 2, 2)
    A = sigma @ sigma.T
    return 0.5 * torch.einsum('ij,bij->b', A, Hx).reshape(-1, 1)

# =========================================================
# 5. Riccati benchmark
# =========================================================
class RiccatiSolver:
    def __init__(self, H, M, C, D, R, sigma, T, device):
        self.H = H
        self.M = M
        self.C = C
        self.D = D
        self.R = R
        self.sigma = sigma
        self.T = T
        self.device = device
        self.S_list = []
        self.dt = None

    def solve(self, N=4000):
        self.dt = self.T / N
        S = self.R.clone()
        S_rev = [S.clone()]

        for _ in range(N):
            dS = -2 * self.H.T @ S + S @ self.M @ torch.inverse(self.D) @ self.M.T @ S - self.C
            S = S - self.dt * dS
            S_rev.append(S.clone())

        self.S_list = list(reversed(S_rev))

    def S(self, t_scalar):
        idx = int(float(t_scalar) / self.T * (len(self.S_list) - 1))
        idx = max(0, min(idx, len(self.S_list) - 1))
        return self.S_list[idx]

    def value(self, t_scalar, x):
        S_t = self.S(t_scalar)
        quad = torch.sum((x @ S_t) * x, dim=1, keepdim=True)

        idx = int(float(t_scalar) / self.T * (len(self.S_list) - 1))
        idx = max(0, min(idx, len(self.S_list) - 1))

        const = 0.0
        A = self.sigma @ self.sigma.T
        for j in range(idx, len(self.S_list)):
            const += torch.trace(A @ self.S_list[j]).item() * self.dt

        const = torch.tensor([[const]], dtype=torch.float32, device=x.device)
        return quad + const

    def control(self, t_scalar, x):
        S_t = self.S(t_scalar)
        K = torch.inverse(self.D) @ self.M.T @ S_t
        return -(x @ K.T)

# =========================================================
# 6. PIA model
# =========================================================
class LQR_PIA:
    def __init__(self, critic, actor, H, M, C, D, R, sigma, T, device):
        self.critic = critic
        self.actor = actor
        self.H = H
        self.M = M
        self.C = C
        self.D = D
        self.R = R
        self.sigma = sigma
        self.T = T
        self.device = device

    def pde_residual(self, size):
        t = torch.rand(size, 1, device=self.device) * self.T
        x = -2 + 4 * torch.rand(size, 2, device=self.device)

        tx = torch.cat([t, x], dim=1).clone().detach().requires_grad_(True)

        v = self.critic(tx)
        dv_dtx = grad(v, tx)

        vt = dv_dtx[:, 0:1]
        x_var = tx[:, 1:3]
        vx = dv_dtx[:, 1:3]
        a = self.actor(tx)

        drift = x_var @ self.H.T + a @ self.M.T
        diff = diffusion_term_from_tx(v, tx, self.sigma)

        xCx = torch.sum((x_var @ self.C) * x_var, dim=1, keepdim=True)
        aDa = torch.sum((a @ self.D) * a, dim=1, keepdim=True)

        residual = vt + diff + torch.sum(vx * drift, dim=1, keepdim=True) + xCx + aDa
        return residual

    def terminal_loss(self, size):
        x = -2 + 4 * torch.rand(size, 2, device=self.device)
        t = torch.ones(size, 1, device=self.device) * self.T
        tx = torch.cat([t, x], dim=1)

        vT = self.critic(tx)
        target = torch.sum((x @ self.R) * x, dim=1, keepdim=True)
        return (vT - target) ** 2

    def value_loss(self, size=256):
        return torch.mean(self.pde_residual(size) ** 2 + self.terminal_loss(size))

    def hamiltonian(self, size=256):
        t = torch.rand(size, 1, device=self.device) * self.T
        x = -2 + 4 * torch.rand(size, 2, device=self.device)

        tx = torch.cat([t, x], dim=1).clone().detach().requires_grad_(True)
        v = self.critic(tx)
        dv_dtx = grad(v, tx)

        x_var = tx[:, 1:3]
        vx = dv_dtx[:, 1:3]
        a = self.actor(tx)

        Hx = x_var @ self.H.T
        Ma = a @ self.M.T

        Hamil = (
            torch.sum(vx * Hx, dim=1, keepdim=True)
            + torch.sum(vx * Ma, dim=1, keepdim=True)
            + torch.sum((x_var @ self.C) * x_var, dim=1, keepdim=True)
            + torch.sum((a @ self.D) * a, dim=1, keepdim=True)
        )
        return torch.mean(Hamil)

# =========================================================
# 7. Trainer
# =========================================================
class TrainPIA:
    def __init__(self, critic, actor, pia, ric, device):
        self.critic = critic
        self.actor = actor
        self.pia = pia
        self.ric = ric
        self.device = device

        self.H_list = []
        self.v_mse_list = []
        self.a_mse_list = []

    def evaluate_metrics(self, n_test=300):
        x = -2 + 4 * torch.rand(n_test, 2, device=self.device)
        t = torch.zeros(n_test, 1, device=self.device)
        tx = torch.cat([t, x], dim=1)

        tx_var = tx.clone().detach().requires_grad_(True)
        x_var = tx_var[:, 1:3]

        v = self.critic(tx_var)
        dv_dtx = grad(v, tx_var)
        vx = dv_dtx[:, 1:3]
        a = self.actor(tx_var)

        H_eval = (
            torch.sum(vx * (x_var @ self.pia.H.T), dim=1, keepdim=True)
            + torch.sum(vx * (a @ self.pia.M.T), dim=1, keepdim=True)
            + torch.sum((x_var @ self.pia.C) * x_var, dim=1, keepdim=True)
            + torch.sum((a @ self.pia.D) * a, dim=1, keepdim=True)
        ).mean().item()

        with torch.no_grad():
            v_true = self.ric.value(0.0, x)
            a_true = self.ric.control(0.0, x)

            v_nn = self.critic(tx)
            a_nn = self.actor(tx)

            v_mse = torch.mean((v_nn - v_true) ** 2).item()
            a_mse = torch.mean((a_nn - a_true) ** 2).item()

        return H_eval, v_mse, a_mse

    def train(self, outer_iters=8, value_steps=3000, policy_steps=1500, lr_v=0.002, lr_a=0.001):
        opt_v = optim.Adam(self.critic.parameters(), lr=lr_v)
        opt_a = optim.Adam(self.actor.parameters(), lr=lr_a)

        for k in range(outer_iters):
            print(f"\n=== Outer Iteration {k+1}/{outer_iters} ===")

            # value update
            self.actor.eval()
            self.critic.train()
            for _ in range(value_steps):
                opt_v.zero_grad()
                loss_v = self.pia.value_loss(size=256)
                loss_v.backward()
                opt_v.step()

            # policy update
            self.critic.eval()
            for p in self.critic.parameters():
                p.requires_grad = False

            self.actor.train()
            for _ in range(policy_steps):
                opt_a.zero_grad()
                loss_a = self.pia.hamiltonian(size=256)
                loss_a.backward()
                opt_a.step()

            for p in self.critic.parameters():
                p.requires_grad = True

            H_eval, v_mse, a_mse = self.evaluate_metrics()
            self.H_list.append(H_eval)
            self.v_mse_list.append(v_mse)
            self.a_mse_list.append(a_mse)

            print(f"Eval | H = {H_eval:.6f}, V_MSE = {v_mse:.6f}, A_MSE = {a_mse:.6f}")

# =========================================================
# 8. Only three figures
# =========================================================
def plot_three_figures(critic, actor, ric, trainer, device):
    # Fix t = 0 and x2 = 0, vary x1
    x1_grid = torch.linspace(-2, 2, 200, device=device).reshape(-1, 1)
    x2_grid = torch.zeros_like(x1_grid)
    t_grid = torch.zeros_like(x1_grid)

    x = torch.cat([x1_grid, x2_grid], dim=1)
    tx = torch.cat([t_grid, x1_grid, x2_grid], dim=1)

    with torch.no_grad():
        v_nn = critic(tx).cpu().numpy().flatten()
        a_nn = actor(tx).cpu().numpy()
        v_true = ric.value(0.0, x).cpu().numpy().flatten()
        a_true = ric.control(0.0, x).cpu().numpy()

    x1_np = x1_grid.cpu().numpy().flatten()

    # Figure 1: value comparison
    plt.figure(figsize=(6, 4))
    plt.plot(x1_np, v_true, label='Exact value')
    plt.plot(x1_np, v_nn, '--', label='NN value')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$v(0, x_1, 0)$')
    plt.title('Value function comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Figure 2: control comparison
    plt.figure(figsize=(7, 4))
    plt.plot(x1_np, a_true[:, 0], label='Exact $a_1$')
    plt.plot(x1_np, a_nn[:, 0], '--', label='NN $a_1$')
    plt.plot(x1_np, a_true[:, 1], label='Exact $a_2$')
    plt.plot(x1_np, a_nn[:, 1], '--', label='NN $a_2$')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$a(0, x_1, 0)$')
    plt.title('Control comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Figure 3: convergence
    plt.figure(figsize=(7, 4))
    plt.plot(trainer.v_mse_list, marker='o', label='Value MSE')
    plt.plot(trainer.a_mse_list, marker='s', label='Action MSE')
    plt.plot(trainer.H_list, marker='^', label='Hamiltonian')
    plt.xlabel('Outer iteration')
    plt.ylabel('Metric')
    plt.title('Convergence across outer iterations')
    plt.legend()
    plt.grid(True)
    plt.show()

# =========================================================
# 9. MAIN
# =========================================================
if __name__ == "__main__":
    T = 1.0

    H = torch.tensor([[0.1, 0.0],
                      [0.0, 0.2]], dtype=torch.float32, device=device)
    M = torch.eye(2, dtype=torch.float32, device=device)
    C = torch.eye(2, dtype=torch.float32, device=device)
    D = torch.eye(2, dtype=torch.float32, device=device)
    R = torch.eye(2, dtype=torch.float32, device=device)
    sigma = 0.3 * torch.eye(2, dtype=torch.float32, device=device)

    # critic for v, actor for a
    critic = Net(n_layer=3, n_hidden=128, dim=3).to(device)
    actor = Actor(hidden_sizes=(64, 64)).to(device)

    # benchmark
    ric = RiccatiSolver(H, M, C, D, R, sigma, T, device)
    ric.solve(N=4000)

    # training objects
    pia = LQR_PIA(critic, actor, H, M, C, D, R, sigma, T, device)
    trainer = TrainPIA(critic, actor, pia, ric, device)

    # train
    trainer.train(
        outer_iters=8,
        value_steps=3000,
        policy_steps=1500,
        lr_v=0.002,
        lr_a=0.001
    )

    # plot 3 required figures
    plot_three_figures(critic, actor, ric, trainer, device)