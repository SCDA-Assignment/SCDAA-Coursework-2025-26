import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# =========================================================
# DEVICE / SEED
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

# =========================================================
# 1. YOUR FFN: used for control a(t,x)
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

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.net(x)

# =========================================================
# 2. NOTEBOOK-STYLE NET: used for value v(t,x)
# input dim = 3 = (t, x1, x2)
# output dim = 1
# =========================================================
class Net(nn.Module):
    def __init__(self, n_layer, n_hidden, dim):
        super(Net, self).__init__()
        self.dim = dim
        self.input_layer = nn.Linear(dim, n_hidden)
        self.hidden_layers = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for _ in range(n_layer)])
        self.output_layer = nn.Linear(n_hidden, 1)

    def forward(self, x):
        o = self.act(self.input_layer(x))
        for li in self.hidden_layers:
            o = self.act(li(o))
        out = self.output_layer(o)
        return out

    def act(self, x):
        return x * torch.sigmoid(x)  # swish

# =========================================================
# 3. ACTOR: wrapper around FFN
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
    tx = [t, x1, x2]
    return 0.5 * tr(sigma sigma^T Hess_x v)
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

    Hx = torch.stack(h_rows, dim=1)  # (batch, 2, 2)
    A = sigma @ sigma.T
    return 0.5 * torch.einsum('ij,bij->b', A, Hx).reshape(-1, 1)

# =========================================================
# 5. RICCATI SOLVER: Exercise 1.1 benchmark
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
        """
        v(t,x) = x^T S(t) x + ∫_t^T tr(sigma sigma^T S(r)) dr
        x shape: (batch, 2)
        """
        idx = int(float(t_scalar) / self.T * (len(self.S_list) - 1))
        idx = max(0, min(idx, len(self.S_list) - 1))
        S_t = self.S_list[idx]

        quad = torch.sum((x @ S_t) * x, dim=1, keepdim=True)

        const = 0.0
        A = self.sigma @ self.sigma.T
        for j in range(idx, len(self.S_list)):
            const += torch.trace(A @ self.S_list[j]).item() * self.dt

        const = torch.tensor([[const]], dtype=torch.float32, device=x.device)
        return quad + const

    def control(self, t_scalar, x):
        """
        a*(t,x) = -D^{-1} M^T S(t) x
        """
        S_t = self.S(t_scalar)
        K = torch.inverse(self.D) @ self.M.T @ S_t
        return -(x @ K.T)

# =========================================================
# 6. PIA PDE / HAMILTONIAN
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

    def value_loss(self, size=2**8):
        pde_err = self.pde_residual(size) ** 2
        term_err = self.terminal_loss(size)
        return torch.mean(pde_err + term_err)

    def hamiltonian(self, size=2**8):
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
# 7. TRAINER
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
        self.value_loss_hist = []
        self.policy_loss_hist = []

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

            for e in range(value_steps):
                opt_v.zero_grad()
                loss_v = self.pia.value_loss(size=2**8)
                loss_v.backward()
                opt_v.step()

                if e % 300 == 299:
                    print(f"Value step {e+1} | loss = {loss_v.item():.6f}")
                    self.value_loss_hist.append(loss_v.item())

            # policy update
            self.critic.eval()
            for p in self.critic.parameters():
                p.requires_grad = False

            self.actor.train()
            for e in range(policy_steps):
                opt_a.zero_grad()
                loss_a = self.pia.hamiltonian(size=2**8)
                loss_a.backward()
                opt_a.step()

                if e % 300 == 299:
                    print(f"Policy step {e+1} | H = {loss_a.item():.6f}")
                    self.policy_loss_hist.append(loss_a.item())

            for p in self.critic.parameters():
                p.requires_grad = True

            H_eval, v_mse, a_mse = self.evaluate_metrics()
            self.H_list.append(H_eval)
            self.v_mse_list.append(v_mse)
            self.a_mse_list.append(a_mse)

            print(f"Eval | H = {H_eval:.6f}, V_MSE = {v_mse:.6f}, A_MSE = {a_mse:.6f}")

# =========================================================
# 8. DEMO FOR VALUE SURFACE
# =========================================================
class DemoValue:
    def __init__(self, critic, ric, T, nt=80, nx=80, x2_fixed=0.0, device='cpu'):
        self.critic = critic
        self.ric = ric
        self.T = T
        self.nt = nt
        self.nx = nx
        self.x2_fixed = x2_fixed
        self.device = device

        self.t_range = np.linspace(0, T, nt)
        self.x1_range = np.linspace(-2, 2, nx)

    def get_solution(self):
        self.est_solution = []
        self.true_solution = []

        for t in self.t_range:
            for x1 in self.x1_range:
                x = torch.tensor([[x1, self.x2_fixed]], dtype=torch.float32, device=self.device)
                tx = torch.tensor([[t, x1, self.x2_fixed]], dtype=torch.float32, device=self.device)

                with torch.no_grad():
                    v_est = self.critic(tx).item()
                    v_true = self.ric.value(t, x).item()

                self.est_solution.append(v_est)
                self.true_solution.append(v_true)

        self.est_solution = np.array(self.est_solution).reshape(self.nt, self.nx)
        self.true_solution = np.array(self.true_solution).reshape(self.nt, self.nx)

    def plot_mesh(self):
        t, x1 = np.meshgrid(self.t_range, self.x1_range, indexing='ij')

        fig1 = plt.figure(figsize=(8, 6))
        ax1 = fig1.add_subplot(111, projection='3d')
        ax1.plot_surface(t, x1, self.est_solution, cmap=cm.RdYlBu_r, edgecolor='none')
        ax1.set_title('Estimated value surface')
        ax1.set_xlabel('t')
        ax1.set_ylabel('x1')
        ax1.set_zlabel('v')
        plt.show()

        fig2 = plt.figure(figsize=(8, 6))
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.plot_surface(t, x1, self.est_solution - self.true_solution, cmap=cm.RdYlBu_r, edgecolor='none')
        ax2.set_title('Value error surface')
        ax2.set_xlabel('t')
        ax2.set_ylabel('x1')
        ax2.set_zlabel('error')
        plt.show()

# =========================================================
# 9. PLOTTING FIVE REQUIRED FIGURES
# =========================================================
def plot_all_results(critic, actor, ric, trainer, device):
    n = 200
    x = -2 + 4 * torch.rand(n, 2, device=device)
    t = torch.zeros(n, 1, device=device)
    tx = torch.cat([t, x], dim=1)

    with torch.no_grad():
        v_nn = critic(tx)
        a_nn = actor(tx)
        v_true = ric.value(0.0, x)
        a_true = ric.control(0.0, x)

    # 1 value compare
    plt.figure(figsize=(6, 4))
    plt.plot(v_true.cpu().numpy(), label='Exact')
    plt.plot(v_nn.cpu().numpy(), '--', label='NN')
    plt.title('Value function comparison')
    plt.xlabel('Sample index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2 control compare
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(a_true[:, 0].cpu().numpy(), label='Exact a1')
    plt.plot(a_nn[:, 0].cpu().numpy(), '--', label='NN a1')
    plt.title('Control dim 1')
    plt.xlabel('Sample index')
    plt.ylabel('Control')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(a_true[:, 1].cpu().numpy(), label='Exact a2')
    plt.plot(a_nn[:, 1].cpu().numpy(), '--', label='NN a2')
    plt.title('Control dim 2')
    plt.xlabel('Sample index')
    plt.ylabel('Control')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3 Hamiltonian
    plt.figure(figsize=(6, 4))
    plt.plot(trainer.H_list, marker='o')
    plt.title('Hamiltonian across outer iterations')
    plt.xlabel('Outer iteration')
    plt.ylabel('Hamiltonian')
    plt.grid(True)
    plt.show()

    # 4 value mse
    plt.figure(figsize=(6, 4))
    plt.plot(trainer.v_mse_list, marker='o')
    plt.title('Value MSE across outer iterations')
    plt.xlabel('Outer iteration')
    plt.ylabel('Value MSE')
    plt.grid(True)
    plt.show()

    # 5 action mse
    plt.figure(figsize=(6, 4))
    plt.plot(trainer.a_mse_list, marker='o')
    plt.title('Action MSE across outer iterations')
    plt.xlabel('Outer iteration')
    plt.ylabel('Action MSE')
    plt.grid(True)
    plt.show()

# =========================================================
# 10. MAIN
# =========================================================
if __name__ == "__main__":
    # problem parameters
    T = 1.0

    H = torch.tensor([[0.1, 0.0],
                      [0.0, 0.2]], dtype=torch.float32, device=device)

    M = torch.eye(2, dtype=torch.float32, device=device)
    C = torch.eye(2, dtype=torch.float32, device=device)
    D = torch.eye(2, dtype=torch.float32, device=device)
    R = torch.eye(2, dtype=torch.float32, device=device)

    sigma = 0.3 * torch.eye(2, dtype=torch.float32, device=device)

    # models
    critic = Net(n_layer=3, n_hidden=128, dim=3).to(device)   # solve v(t,x)
    actor = Actor(hidden_sizes=(64, 64)).to(device)            # solve a(t,x)

    # Riccati benchmark
    ric = RiccatiSolver(H, M, C, D, R, sigma, T, device)
    ric.solve(N=4000)

    # PIA object + trainer
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

    # five required figures
    plot_all_results(critic, actor, ric, trainer, device)

    # optional value surface figures
    demo = DemoValue(critic, ric, T, nt=80, nx=80, x2_fixed=0.0, device=device)
    demo.get_solution()
    demo.plot_mesh()