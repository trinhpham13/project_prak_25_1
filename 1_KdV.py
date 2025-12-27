import torch
import torch.nn as nn
import time
import numpy as np
import os
import json

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# Define Physics-Informed Neural Network (PINN)
class PINN(nn.Module):
    def __init__(self, layers, activation='tanh'):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.activation_dict = {
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'softplus': nn.Softplus(),
        }
        self.activation = self.activation_dict[activation]

    def forward(self, x):
        inputs = x
        for layer in self.layers[:-1]:
            inputs = self.activation(layer(inputs))
        return self.layers[-1](inputs)


L, T = 5.0, 1.0
N_eq, N_ic, N_bc = 10000, 500, 500


def exact_soliton(x, t):
    return 0.5 / torch.cosh(0.5 * (x - t))**2

# Loss function
def loss_function(model, x_eq, t_eq, x_ic, t_ic, x_bc, t_bc):
    x_eq = x_eq.requires_grad_(True)
    t_eq = t_eq.requires_grad_(True)
    u_eq = model(torch.cat([x_eq, t_eq], dim=1))
    
    u_t = torch.autograd.grad(u_eq, t_eq, grad_outputs=torch.ones_like(u_eq), create_graph=True)[0]
    u_x = torch.autograd.grad(u_eq, x_eq, grad_outputs=torch.ones_like(u_eq), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_eq, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx, x_eq, grad_outputs=torch.ones_like(u_xx), create_graph=True)[0]
    
    f_eq = u_t + 6.0 * u_eq * u_x + u_xxx
    loss_eq = torch.mean(f_eq**2)

    u_ic_pred = model(torch.cat([x_ic, t_ic], dim=1))
    u_ic_true = exact_soliton(x_ic, t_ic)
    loss_ic = torch.mean((u_ic_pred - u_ic_true)**2)

    u_bc_left = model(torch.cat([x_bc[:N_bc//2], t_bc[:N_bc//2]], dim=1))
    u_bc_right = model(torch.cat([x_bc[N_bc//2:], t_bc[N_bc//2:]], dim=1))
    loss_bc_dirichlet = torch.mean(u_bc_left**2) + torch.mean(u_bc_right**2)

    x_bc_left = x_bc[:N_bc//2].requires_grad_(True)
    t_bc_left = t_bc[:N_bc//2].requires_grad_(True)
    u_bc_left_grad = model(torch.cat([x_bc_left, t_bc_left], dim=1))
    u_x_bc_left = torch.autograd.grad(u_bc_left_grad, x_bc_left, grad_outputs=torch.ones_like(u_bc_left_grad), create_graph=True)[0]
    loss_bc_neumann = torch.mean(u_x_bc_left**2)
    
    loss_bc = loss_bc_dirichlet + loss_bc_neumann

    return loss_eq + 10.0 * loss_ic + 10.0 * loss_bc

# on grid
def evaluate_model(model, device):
    x_test = np.linspace(-L, L, 200)
    t_test = np.linspace(0, T, 50)
    X, T_mesh = np.meshgrid(x_test, t_test)
    x_flat = torch.tensor(X.flatten()[:, None], dtype=torch.float32).to(device)
    t_flat = torch.tensor(T_mesh.flatten()[:, None], dtype=torch.float32).to(device)
    with torch.no_grad():
        u_pred = model(torch.cat([x_flat, t_flat], dim=1)).cpu().numpy()
    u_exact = exact_soliton(x_flat.cpu(), t_flat.cpu()).numpy()
    return np.mean((u_pred - u_exact) ** 2)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


architectures = [
    [2, 20, 20, 1],
    [2, 50, 50, 1],
    [2, 100, 100, 100, 1],
    [2, 200, 200, 1],
    [2, 50, 50, 50, 50, 1],
]

results = {}


for arch in architectures:
    arch_key = "-".join(map(str, arch))
    print(f"\nTraining architecture: {arch_key}")
    
    model = PINN(arch, activation='tanh').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

  
    x_eq = (torch.rand(N_eq, 1, device=device) * 2*L - L)
    t_eq = torch.rand(N_eq, 1, device=device) * T
    x_ic = (torch.rand(N_ic, 1, device=device) * 2*L - L)
    t_ic = torch.zeros(N_ic, 1, device=device)
    t_bc = torch.rand(N_bc, 1, device=device) * T
    x_bc = torch.cat([-L * torch.ones(N_bc//2, 1, device=device), L * torch.ones(N_bc//2, 1, device=device)], dim=0)
    t_bc = torch.cat([t_bc[:N_bc//2], t_bc[:N_bc//2]], dim=0)

    start_time = time.time()
    for epoch in range(20001):
        optimizer.zero_grad()
        loss = loss_function(model, x_eq, t_eq, x_ic, t_ic, x_bc, t_bc)
        loss.backward()
        optimizer.step()
    train_time = time.time() - start_time

    mse = evaluate_model(model, device)
    results[arch_key] = {"MSE": float(mse), "Training Time (s)": float(train_time)}
    
    torch.save(model.state_dict(), f"models/model_kdv_arch_{arch_key}.pth")
    print(f"   MSE: {mse:.2e} | Training time: {train_time:.2f} seconds")


with open("results/kdv_architecture_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nTraining complete. Results saved to: results/kdv_architecture_results.json")