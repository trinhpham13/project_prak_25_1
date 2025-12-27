import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import json

class PINN(torch.nn.Module):
    def __init__(self, layers, activation='tanh'):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(torch.nn.Linear(layers[i], layers[i + 1]))
        self.activation_dict = {'tanh': torch.nn.Tanh()}
        self.activation = self.activation_dict[activation]
    
    def forward(self, x):
        inputs = x
        for layer in self.layers[:-1]:
            inputs = self.activation(layer(inputs))
        return self.layers[-1](inputs)

# Problem parameters
L, T = 5.0, 1.0
device = torch.device("cpu")

# Exact soliton solution
def exact_soliton(x, t):
    return 0.5 / torch.cosh(0.5 * (x - t))**2

# Load results
with open("results/kdv_architecture_results.json", "r") as f:
    results = json.load(f)

# Process architecture info
arch_data = []
for key, res in results.items():
    layers = list(map(int, key.split("-")))
    num_params = sum(layers[i] * layers[i+1] for i in range(len(layers)-1))
    num_layers = len(layers) - 2
    arch_data.append((num_params, num_layers, res["MSE"], res["Training Time (s)"], layers, key))

arch_data.sort(key=lambda x: x[0])

# Plot performance metrics
params, depths, mses, times, _, _ = zip(*arch_data)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.loglog(params, mses, 'o-', markersize=8, linewidth=2)
plt.xlabel("Number of parameters (log scale)")
plt.ylabel("MSE (log scale)")
plt.title("Accuracy vs. Network Size")
plt.grid(True, which="both", linestyle=":")

plt.subplot(1, 2, 2)
plt.plot(depths, times, 's-', color='orange', markersize=8, linewidth=2)
plt.xlabel("Number of hidden layers")
plt.ylabel("Training time (seconds)")
plt.title("Efficiency vs. Network Depth")
plt.grid(True, linestyle=":")

plt.tight_layout()
os.makedirs("grafs", exist_ok=True)
plt.savefig("grafs/kdv_architecture_comparison.png", dpi=200, bbox_inches='tight')
plt.show()

# Plot solution for each architecture
x_test = np.linspace(-L, L, 200)
t_test = np.linspace(0, T, 50)
X, T_mesh = np.meshgrid(x_test, t_test)
U_exact = exact_soliton(torch.tensor(X), torch.tensor(T_mesh)).numpy()

for _, _, _, _, layers, key in arch_data:
    model_path = f"models/model_kdv_arch_{key}.pth"
    if not os.path.exists(model_path):
        continue

    model = PINN(layers, activation='tanh')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    x_flat = torch.tensor(X.flatten()[:, None], dtype=torch.float32)
    t_flat = torch.tensor(T_mesh.flatten()[:, None], dtype=torch.float32)
    with torch.no_grad():
        U_pred = model(torch.cat([x_flat, t_flat], dim=1)).numpy().reshape(X.shape)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.contourf(X, T_mesh, U_exact, levels=50, cmap='jet')
    plt.colorbar()
    plt.title('Exact Solution')
    plt.xlabel('x')
    plt.ylabel('t')

    plt.subplot(1, 2, 2)
    plt.contourf(X, T_mesh, U_pred, levels=50, cmap='jet')
    plt.colorbar()
    plt.title(f'PINN Prediction: {layers}')
    plt.xlabel('x')
    plt.ylabel('t')

    plt.tight_layout()
    plt.savefig(f"grafs/kdv_solution_{key}.png", dpi=150, bbox_inches='tight')
    plt.show()

print("Visualization complete. All plots saved to the 'grafs' directory.")