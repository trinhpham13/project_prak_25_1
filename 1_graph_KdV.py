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


L, T = 5.0, 1.0
device = torch.device("cpu")

def exact_soliton(x, t):
    return 0.5 / torch.cosh(0.5 * (x - t))**2

with open("results/kdv_architecture_results.json", "r") as f:
    results = json.load(f)

architectures = [
    [2, 20, 20, 1],
    [2, 50, 50, 1],
    [2, 100, 100, 100, 1],
    [2, 200, 200, 1],
    [2, 50, 50, 50, 50, 1],
]

arch_names = [
    "2-20-20-1",
    "2-50-50-1",
    "2-100-100-100-1",
    "2-200-200-1",
    "2-50-50-50-50-1"
]

arch_data = []
for i, layers in enumerate(architectures):
    key = arch_names[i]
    num_params = sum(layers[j] * layers[j+1] for j in range(len(layers)-1))
    num_hidden_layers = len(layers) - 2
    mse = results[key]["MSE"]
    time_sec = results[key]["Training Time (s)"]
    arch_data.append((num_params, num_hidden_layers, mse, time_sec, layers, key))

x_test = np.linspace(-L, L, 200)
t_test = np.linspace(0, T, 50)
X, T_mesh = np.meshgrid(x_test, t_test)
U_exact = exact_soliton(torch.tensor(X), torch.tensor(T_mesh)).numpy()

os.makedirs("grafs", exist_ok=True)

for layers, key in zip(architectures, arch_names):
    model_path = f"models/model_kdv_arch_{key}.pth"
    if not os.path.exists(model_path):
        print(f"Warning: Model {key} not found. Skipping.")
        continue

    print(f"Generating comparison plot for: {key}")
    model = PINN(layers, activation='tanh')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    x_flat = torch.tensor(X.flatten()[:, None], dtype=torch.float32)
    t_flat = torch.tensor(T_mesh.flatten()[:, None], dtype=torch.float32)
    with torch.no_grad():
        U_pred = model(torch.cat([x_flat, t_flat], dim=1)).numpy().reshape(X.shape)

    plt.figure(figsize=(14, 5))
    
    # Exact solution
    plt.subplot(1, 2, 1)
    im1 = plt.contourf(X, T_mesh, U_exact, levels=50, cmap='jet')
    plt.colorbar(im1)
    plt.title('Exact Solution')
    plt.xlabel('x')
    plt.ylabel('t')

    plt.subplot(1, 2, 2)
    im2 = plt.contourf(X, T_mesh, U_pred, levels=50, cmap='jet')
    plt.colorbar(im2)
    plt.title(f'PINN Prediction\nArchitecture: {key}\nMSE = {results[key]["MSE"]:.2e}')
    plt.xlabel('x')
    plt.ylabel('t')

    plt.tight_layout()
    plt.savefig(f"grafs/kdv_comparison_{key}.png", dpi=200, bbox_inches='tight')
    plt.close()  

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

arch_data_sorted_by_params = sorted(arch_data, key=lambda x: x[0])
params, _, mses, _, _, keys = zip(*arch_data_sorted_by_params)
axes[0].semilogy(params, mses, 'o-', linewidth=2, markersize=8)
for i, (p, m, k) in enumerate(zip(params, mses, keys)):
    axes[0].annotate(k, (p, m), textcoords="offset points", xytext=(0,10), ha='center')
axes[0].set_xlabel('Number of parameters (log scale)')
axes[0].set_ylabel('MSE (log scale)')
axes[0].set_title('Accuracy vs. Network Size')
axes[0].grid(True, which="both", linestyle=":")

arch_data_sorted_by_depth = sorted(arch_data, key=lambda x: x[1])
depths_plot, times_plot, _, _, _, keys_plot = zip(*arch_data_sorted_by_depth)
axes[1].plot(depths_plot, times_plot, 's-', color='orange', linewidth=2, markersize=8)
for i, (d, t, k) in enumerate(zip(depths_plot, times_plot, keys_plot)):
    axes[1].annotate(k, (d, t), textcoords="offset points", xytext=(0,10), ha='center')
axes[1].set_xlabel('Number of hidden layers')
axes[1].set_ylabel('Training time (seconds)')
axes[1].set_title('Efficiency vs. Network Depth')
axes[1].grid(True, linestyle=":")

ax_table = axes[2]
ax_table.axis('tight')
ax_table.axis('off')
table_data = [[key, f"{mse:.2e}", f"{time_sec:.1f}"] for key, (_, _, mse, time_sec, _, _) in zip(keys, arch_data)]
table = ax_table.table(cellText=table_data,
                       colLabels=['Architecture', 'MSE', 'Time (s)'],
                       loc='center',
                       cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
ax_table.set_title('Summary Table')

best_key = "2-50-50-50-50-1"
if best_key in results:
    model_path = f"models/model_kdv_arch_{best_key}.pth"
    if os.path.exists(model_path):
        model = PINN(architectures[4], activation='tanh')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        with torch.no_grad():
            U_pred_best = model(torch.cat([x_flat, t_flat], dim=1)).numpy().reshape(X.shape)
        axes[3].contourf(X, T_mesh, U_pred_best, levels=50, cmap='jet')
        axes[3].set_title(f'Best Model Prediction\n({best_key})')
        axes[3].set_xlabel('x')
        axes[3].set_ylabel('t')
        plt.colorbar(axes[3].collections[0], ax=axes[3])
    else:
        axes[3].text(0.5, 0.5, 'Model not found', ha='center', va='center', transform=axes[3].transAxes)
else:
    axes[3].text(0.5, 0.5, 'Best model not found', ha='center', va='center', transform=axes[3].transAxes)

plt.tight_layout()
plt.savefig("grafs/kdv_analysis_summary.png", dpi=200, bbox_inches='tight')
plt.show()

print(" All comparison plots and summary saved to 'grafs/' directory.")