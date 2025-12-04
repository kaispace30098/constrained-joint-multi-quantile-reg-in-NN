"""
══════════════════════════════════════════════════════════════════════════════════
AUGMENTED LAGRANGIAN METHOD: PAPER DEMONSTRATION
"Learning from Weakness" - Showing Adaptive Rho
══════════════════════════════════════════════════════════════════════════════════

CONFIGURATION:
- Seed: 123 (Forces initial crossing)
- Rho Init: 0.01 (Starts very weak to force adaptation)
- Update Logic: Standard (Sufficient Decrease)

This script generates the 4-panel diagnostic plot showing the "staircase"
increase of rho, which is ideal for academic figures.

Author: Kaihua Chang, Arizona Department of Education
══════════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# 1. SETUP
# ══════════════════════════════════════════════════════════════════════════════

# Seed 123 guarantees initial crossing for this dataset
torch.manual_seed(123)
np.random.seed(123)

# 20-point Heteroscedastic Dataset
X_raw = np.array([0.2095, 0.6809, 1.2936, 1.8535, 2.3583, 2.4368, 2.8754,
                  4.1162, 4.567, 4.7146, 4.8946, 4.9042, 5.8864, 6.205,
                  6.3962, 7.5324, 7.7828, 8.4835, 9.4854, 9.9582])

Y_raw = np.array([1.77268, 2.529927, 2.00102, 2.101015, 2.494044, 2.164226,
                  2.44769, 2.574242, 4.314459, 1.569597, 2.467982, 2.153414,
                  1.925144, 1.679639, 4.556762, 3.509959, 3.522292, 3.099583,
                  0.368901, 3.069406])

X_tensor = torch.tensor(X_raw, dtype=torch.float64).reshape(-1, 1)
Y_tensor = torch.tensor(Y_raw, dtype=torch.float64).reshape(-1, 1)
QUANTILES = torch.tensor([0.10, 0.15], dtype=torch.float64)

print("="*70)
print("ALM PAPER DEMO: ADAPTIVE PENALTY VISUALIZATION")
print("="*70)

# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

class LinearQuantileModel(nn.Module):
    def __init__(self, num_quantiles):
        super().__init__()
        self.linear = nn.Linear(1, num_quantiles, bias=True).double()
        # Initialize to mean to start neutral
        nn.init.constant_(self.linear.bias, np.mean(Y_raw))
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)

    def forward(self, x):
        return self.linear(x)

def huberized_pinball_loss(preds, targets, quantiles, delta=0.05):
    errors = targets - preds
    abs_errors = torch.abs(errors)
    quadratic = 0.5 * errors**2 / delta
    linear = abs_errors - 0.5 * delta
    huber = torch.where(abs_errors <= delta, quadratic, linear)
    q_broadcast = quantiles.view(1, -1)
    weights = torch.where(errors >= 0, q_broadcast, 1.0 - q_broadcast)
    return torch.mean(weights * huber)

# ══════════════════════════════════════════════════════════════════════════════
# 3. TRAINING LOOP (With Weak Start)
# ══════════════════════════════════════════════════════════════════════════════

def train_alm_paper_demo(X, Y, quantiles, max_outer=50):
    model = LinearQuantileModel(len(quantiles))
    n_samples = len(X)

    # --- PAPER CONFIGURATION ---
    rho = 0.01          # Start WEAK to force crossing
    rho_max = 100000.0
    rho_increase = 4.0  # Aggressive steps (staircase effect)
    gamma = 0.9         # Standard sufficient decrease check
    margin = 1e-4
    tol = 1e-6
    # ---------------------------

    mu = torch.zeros(n_samples, dtype=torch.float64)
    prev_max_viol = float('inf')

    history = {
        'outer_iter': [], 'fit_loss': [], 'max_violation': [],
        'mu_max': [], 'rho': []
    }

    print(f"{'Iter':<5} {'ρ':<10} {'MaxViol':<12} {'Status':<15}")
    print("-" * 50)

    for outer_k in range(max_outer):

        # Inner Optimization
        optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=50, line_search_fn='strong_wolfe')
        current_mu = mu.clone()
        current_rho = rho

        def closure():
            optimizer.zero_grad()
            preds = model(X)
            fit_loss = huberized_pinball_loss(preds, Y, quantiles)
            diffs = preds[:, 1] - preds[:, 0]
            violation = torch.relu(margin - diffs)
            lagrange = torch.sum(current_mu * violation) / n_samples
            penalty = (current_rho / 2.0) * torch.mean(violation ** 2)
            total = fit_loss + lagrange + penalty
            total.backward()
            return total

        optimizer.step(closure)

        # Outer Update
        with torch.no_grad():
            preds = model(X)
            diffs = preds[:, 1] - preds[:, 0]
            violation_pos = torch.relu(margin - diffs)
            max_viol = violation_pos.max().item()
            fit_loss_val = huberized_pinball_loss(preds, Y, quantiles).item()

            # 1. Update Multipliers
            mu = mu + rho * violation_pos

            # 2. Check Convergence
            if max_viol <= tol:
                history['outer_iter'].append(outer_k)
                history['fit_loss'].append(fit_loss_val)
                history['max_violation'].append(max_viol)
                history['mu_max'].append(mu.max().item())
                history['rho'].append(rho)
                print(f"{outer_k:<5} {rho:<10.4f} {max_viol:<12.1e} ✅ Converged")
                break

            # 3. Update Rho (Standard Logic)
            status = "ρ constant"
            # If violation is still > 90% of previous, we are stuck -> Increase Rho
            if max_viol > gamma * prev_max_viol:
                if rho < rho_max:
                    rho *= rho_increase
                    status = f"↑ρ to {rho:.2f}"

            prev_max_viol = max_viol

            print(f"{outer_k:<5} {rho:<10.4f} {max_viol:<12.1e} {status:<15}")

            history['outer_iter'].append(outer_k)
            history['fit_loss'].append(fit_loss_val)
            history['max_violation'].append(max_viol)
            history['mu_max'].append(mu.max().item())
            history['rho'].append(rho)

    return model, history

# ══════════════════════════════════════════════════════════════════════════════
# 4. EXECUTE
# ══════════════════════════════════════════════════════════════════════════════

model, history = train_alm_paper_demo(X_tensor, Y_tensor, QUANTILES)

# Report Coefficients
beta0 = model.linear.bias.detach().numpy().flatten()
beta1 = model.linear.weight.detach().numpy().flatten()

print("\n" + "="*70)
print("FINAL COEFFICIENTS")
print("="*70)
print(f"{'Quantile':<10} | {'Intercept (β0)':<15} | {'Slope (β1)':<15}")
print("-" * 45)
print(f"{'0.10':<10} | {beta0[0]:<15.5f} | {beta1[0]:<15.5f}")
print(f"{'0.15':<10} | {beta0[1]:<15.5f} | {beta1[1]:<15.5f}")
print("-" * 45)

# ══════════════════════════════════════════════════════════════════════════════
# 5. DIAGNOSTIC PLOTS
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# A. Regression Lines
ax1 = axes[0, 0]
x_plot = np.linspace(X_raw.min(), X_raw.max(), 100)
y_q10 = beta0[0] + beta1[0] * x_plot
y_q15 = beta0[1] + beta1[1] * x_plot
ax1.scatter(X_raw, Y_raw, c='black', s=50, label='Data')
ax1.plot(x_plot, y_q10, 'b-', lw=2, label='τ=0.10')
ax1.plot(x_plot, y_q15, 'r-', lw=2, label='τ=0.15')
ax1.fill_between(x_plot, y_q10, y_q15, color='green', alpha=0.1, label='Gap')
ax1.set_title('Non-Crossing Solution')
ax1.legend()
ax1.grid(True, alpha=0.3)

# B. Violation Convergence
ax2 = axes[0, 1]
ax2.semilogy(history['outer_iter'], history['max_violation'], 'b-o')
ax2.axhline(y=1e-6, color='r', ls='--', label='Tolerance')
ax2.set_title('Constraint Violation (Log Scale)')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Max Violation')
ax2.grid(True, alpha=0.3)

# C. Multipliers
ax3 = axes[1, 0]
ax3.plot(history['outer_iter'], history['mu_max'], 'g-s')
ax3.set_title('Max Lagrange Multiplier (μ)')
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Value')
ax3.grid(True, alpha=0.3)

# D. Adaptive Rho vs Loss
ax4 = axes[1, 1]
ax4_twin = ax4.twinx()
ln1 = ax4.plot(history['outer_iter'], history['rho'], 'r-o', label='ρ (Penalty)')
ln2 = ax4_twin.plot(history['outer_iter'], history['fit_loss'], 'b-x', label='Fit Loss')
ax4.set_xlabel('Iteration')
ax4.set_ylabel('ρ (Stepwise Increase)', color='red')
ax4_twin.set_ylabel('Fit Loss', color='blue')
ax4.set_title('Adaptive Penalty & Loss Trade-off')
# Merge legends
lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax4.legend(lns, labs, loc='center right')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('alm_paper_demo.png', dpi=150)
plt.show()

print("\n✅ Plot saved to 'alm_paper_demo.png'")