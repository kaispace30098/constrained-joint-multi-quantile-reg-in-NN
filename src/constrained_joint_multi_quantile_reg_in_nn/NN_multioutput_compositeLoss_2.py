import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA PREPARATION (Toy Dataset)
# ══════════════════════════════════════════════════════════════════════════════
# Raw data
X_raw = np.array([0.2095, 0.6809, 1.2936, 1.8535, 2.3583, 2.4368, 2.8754,
              4.1162, 4.567, 4.7146, 4.8946, 4.9042, 5.8864, 6.205,
              6.3962, 7.5324, 7.7828, 8.4835, 9.4854, 9.9582])
Y_raw = np.array([1.77268, 2.529927, 2.00102, 2.101015, 2.494044, 2.164226,
              2.44769, 2.574242, 4.314459, 1.569597, 2.467982, 2.153414,
              1.925144, 1.679639, 4.556762, 3.509959, 3.522292, 3.099583,
              0.368901, 3.069406])

# Convert to PyTorch tensors (Float64 for high precision)
X_tensor = torch.tensor(X_raw, dtype=torch.float64).reshape(-1, 1)
Y_tensor = torch.tensor(Y_raw, dtype=torch.float64).reshape(-1, 1)

# Define Quantiles
QUANTILES = torch.tensor([0.10, 0.15], dtype=torch.float64)

# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL & LOSS DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

class QuantileModel(nn.Module):
    def __init__(self, num_quantiles):
        super().__init__()
        # Linear Model: Y = X * Beta + Intercept
        self.linear = nn.Linear(1, num_quantiles, bias=True).double()

        # Initialize close to mean to speed up convergence
        nn.init.constant_(self.linear.bias, np.mean(Y_raw))
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.1)

    def forward(self, x):
        return self.linear(x)

def asymmetric_huber_loss(preds, targets, quantiles, delta=0.05):
    """
    Asymmetric Huberized Pinball Loss.
    Provides C1 continuity (smooth gradients) for L-BFGS.
    """
    errors = targets - preds
    abs_errors = torch.abs(errors)

    # Huber Smoothing
    quadratic = 0.5 * errors**2 / delta
    linear = abs_errors - 0.5 * delta
    huber = torch.where(abs_errors <= delta, quadratic, linear)

    # Asymmetric Weighting
    q_broadcast = quantiles.view(1, -1)
    weights = torch.where(errors >= 0, q_broadcast, 1.0 - q_broadcast)

    return torch.mean(weights * huber)

def squared_relu_penalty(preds, margin=1e-3):
    """
    Squared ReLU Penalty with Safety Margin.
    Enforces q_{i+1} >= q_{i} + margin.
    """
    diffs = preds[:, 1:] - preds[:, :-1]

    # Violation: if diff is smaller than margin
    violation = margin - diffs

    # Squared ReLU: (max(0, violation))^2
    penalty = torch.mean(torch.relu(violation) ** 2)

    return penalty

# ══════════════════════════════════════════════════════════════════════════════
# 3. TRAINING ENGINE (WITH BOUNDARY ANCHORS)
# ══════════════════════════════════════════════════════════════════════════════

def train_model(mode="fixed", epochs=200, learning_rate=1.0, fixed_lambda=None):
    print(f"\n--- STARTING TRAINING MODE: {mode.upper()} ---")

    model = QuantileModel(num_quantiles=len(QUANTILES))
    optimizer = optim.LBFGS(
        model.parameters(),
        lr=learning_rate,
        max_iter=50,
        tolerance_grad=1e-9,      # High precision
        tolerance_change=1e-11,   # High precision
        line_search_fn='strong_wolfe'
    )

    # --- ANCHOR POINTS ---
    # We enforce monotonicity not just on data, but at the boundaries.
    # This prevents "Out-of-Sample Crossing" (e.g., at X=0).
    X_anchors = torch.tensor([[0.0], [12.0]], dtype=torch.float64)

    history = []

    for epoch in range(epochs):
        def closure():
            optimizer.zero_grad()

            # A. Main Data Flow
            preds_data = model(X_tensor)
            loss_fit = asymmetric_huber_loss(preds_data, Y_tensor, QUANTILES, delta=0.05)
            loss_mono_data = squared_relu_penalty(preds_data, margin=1e-3)

            # B. Anchor Flow (Constraint only, no fit loss)
            preds_anchors = model(X_anchors)
            loss_mono_anchors = squared_relu_penalty(preds_anchors, margin=1e-3)

            # Combine Penalties
            loss_mono_total = loss_mono_data + loss_mono_anchors

            # C. Lambda Logic
            if mode == "adaptive":
                # Decay 22.0 -> 3.0
                progress = min(1.0, epoch / epochs)
                lambda_val = 20.0 * np.exp(-3.0 * progress) + 2.0
            else:
                lambda_val = fixed_lambda

            # Total Loss
            total_loss = loss_fit + (lambda_val * loss_mono_total)
            total_loss.backward()

            history.append({
                'epoch': epoch,
                'fit_loss': loss_fit.item(),
                'mono_loss': loss_mono_total.item(),
                'lambda': lambda_val,
                'total': total_loss.item()
            })
            return total_loss

        optimizer.step(closure)

        if epoch % 50 == 0:
            last_log = history[-1]
            print(f"Ep {epoch}: Fit={last_log['fit_loss']:.5f} | "
                  f"Mono={last_log['mono_loss']:.8f} | "
                  f"λ={last_log['lambda']:.2f}")

    return model, history

# ══════════════════════════════════════════════════════════════════════════════
# 4. EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

# Phase 1: Adaptive
print("Phase 1: Running Adaptive Lambda...")
model_adaptive, hist_adaptive = train_model(mode="adaptive", epochs=300)
optimal_lambda = hist_adaptive[-1]['lambda']
print(f"\nDiscovered Optimal Lambda: {optimal_lambda:.4f}")

# Phase 2: Production (Fixed Lambda + Anchors active)
print(f"\nPhase 2: Running Production Model...")
model_fixed, hist_fixed = train_model(mode="fixed", epochs=100, fixed_lambda=optimal_lambda)

# ══════════════════════════════════════════════════════════════════════════════
# 5. FINAL VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

weights = model_fixed.linear.weight.detach().numpy().flatten()
bias = model_fixed.linear.bias.detach().numpy().flatten()

print("\n" + "="*60)
print("FINAL RESULTS (Global Validity Check)")
print("="*60)
print(f"{'Quantile':<10} {'Intercept (Beta 0)':<20} {'Slope (Beta 1)':<20}")
print("-" * 60)
print(f"0.10       {bias[0]:<20.5f} {weights[0]:<20.5f}")
print(f"0.15       {bias[1]:<20.5f} {weights[1]:<20.5f}")
print("-" * 60)

# Check Crossing at critical points
q10_x0 = bias[0]
q15_x0 = bias[1]
diff_x0 = q15_x0 - q10_x0

print("\nCROSSING CHECK (At X=0):")
print(f"Q10: {q10_x0:.6f}")
print(f"Q15: {q15_x0:.6f}")
print(f"Diff: {diff_x0:.6f}")

if diff_x0 >= 0:
    print("RESULT: ✅ OK (Global Validity Achieved)")
else:
    print("RESULT: ❌ CROSSING DETECTED")

final_mono_loss = hist_fixed[-1]['mono_loss']
print(f"\nFinal Penalty: {final_mono_loss:.10f}")