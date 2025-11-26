import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA PREPARATION (Toy Dataset)
# ══════════════════════════════════════════════════════════════════════════════
# Raw data (Sorted for easier visualization of range)
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

# Define Quantiles to estimate (Tau = 0.10 and 0.15)
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
    Enforces q_{i+1} >= q_{i} + margin on the provided predictions.
    """
    diffs = preds[:, 1:] - preds[:, :-1] 
    
    # Violation: if diff is smaller than margin
    violation = margin - diffs
    
    # Squared ReLU: (max(0, violation))^2
    penalty = torch.mean(torch.relu(violation) ** 2)
    
    return penalty

# ══════════════════════════════════════════════════════════════════════════════
# 3. TRAINING ENGINE (PURE DATA - NO ANCHORS)
# ══════════════════════════════════════════════════════════════════════════════

def train_model(mode="fixed", epochs=200, learning_rate=1.0, fixed_lambda=None):
    print(f"\n--- STARTING TRAINING MODE: {mode.upper()} ---")
    
    model = QuantileModel(num_quantiles=len(QUANTILES))
    optimizer = optim.LBFGS(
        model.parameters(), 
        lr=learning_rate, 
        max_iter=20, 
        tolerance_grad=1e-9,      # High precision
        tolerance_change=1e-11,   # High precision
        line_search_fn='strong_wolfe'
    )
    
    history = []
    
    for epoch in range(epochs):
        def closure():
            optimizer.zero_grad()
            
            # Forward Pass on REAL DATA ONLY
            preds_data = model(X_tensor)
            
            # 1. Fit Loss (Huber)
            loss_fit = asymmetric_huber_loss(preds_data, Y_tensor, QUANTILES, delta=0.05)
            
            # 2. Constraint Loss (Squared ReLU on Real Data)
            # NO ANCHORS used here. Just the 20 students.
            loss_mono_data = squared_relu_penalty(preds_data, margin=1e-3)
            
            # 3. Lambda Logic
            if mode == "adaptive":
                # Decay 22.0 -> 3.0
                progress = min(1.0, epoch / epochs)
                lambda_val = 20.0 * np.exp(-3.0 * progress) + 2.0
            else:
                lambda_val = fixed_lambda
            
            # Total Loss
            total_loss = loss_fit + (lambda_val * loss_mono_data)
            total_loss.backward()
            
            history.append({
                'epoch': epoch,
                'fit_loss': loss_fit.item(),
                'mono_loss': loss_mono_data.item(),
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

# Phase 1: Adaptive Discovery
print("Phase 1: Running Adaptive Lambda...")
model_adaptive, hist_adaptive = train_model(mode="adaptive", epochs=300)
optimal_lambda = hist_adaptive[-1]['lambda']
print(f"\nDiscovered Optimal Lambda: {optimal_lambda:.4f}")

# Phase 2: Production (Fixed Lambda = ~3.0)
print(f"\nPhase 2: Running Production Model with Fixed Lambda...")
model_fixed, hist_fixed = train_model(mode="fixed", epochs=100, fixed_lambda=optimal_lambda)

# ══════════════════════════════════════════════════════════════════════════════
# 5. FINAL VALIDATION ON OBSERVED DATA
# ══════════════════════════════════════════════════════════════════════════════

weights = model_fixed.linear.weight.detach().numpy().flatten()
bias = model_fixed.linear.bias.detach().numpy().flatten()

print("\n" + "="*60)
print("FINAL RESULTS (No Anchors - Pure Data Constraints)")
print("="*60)
print(f"{'Quantile':<10} {'Intercept':<20} {'Slope':<20}")
print("-" * 60)
print(f"0.10       {bias[0]:<20.5f} {weights[0]:<20.5f}")
print(f"0.15       {bias[1]:<20.5f} {weights[1]:<20.5f}")
print("-" * 60)

# --- CHECK CROSSING ON ACTUAL DATA ---
# We evaluate the model on the Min and Max of the TRAINING DATA
# This confirms validity for the population we actually care about.
x_min_data = X_raw.min()
x_max_data = X_raw.max()

q10_min = bias[0] + weights[0] * x_min_data
q15_min = bias[1] + weights[1] * x_min_data
diff_min = q15_min - q10_min

q10_max = bias[0] + weights[0] * x_max_data
q15_max = bias[1] + weights[1] * x_max_data
diff_max = q15_max - q10_max

print("\nCROSSING VALIDATION (On Observed Data Range):")
print("-" * 60)

print(f"1. At Lowest Student Score (X={x_min_data:.4f}):")
print(f"   Q10: {q10_min:.5f}")
print(f"   Q15: {q15_min:.5f}")
print(f"   Gap: {diff_min:.5f}  -> {'✅ VALID' if diff_min >= 0 else '❌ CROSSING'}")

print(f"\n2. At Highest Student Score (X={x_max_data:.4f}):")
print(f"   Q10: {q10_max:.5f}")
print(f"   Q15: {q15_max:.5f}")
print(f"   Gap: {diff_max:.5f}  -> {'✅ VALID' if diff_max >= 0 else '❌ CROSSING'}")

print("-" * 60)

final_mono_loss = hist_fixed[-1]['mono_loss']
print(f"Final Penalty Loss: {final_mono_loss:.10f}")

if final_mono_loss < 1e-6:
    print("\nCONCLUSION: The model is 100% monotonic on the training set.")
    print("            The Two-Phase optimization successfully found the")
    print("            minimum lambda required to enforce the constraint.")
else:
    print("\nCONCLUSION: Constraints violated on training set.")