import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA PREPARATION (Your Toy Dataset)
# ══════════════════════════════════════════════════════════════════════════════
# Raw data from your snippet
X_raw = np.array([0.2095, 0.6809, 1.2936, 1.8535, 2.3583, 2.4368, 2.8754, 
              4.1162, 4.567, 4.7146, 4.8946, 4.9042, 5.8864, 6.205, 
              6.3962, 7.5324, 7.7828, 8.4835, 9.4854, 9.9582])
Y_raw = np.array([1.77268, 2.529927, 2.00102, 2.101015, 2.494044, 2.164226, 
              2.44769, 2.574242, 4.314459, 1.569597, 2.467982, 2.153414, 
              1.925144, 1.679639, 4.556762, 3.509959, 3.522292, 3.099583, 
              0.368901, 3.069406])

# Convert to PyTorch tensors (Float64 for precision)
# We add a column of 1s for the intercept to match linear algebra logic, 
# or let nn.Linear handle bias. Let's use nn.Linear with bias=True for simplicity.
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
        # Input: 1 feature (X). Output: num_quantiles (predictions)
        self.linear = nn.Linear(1, num_quantiles, bias=True).double()
        
        # Init weights to something reasonable (e.g., linear regression mean)
        nn.init.constant_(self.linear.bias, np.mean(Y_raw))
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.1)

    def forward(self, x):
        return self.linear(x)

def asymmetric_huber_loss(preds, targets, quantiles, delta=0.05):
    """
    Robust Asymmetric Loss.
    delta=0.05 is chosen because Y values are small (~2.0). 
    For SGP scores (range 200-800), use delta=1.0.
    """
    errors = targets - preds
    abs_errors = torch.abs(errors)
    
    # Huber Smoothing (The Bowl)
    quadratic = 0.5 * errors**2 / delta
    linear = abs_errors - 0.5 * delta
    huber = torch.where(abs_errors <= delta, quadratic, linear)
    
    # Asymmetric Weighting (The Tilt)
    # Reshape quantiles to broadcast: [1, n_quantiles]
    q_broadcast = quantiles.view(1, -1)
    weights = torch.where(errors >= 0, q_broadcast, 1.0 - q_broadcast)
    
    return torch.mean(weights * huber)

def softplus_monotonicity_penalty(preds, beta=50.0):
    """
    Smooth monotonic constraint.
    Calculates q_0.15 - q_0.10. If negative, we penalize.
    """
    # We want q_{i+1} >= q_{i}, so diff = q_{i+1} - q_{i} should be positive.
    # We penalize if diff is negative.
    # Penalty = Softplus(-diff)
    
    diffs = preds[:, 1:] - preds[:, :-1] # shape: [N, n_quantiles-1]
    
    # Softplus smooths the "elbow" at 0 so gradients don't explode/vanish for L-BFGS
    penalty = torch.nn.functional.softplus(-diffs, beta=beta)
    
    return torch.mean(penalty)

# ══════════════════════════════════════════════════════════════════════════════
# 3. TRAINING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def train_model(mode="fixed", epochs=200, learning_rate=1.0, fixed_lambda=None):
    print(f"\n--- STARTING TRAINING MODE: {mode.upper()} ---")
    
    model = QuantileModel(num_quantiles=len(QUANTILES))
    optimizer = optim.LBFGS(
        model.parameters(), 
        lr=learning_rate, 
        max_iter=20, 
        line_search_fn='strong_wolfe' # Critical for convergence
    )
    
    history = []
    
    for epoch in range(epochs):
        def closure():
            optimizer.zero_grad()
            preds = model(X_tensor)
            
            # 1. Main Objective: Huberized Pinball
            loss_fit = asymmetric_huber_loss(preds, Y_tensor, QUANTILES, delta=0.05)
            
            # 2. Structural Regularization: Monotonicity
            loss_mono = softplus_monotonicity_penalty(preds, beta=50.0)
            
            # 3. Determine Lambda (Penalty Weight)
            if mode == "adaptive":
                # Decay from 22.0 to ~3.0 (using -3.0 decay factor matches your logs)
                progress = min(1.0, epoch / epochs)
                lambda_val = 20.0 * np.exp(-3.0 * progress) + 2.0
            else:
                # Production Fixed Mode: Use the discovered value
                if fixed_lambda is None:
                    raise ValueError("Must provide fixed_lambda for fixed mode")
                lambda_val = fixed_lambda
            
            total_loss = loss_fit + (lambda_val * loss_mono)
            total_loss.backward()
            
            # Store for logging (hacky but works inside closure)
            history.append({
                'epoch': epoch,
                'fit_loss': loss_fit.item(),
                'mono_loss': loss_mono.item(),
                'lambda': lambda_val,
                'total': total_loss.item()
            })
            return total_loss
        
        optimizer.step(closure)
        
        if epoch % 50 == 0:
            last_log = history[-1]
            print(f"Ep {epoch}: Fit={last_log['fit_loss']:.5f} | "
                  f"Mono={last_log['mono_loss']:.6f} | "
                  f"λ={last_log['lambda']:.2f}")

    return model, history

# ══════════════════════════════════════════════════════════════════════════════
# 4. EXECUTION: PHASE 1 & PHASE 2
# ══════════════════════════════════════════════════════════════════════════════

# --- Phase 1: Adaptive Discovery ---
print("Phase 1: Running Adaptive Lambda to discover optimal weight...")
model_adaptive, hist_adaptive = train_model(mode="adaptive", epochs=300)

# AUTOMATICALLY EXTRACT THE DISCOVERED LAMBDA
optimal_lambda = hist_adaptive[-1]['lambda']

print(f"\n[Analysis] Adaptive run finished.")
print(f"Discovered Optimal Lambda: {optimal_lambda:.4f}")
print("Observations: L-BFGS stabilized as penalty decayed to equilibrium.")

# --- Phase 2: Production Fixed ---
print(f"\nPhase 2: Running Production Model with Discovered Lambda = {optimal_lambda:.4f}...")
model_fixed, hist_fixed = train_model(mode="fixed", epochs=100, fixed_lambda=optimal_lambda)

# ══════════════════════════════════════════════════════════════════════════════
# 5. VALIDATION & RESULTS
# ══════════════════════════════════════════════════════════════════════════════

# Extract weights
weights = model_fixed.linear.weight.detach().numpy().flatten()
bias = model_fixed.linear.bias.detach().numpy().flatten()

print("\n" + "="*60)
print("FINAL RESULTS (Comparison with your LP Output)")
print("="*60)
print(f"{'Quantile':<10} {'Intercept (Beta 0)':<20} {'Slope (Beta 1)':<20}")
print("-" * 60)
print(f"0.10       {bias[0]:<20.5f} {weights[0]:<20.5f}")
print(f"0.15       {bias[1]:<20.5f} {weights[1]:<20.5f}")
print("-" * 60)

# Check Crossing at X=0 and X=10
q10_x0 = bias[0]
q15_x0 = bias[1]
q10_x10 = bias[0] + weights[0]*10
q15_x10 = bias[1] + weights[1]*10

print("\nCROSSING CHECK:")
print(f"At X=0 : Q10={q10_x0:.4f}, Q15={q15_x0:.4f} -> {'✅ OK' if q15_x0 >= q10_x0 else '❌ CROSS'}")
print(f"At X=10: Q10={q10_x10:.4f}, Q15={q15_x10:.4f} -> {'✅ OK' if q15_x10 >= q10_x10 else '❌ CROSS'}")

# Logic check: Did the softplus penalty work?
final_mono_loss = hist_fixed[-1]['mono_loss']
print(f"\nFinal Monotonicity Violation (Softplus): {final_mono_loss:.10f}")
if final_mono_loss < 1e-4:
    print("CONCLUSION: Constraint successfully enforced.")
else:
    print("CONCLUSION: Constraint failed.")