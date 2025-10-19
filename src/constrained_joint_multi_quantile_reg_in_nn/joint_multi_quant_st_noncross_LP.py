import numpy as np
from scipy.optimize import linprog

# Data
X = np.array([0.2095, 0.6809, 1.2936, 1.8535, 2.3583, 2.4368, 2.8754, 
              4.1162, 4.567, 4.7146, 4.8946, 4.9042, 5.8864, 6.205, 
              6.3962, 7.5324, 7.7828, 8.4835, 9.4854, 9.9582])
Y = np.array([1.77268, 2.529927, 2.00102, 2.101015, 2.494044, 2.164226, 
              2.44769, 2.574242, 4.314459, 1.569597, 2.467982, 2.153414, 
              1.925144, 1.679639, 4.556762, 3.509959, 3.522292, 3.099583, 
              0.368901, 3.069406])

def joint_quantile_regression_no_crossing(X, Y, tau1=0.10, tau2=0.15):
    """
    Joint quantile regression with non-crossing constraints
    
    Variables: [β₀⁽¹⁾, β₁⁽¹⁾, u⁺⁽¹⁾, u⁻⁽¹⁾, β₀⁽²⁾, β₁⁽²⁾, u⁺⁽²⁾, u⁻⁽²⁾]
    
    Objective: 
      tau1*sum(u⁺⁽¹⁾) + (1-tau1)*sum(u⁻⁽¹⁾) + tau2*sum(u⁺⁽²⁾) + (1-tau2)*sum(u⁻⁽²⁾)
    
    Equality constraints:
      y = β₀⁽ᵗ⁾ + β₁⁽ᵗ⁾*x + u⁺⁽ᵗ⁾ - u⁻⁽ᵗ⁾  for each quantile t
    
    Non-crossing constraints (at each x_i):
      β₀⁽²⁾ + β₁⁽²⁾*x_i >= β₀⁽¹⁾ + β₁⁽¹⁾*x_i
      => (β₀⁽²⁾ - β₀⁽¹⁾) + (β₁⁽²⁾ - β₁⁽¹⁾)*x_i >= 0
    """
    n = len(Y)
    
    # Variable structure:
    # [β₀⁽¹⁾, β₁⁽¹⁾, u⁺₁⁽¹⁾, ..., u⁺ₙ⁽¹⁾, u⁻₁⁽¹⁾, ..., u⁻ₙ⁽¹⁾,
    #  β₀⁽²⁾, β₁⁽²⁾, u⁺₁⁽²⁾, ..., u⁺ₙ⁽²⁾, u⁻₁⁽²⁾, ..., u⁻ₙ⁽²⁾]
    
    n_vars_per_quantile = 2 + 2*n  # 2 betas + n u+ + n u-
    n_total_vars = 2 * n_vars_per_quantile
    
    # Objective function
    c = np.zeros(n_total_vars)
    # First quantile (tau1)
    c[2:2+n] = tau1          # u⁺⁽¹⁾
    c[2+n:2+2*n] = (1-tau1)  # u⁻⁽¹⁾
    # Second quantile (tau2)
    offset = n_vars_per_quantile
    c[offset+2:offset+2+n] = tau2          # u⁺⁽²⁾
    c[offset+2+n:offset+2+2*n] = (1-tau2)  # u⁻⁽²⁾
    
    # === EQUALITY CONSTRAINTS ===
    # For quantile 1: y_i = β₀⁽¹⁾ + β₁⁽¹⁾*x_i + u⁺ᵢ⁽¹⁾ - u⁻ᵢ⁽¹⁾
    A_eq_1 = np.zeros((n, n_total_vars))
    for i in range(n):
        A_eq_1[i, 0] = 1      # β₀⁽¹⁾
        A_eq_1[i, 1] = X[i]   # β₁⁽¹⁾
        A_eq_1[i, 2+i] = 1    # u⁺ᵢ⁽¹⁾
        A_eq_1[i, 2+n+i] = -1 # u⁻ᵢ⁽¹⁾
    
    # For quantile 2: y_i = β₀⁽²⁾ + β₁⁽²⁾*x_i + u⁺ᵢ⁽²⁾ - u⁻ᵢ⁽²⁾
    A_eq_2 = np.zeros((n, n_total_vars))
    for i in range(n):
        A_eq_2[i, offset+0] = 1      # β₀⁽²⁾
        A_eq_2[i, offset+1] = X[i]   # β₁⁽²⁾
        A_eq_2[i, offset+2+i] = 1    # u⁺ᵢ⁽²⁾
        A_eq_2[i, offset+2+n+i] = -1 # u⁻ᵢ⁽²⁾
    
    A_eq = np.vstack([A_eq_1, A_eq_2])
    b_eq = np.hstack([Y, Y])
    
    # === INEQUALITY CONSTRAINTS (Non-crossing) ===
    # At each x_i: β₀⁽²⁾ + β₁⁽²⁾*x_i >= β₀⁽¹⁾ + β₁⁽¹⁾*x_i
    # Rearranged: -β₀⁽¹⁾ - β₁⁽¹⁾*x_i + β₀⁽²⁾ + β₁⁽²⁾*x_i >= 0
    # For linprog (A_ub @ x <= b_ub), multiply by -1:
    # β₀⁽¹⁾ + β₁⁽¹⁾*x_i - β₀⁽²⁾ - β₁⁽²⁾*x_i <= 0
    A_ub = np.zeros((n, n_total_vars))
    for i in range(n):
        A_ub[i, 0] = 1           # β₀⁽¹⁾
        A_ub[i, 1] = X[i]        # β₁⁽¹⁾
        A_ub[i, offset+0] = -1   # -β₀⁽²⁾
        A_ub[i, offset+1] = -X[i] # -β₁⁽²⁾
    b_ub = np.zeros(n)
    
    # Bounds: betas unbounded, u⁺ >= 0, u⁻ >= 0
    bounds = [(None, None)] * 2 + [(0, None)] * (2*n)  # First quantile
    bounds += [(None, None)] * 2 + [(0, None)] * (2*n)  # Second quantile
    
    # Solve
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                     bounds=bounds, method='highs')
    
    if result.success:
        beta1 = result.x[0:2]
        beta2 = result.x[offset:offset+2]
        u_plus_1 = result.x[2:2+n]
        u_minus_1 = result.x[2+n:2+2*n]
        u_plus_2 = result.x[offset+2:offset+2+n]
        u_minus_2 = result.x[offset+2+n:offset+2+2*n]
        return beta1, beta2, u_plus_1, u_minus_1, u_plus_2, u_minus_2, result
    else:
        raise ValueError(f"Optimization failed: {result.message}")

# Run joint quantile regression with non-crossing constraints
print("="*70)
print("JOINT QUANTILE REGRESSION WITH NON-CROSSING CONSTRAINTS")
print("="*70)

beta_10, beta_15, up1, um1, up2, um2, res = joint_quantile_regression_no_crossing(X, Y, 0.10, 0.15)

print(f"\n10% Quantile (τ = 0.10):")
print(f"  β₀ = {beta_10[0]:.6f}")
print(f"  β₁ = {beta_10[1]:.6f}")
print(f"  Sum(U+) = {up1.sum():.6f}")
print(f"  Sum(U-) = {um1.sum():.6f}")

print(f"\n15% Quantile (τ = 0.15):")
print(f"  β₀ = {beta_15[0]:.6f}")
print(f"  β₁ = {beta_15[1]:.6f}")
print(f"  Sum(U+) = {up2.sum():.6f}")
print(f"  Sum(U-) = {um2.sum():.6f}")

print(f"\nOptimization Status: {res.message}")
print(f"Total Objective Value: {res.fun:.6f}")

print("\n" + "="*70)
print("FITTED FORMULAS (NO CROSSING)")
print("="*70)
print(f"\n10% Quantile Formula:")
print(f"  Y = {beta_10[0]:.6f} + {beta_10[1]:.6f} * X")

print(f"\n15% Quantile Formula:")
print(f"  Y = {beta_15[0]:.6f} + {beta_15[1]:.6f} * X")

# Verify no crossing at key points
print("\n" + "="*70)
print("VERIFICATION: Quantile Values at Data Points")
print("="*70)
print(f"{'X':<10} {'Q(0.10)':<12} {'Q(0.15)':<12} {'Difference':<12} {'Valid?'}")
print("-"*70)
for i in [0, len(X)//2, len(X)-1]:  # Check first, middle, last
    q10 = beta_10[0] + beta_10[1] * X[i]
    q15 = beta_15[0] + beta_15[1] * X[i]
    diff = q15 - q10
    valid = "✓" if diff >= -1e-6 else "✗"
    print(f"{X[i]:<10.4f} {q10:<12.6f} {q15:<12.6f} {diff:<12.6f} {valid}")

print("="*70)