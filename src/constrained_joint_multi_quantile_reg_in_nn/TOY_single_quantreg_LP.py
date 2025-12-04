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

def quantile_regression_lp(X, Y, tau):
    """
    Quantile regression using linear programming
    
    minimize: tau * sum(u+) + (1-tau) * sum(u-)
    subject to: y = X*beta + u+ - u-
                u+ >= 0, u- >= 0
    
    Variables: [beta_0, beta_1, u+_1, ..., u+_n, u-_1, ..., u-_n]
    """
    n = len(Y)
    
    # Design matrix (add intercept)
    X_mat = np.column_stack([np.ones(n), X])
    n_params = X_mat.shape[1]
    
    # Objective function coefficients
    c = np.zeros(n_params + 2*n)
    c[n_params:n_params+n] = tau      # u+ coefficients
    c[n_params+n:] = (1 - tau)        # u- coefficients
    
    # Equality constraints: X*beta + u+ - u- = y
    A_eq = np.hstack([X_mat, np.eye(n), -np.eye(n)])
    b_eq = Y
    
    # Bounds: beta unbounded, u+ >= 0, u- >= 0
    bounds = [(None, None)] * n_params + [(0, None)] * (2*n)
    
    # Solve
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if result.success:
        beta = result.x[:n_params]
        u_plus = result.x[n_params:n_params+n]
        u_minus = result.x[n_params+n:]
        return beta, u_plus, u_minus
    else:
        raise ValueError(f"Optimization failed: {result.message}")

# Fit 10% and 15% quantiles
print("="*60)
print("QUANTILE REGRESSION - LINEAR PROGRAMMING")
print("="*60)

# 10% Quantile
tau_10 = 0.10
beta_10, u_plus_10, u_minus_10 = quantile_regression_lp(X, Y, tau_10)

print(f"\n10% Quantile (τ = 0.10):")
print(f"  β₀ = {beta_10[0]:.6f}")
print(f"  β₁ = {beta_10[1]:.6f}")
print(f"  Sum(U+) = {u_plus_10.sum():.6f}")
print(f"  Sum(U-) = {u_minus_10.sum():.6f}")

# 15% Quantile
tau_15 = 0.15
beta_15, u_plus_15, u_minus_15 = quantile_regression_lp(X, Y, tau_15)

print(f"\n15% Quantile (τ = 0.15):")
print(f"  β₀ = {beta_15[0]:.6f}")
print(f"  β₁ = {beta_15[1]:.6f}")
print(f"  Sum(U+) = {u_plus_15.sum():.6f}")
print(f"  Sum(U-) = {u_minus_15.sum():.6f}")

print("\n" + "="*60)
print("FITTED FORMULAS")
print("="*60)
print(f"\n10% Quantile Formula:")
print(f"  Y = {beta_10[0]:.6f} + {beta_10[1]:.6f} * X")

print(f"\n15% Quantile Formula:")
print(f"  Y = {beta_15[0]:.6f} + {beta_15[1]:.6f} * X")
print("\n" + "="*60)