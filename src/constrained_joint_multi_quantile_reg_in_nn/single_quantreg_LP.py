import numpy as np
from scipy.optimize import linprog

# Dataset
X = np.array([10]*10 + [30]*10 + [50]*10)
y = np.array([10,11,20,30,40,50,60,70,80,90] +  # X=10
             [25,26,30,35,40,45,50,55,60,65] +  # X=30
             [50,45,40,35,30,25,20,15,10,5])     # X=50

# Quantile regression via linear programming
def quantile_regression(X, y, tau):
    n = len(y)
    # Variables: [beta0, beta1, u_plus, u_minus]
    c = np.concatenate([np.zeros(2), tau*np.ones(n), (1-tau)*np.ones(n)])
    
    # Constraints: y = beta0 + beta1*x + u_plus - u_minus
    A_eq = np.column_stack([np.ones(n), X, np.eye(n), -np.eye(n)])
    b_eq = y
    
    # Bounds: u_plus, u_minus >= 0
    bounds = [(None, None), (None, None)] + [(0, None)]*(2*n)
    
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    return result.x[0], result.x[1]  # beta0, beta1

# Calculate quantiles
beta0_10, beta1_10 = quantile_regression(X, y, 0.10)
beta0_11, beta1_11 = quantile_regression(X, y, 0.15)

print(f"10% Quantile: y = {beta1_10:.6f}x + {beta0_10:.6f}")
print(f"15% Quantile: y = {beta1_11:.6f}x + {beta0_11:.6f}")