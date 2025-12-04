import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
# I Keep this file because it shows magically even shared layer can have some kind of knowledge to avoid crossing!
# --- SET RANDOM SEEDS FOR REPRODUCIBILITY ---
def set_seed(seed_value=42):
    """Sets seeds for reproducibility across torch, numpy, and python built-in random."""
    # 1. Python random module
    random.seed(seed_value)
    # 2. Numpy
    np.random.seed(seed_value)
    # 3. Torch
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value) # if using multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed(42)

# --- 1. Data Preparation (Used for NN) ---
X_np = np.array([0.2095, 0.6809, 1.2936, 1.8535, 2.3583, 2.4368, 2.8754, 4.1162, 4.567, 4.7146, 
                 4.8946, 4.9042, 5.8864, 6.205, 6.3962, 7.5324, 7.7828, 8.4835, 9.4854, 9.9582])
Y_np = np.array([1.77268, 2.529927, 2.00102, 2.101015, 2.494044, 2.164226, 2.44769, 2.574242, 
                 4.314459, 1.569597, 2.467982, 2.153414, 1.925144, 1.679639, 4.556762, 3.509959, 
                 3.522292, 3.099583, 0.368901, 3.069406])

# Add intercept column for NN input
X_mat = np.column_stack([np.ones_like(X_np), X_np])

# Convert to PyTorch Tensors
X_torch = torch.tensor(X_mat, dtype=torch.float32)
Y_torch = torch.tensor(Y_np, dtype=torch.float32).unsqueeze(1) # [N, 1]

# --- Quantile Loss Function (Check Function) ---
def quantile_loss(y_true, y_pred, tau):
    u = y_true - y_pred 
    loss = (u * (tau - (u < 0).float())).mean()
    return loss

# --- Model Definition ---
class JointLinearQuantileModel(nn.Module):
    def __init__(self, n_features, n_quantiles):
        super().__init__()
        # nn.Linear is used to simulate the two parameters (beta0 and beta1)
        # Weights are initialized randomly here, hence the need for seeding!
        self.linear = nn.Linear(n_features, n_quantiles, bias=False)
        self.bias = nn.Parameter(torch.zeros(n_quantiles))

    def forward(self, x):
        return self.linear(x) + self.bias
    
# --- Training Function ---
def train_model(model, X, Y, taus, epochs=10000, lr=0.001): # Using lr=0.001
    optimizer = optim.Adam(model.parameters(), lr=lr) 
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        Y_pred = model(X)
        
        total_loss = 0
        for i, tau in enumerate(taus):
            total_loss += quantile_loss(Y, Y_pred[:, i].unsqueeze(1), tau)
        
        total_loss.backward()
        optimizer.step()

    # Extract final parameters
    W = model.linear.weight.data.numpy()
    B = model.bias.data.numpy()

    # Extracting coefficients for Q(0.10) (index 0)
    beta_0_1 = W[0, 0] + B[0]; beta_1_1 = W[0, 1]
    # Extracting coefficients for Q(0.15) (index 1)
    beta_0_2 = W[1, 0] + B[1]; beta_1_2 = W[1, 1]
    
    return [beta_0_1, beta_1_1], [beta_0_2, beta_1_2]

# --- Execution ---
taus = [0.10, 0.15]
model = JointLinearQuantileModel(n_features=2, n_quantiles=2)
# Using lr=0.001 for better stability with Adam and Check Function
beta_nn_10, beta_nn_15 = train_model(model, X_torch, Y_torch, taus, epochs=10000, lr=0.001) 

# --- Output ---
print("="*70)
print("UNCONSTRAINED QUANTILE REGRESSION FORMULAS (PyTorch NN - Seeded)")
print("="*70)

print("\n10% Quantile Formula (τ = 0.10):")
print(f"  Y = {beta_nn_10[0]:.6f} + {beta_nn_10[1]:.6f} * X")

print("\n15% Quantile Formula (τ = 0.15):")
print(f"  Y = {beta_nn_15[0]:.6f} + {beta_nn_15[1]:.6f} * X")
print("="*70)
