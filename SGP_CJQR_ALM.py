"""
═══════════════════════════════════════════════════════════════════════════════
FINAL SGP COMPARISON SUITE: "THE VALIDITY PROOF"
═══════════════════════════════════════════════════════════════════════════════

FEATURES:
1.  100 Quantiles (Matches R by stats model).
2.  "Affected SGP" Metric: Checks if crossing happens near the assigned rank.
3.  High-Precision ALM:
    - Warm-Up = 100 iters (Lowers RMSE).
    - Rho Cap = 10,000,000 (Fixes Grade 11).
    - Gamma = 0.5 (Stricter trigger for better fit).
4.  Simulation: AASA Scale (Sigma~10) for realistic RMSE comparison.
5.  VISUALIZATION: Plots 100 quantile curves for Math Grade 4 to show tangling.

Author: Tom Chang
Date: December 2025
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import interpolate, stats
from statsmodels.regression.quantile_regression import QuantReg
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("FINAL SGP COMPARISON SUITE")
print("Config: 100 Quantiles | Warm-Up=100 | Affected SGP Check")
print("=" * 80)
print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {np.__version__}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_bspline_basis(x, knots, boundaries, degree=3):
    t = np.concatenate([
        [boundaries[0]] * (degree + 1),
        knots,
        [boundaries[1]] * (degree + 1)
    ])
    try:
        matrix = interpolate.BSpline.design_matrix(x, t, k=degree).toarray()[:, 1:]
        return matrix
    except:
        return np.zeros((len(x), len(knots) + degree))

def calculate_knots_boundaries(scores, percentiles=[0.2, 0.4, 0.6, 0.8], extension=0.1):
    if len(scores) < 100: return None, None
    knots = np.quantile(scores, percentiles)
    score_range = scores.max() - scores.min()
    boundaries = np.array([
        scores.min() - extension * score_range,
        scores.max() + extension * score_range
    ])
    return knots, boundaries

def isotonize_predictions(predictions):
    return np.sort(predictions, axis=1)

def calculate_sgp(predictions, y_actual):
    preds_sorted = np.sort(predictions, axis=1)
    sgps = []
    for i in range(len(y_actual)):
        comparison = preds_sorted[i, :] < y_actual[i]
        if np.any(comparison):
            rank = np.where(comparison)[0][-1] + 1
        else:
            rank = 1
        sgps.append(rank)
    return np.clip(np.array(sgps), 1, 99)

def count_affected_students(raw_preds, sgps, window=10):
    diffs = raw_preds[:, 1:] - raw_preds[:, :-1]
    is_crossing = diffs < 0

    affected_count = 0
    n_quantiles = raw_preds.shape[1]

    for i in range(len(sgps)):
        s = sgps[i]
        idx_start = max(0, s - window)
        idx_end = min(n_quantiles - 2, s + window)

        if idx_start < idx_end:
            if np.any(is_crossing[i, idx_start:idx_end]):
                affected_count += 1

    return affected_count

def plot_quantile_curves(model, X_data, knots, bounds, title, filename):
    """
    Visualizes 100 quantile curves for a single prior variable (Order 1).
    This shows the "Tangle" vs "Non-Crossing".
    """
    # Generate a smooth range of Prior Scores (X axis)
    x_min, x_max = X_data.min(), X_data.max()
    x_grid = np.linspace(x_min, x_max, 200).reshape(-1, 1)

    # Predict quantiles for this grid using the trained model
    # We use the model's predict_raw method but need to handle different model types
    if hasattr(model, 'predict_raw'):
        # For our class wrappers
        # Temporarily swap the internal knots/bounds to the grid context if needed
        # But predict_raw expects X to match the training dimension.
        # Since this is for Math Grade 4 (Order 1), X is just 1 column.

        # Manually build basis for grid to be safe
        # Note: IndepQRModel and ALMModel store knots/bounds differently in fit
        # We will assume Order 1 and pass manual X construction
        pass

    # Easier approach: Use the predict logic directly here
    # Build design matrix for grid
    X_parts = [np.ones((len(x_grid), 1))]
    # Order 1: single column
    basis = create_bspline_basis(x_grid.flatten(), knots[0], bounds[0])
    X_parts.append(basis)
    X_mat = np.hstack(X_parts)

    plt.figure(figsize=(12, 8))

    # Predict and Plot
    if isinstance(model, IndepQRModel):
        for tau in model.quantiles:
            if tau in model.models:
                y_pred = model.models[tau].predict(X_mat)
                plt.plot(x_grid, y_pred, color='black', alpha=0.1, linewidth=0.5)

    elif isinstance(model, ALMModel):
        # ALM uses PyTorch model
        X_tensor = torch.tensor(X_mat, dtype=torch.float64)
        with torch.no_grad():
            preds_scaled = model.model(X_tensor).numpy()
        # Rescale
        y_pred_grid = preds_scaled * model.y_std + model.y_mean

        for i in range(y_pred_grid.shape[1]):
            plt.plot(x_grid, y_pred_grid[:, i], color='blue', alpha=0.1, linewidth=0.5)

    plt.title(title)
    plt.xlabel("Prior Year Scale Score (Grade 3)")
    plt.ylabel("Current Year Scale Score (Grade 4)")
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"    📸 Saved curve plot: {filename}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. INDEPENDENT QR MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class IndepQRModel:
    def __init__(self, n_quantiles=100):
        self.n_quantiles = n_quantiles
        self.quantiles = (np.arange(1, n_quantiles + 1) - 0.5) / 100
        self.models = {}
        self.knots_list = []
        self.bounds_list = []
        self.n_priors = 0

    def fit(self, X, y, knots_list, bounds_list, verbose=False):
        self.knots_list = knots_list
        self.bounds_list = bounds_list
        self.n_priors = len(knots_list)

        X_parts = [np.ones((len(X), 1))]
        for i in range(self.n_priors):
            col = X[:, i] if X.ndim > 1 else X.flatten()
            X_parts.append(create_bspline_basis(col, knots_list[i], bounds_list[i]))
        X_mat = np.hstack(X_parts)

        converged = 0
        for tau in self.quantiles:
            try:
                mod = QuantReg(y, X_mat)
                self.models[tau] = mod.fit(q=tau, max_iter=2000, p_tol=1e-4)
                converged += 1
            except: pass

        if verbose: print(f"    Indep-QR: {converged}/{self.n_quantiles} quantiles converged")

    def predict_raw(self, X):
        X_parts = [np.ones((len(X), 1))]
        for i in range(self.n_priors):
            col = X[:, i] if X.ndim > 1 else X.flatten()
            X_parts.append(create_bspline_basis(col, self.knots_list[i], self.bounds_list[i]))
        X_mat = np.hstack(X_parts)

        preds = []
        for tau in self.quantiles:
            if tau in self.models:
                preds.append(self.models[tau].predict(X_mat))
            else:
                preds.append(np.full(len(X), np.nan))
        return np.column_stack(preds)

    def predict_isotonized(self, X):
        return isotonize_predictions(self.predict_raw(X))

    def count_crossings(self, X):
        raw = self.predict_raw(X)
        return np.sum(np.any(raw[:, 1:] - raw[:, :-1] < 0, axis=1))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ALM MODEL (High Precision Config)
# ═══════════════════════════════════════════════════════════════════════════════

class LinearQuantileModel(nn.Module):
    def __init__(self, n_features, n_quantiles):
        super().__init__()
        self.linear = nn.Linear(n_features, n_quantiles, bias=False)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
    def forward(self, X): return self.linear(X)

class ALMModel:
    def __init__(self, n_quantiles=100):
        self.n_quantiles = n_quantiles
        self.quantiles = (np.arange(1, n_quantiles + 1) - 0.5) / 100
        self.model = None
        self.knots_list = []
        self.bounds_list = []
        self.y_mean = 0.0
        self.y_std = 1.0

    def fit(self, X, y, knots_list, bounds_list, verbose=False):
        self.knots_list = knots_list
        self.bounds_list = bounds_list

        # Standardize
        self.y_mean = y.mean()
        self.y_std = y.std()
        y_scaled = (y - self.y_mean) / self.y_std

        # Design Matrix
        X_parts = [np.ones((len(X), 1))]
        for i in range(len(knots_list)):
            col = X[:, i] if X.ndim > 1 else X.flatten()
            X_parts.append(create_bspline_basis(col, knots_list[i], bounds_list[i]))
        X_design = np.hstack(X_parts)

        X_tensor = torch.tensor(X_design, dtype=torch.float64)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float64).unsqueeze(1)
        q_tensor = torch.tensor(self.quantiles, dtype=torch.float64)

        n_samples = len(X)
        self.model = LinearQuantileModel(X_design.shape[1], self.n_quantiles).double()

        # --- TUNED PARAMETERS FOR ACCURACY & VALIDITY ---
        warmup_iters = 100      # Extended Warm-Up
        max_outer = 100         # High iterations
        gamma = 0.5             # Strict decrease check
        rho = 1.0               # Gentle start
        rho_increase = 1.5      # Slow growth
        rho_max = 10000000.0    # 10 Million Cap

        # Phase 1: Warm-Up
        if verbose: print(f"    [Phase 1] Warm-up ({warmup_iters} iters)...")
        opt_warm = optim.LBFGS(self.model.parameters(), lr=1.0, max_iter=warmup_iters, line_search_fn='strong_wolfe')
        def closure_w():
            opt_warm.zero_grad()
            preds = self.model(X_tensor)
            err = y_tensor - preds
            huber = torch.where(torch.abs(err)<=0.1, 0.5*err**2/0.1, torch.abs(err)-0.05)
            w = torch.where(err>=0, q_tensor, 1-q_tensor)
            loss = torch.mean(w * huber)
            loss.backward()
            return loss
        opt_warm.step(closure_w)

        # Phase 2: ALM
        if verbose: print("    [Phase 2] ALM Optimization...")
        mu = torch.zeros(len(X), self.n_quantiles-1, dtype=torch.float64)
        prev_max_viol = float('inf')

        for outer_k in range(max_outer):
            opt = optim.LBFGS(self.model.parameters(), lr=1.0, max_iter=20, line_search_fn='strong_wolfe')
            curr_mu, curr_rho = mu.clone(), rho

            def closure():
                opt.zero_grad()
                preds = self.model(X_tensor)
                err = y_tensor - preds
                huber = torch.where(torch.abs(err)<=0.1, 0.5*err**2/0.1, torch.abs(err)-0.05)
                w = torch.where(err>=0, q_tensor, 1-q_tensor)
                loss_fit = torch.mean(w * huber)

                diffs = preds[:, 1:] - preds[:, :-1]
                viol = torch.relu(1e-5 - diffs)
                loss_lag = torch.sum(curr_mu * viol) / len(X)
                loss_pen = (curr_rho / 2.0) * torch.mean(viol ** 2)

                total = loss_fit + loss_lag + loss_pen
                total.backward()
                return total

            opt.step(closure)

            with torch.no_grad():
                preds = self.model(X_tensor)
                diffs = preds[:, 1:] - preds[:, :-1]
                viol_pos = torch.relu(1e-5 - diffs)
                max_viol = viol_pos.max().item()

                mu += rho * viol_pos

                if max_viol <= 1e-8:
                    if verbose: print(f"    ✅ Converged: Iter {outer_k}, ρ={rho:.1f}")
                    break

                if max_viol > gamma * prev_max_viol:
                    if rho < rho_max: rho *= rho_increase

                prev_max_viol = max_viol

                if verbose and outer_k % 10 == 0:
                    n_cross = int((diffs < 0).any(dim=1).sum())
                    print(f"    [Iter {outer_k}] ρ={rho:<8.1f} MaxViol={max_viol:.1e} Cross={n_cross}")

    def predict_raw(self, X):
        X_parts = [np.ones((len(X), 1))]
        for i in range(len(self.knots_list)):
            col = X[:, i] if X.ndim > 1 else X.flatten()
            X_parts.append(create_bspline_basis(col, self.knots_list[i], self.bounds_list[i]))
        X_tensor = torch.tensor(np.hstack(X_parts), dtype=torch.float64)

        with torch.no_grad():
            preds_scaled = self.model(X_tensor).numpy()

        return preds_scaled * self.y_std + self.y_mean

    def count_crossings(self, X):
        raw = self.predict_raw(X)
        return np.sum(np.any(raw[:, 1:] - raw[:, :-1] < 0, axis=1))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: DATA PREPROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════

class SGPDataPreprocessor:
    def __init__(self): pass

    def load_data(self, filepath):
        print(f"\nLoading {filepath}...")
        try:
            df = pd.read_csv(filepath, on_bad_lines='skip', engine='python')
        except: return None

        if 'GRADE_ENROLLED' in df.columns: df.rename(columns={'GRADE_ENROLLED': 'GRADE'}, inplace=True)
        if 'MATH' in df.values: df.loc[df['CONTENT_AREA']=='MATH', 'CONTENT_AREA']='MATHEMATICS'

        df['YEAR'] = df['YEAR'].astype(str)
        df['GRADE'] = df['GRADE'].astype(int).astype(str)
        df['SCALE_SCORE'] = pd.to_numeric(df['SCALE_SCORE'], errors='coerce')
        return df

    def calculate_knots(self, df, subject, grade):
        s = df[(df['GRADE']==str(grade)) & (df['CONTENT_AREA']==subject)]['SCALE_SCORE'].dropna().values
        return calculate_knots_boundaries(s)

    def prepare_cohorts(self, df, current_year=2025):
        configs = {
            4: [3], 5: [3, 4], 6: [4, 5], 7: [5, 6],
            8: [6, 7], 9: [7, 8], 10: [8, 9], 11: [8, 9]
        }
        prepared = {}
        target_subs = ['MATHEMATICS', 'ELA']

        for subj in target_subs:
            subj_df = df[df['CONTENT_AREA'] == subj]
            for grade, priors in configs.items():
                curr = subj_df[(subj_df['YEAR']==str(current_year)) & (subj_df['GRADE']==str(grade))]
                if curr.empty: continue
                curr = curr[['ID', 'SCALE_SCORE']].rename(columns={'SCALE_SCORE': 'current'})

                cohort = curr
                valid_priors = True
                knots_list, bounds_list = [], []

                for i, p_grade in enumerate(priors):
                    grade_gap = grade - p_grade
                    p_year = current_year - grade_gap
                    prior_df = subj_df[(subj_df['YEAR']==str(p_year)) & (subj_df['GRADE']==str(p_grade))]

                    if prior_df.empty: valid_priors = False; break
                    prior_df = prior_df[['ID', 'SCALE_SCORE']].rename(columns={'SCALE_SCORE': f'prior_{i}'})
                    cohort = cohort.merge(prior_df, on='ID', how='inner')

                    k, b = self.calculate_knots(subj_df, subj, p_grade)
                    if k is None: valid_priors = False; break
                    knots_list.append(k)
                    bounds_list.append(b)

                if valid_priors and len(cohort) > 100:
                    x_cols = sorted([c for c in cohort.columns if 'prior' in c])
                    prepared[(subj, grade)] = {
                        'X': cohort[x_cols].values,
                        'y': cohort['current'].values,
                        'knots': knots_list,
                        'bounds': bounds_list
                    }
                    print(f"  Prepared {subj} Grade {grade}: {len(cohort):,} students")
        return prepared


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: SIMULATION STUDY
# ═══════════════════════════════════════════════════════════════════════════════

def run_simulation_study(n_samples=3000, n_reps=20, verbose=True):
    print("\n" + "=" * 80)
    print("SIMULATION STUDY (Realistic AASA Scale)")
    print("=" * 80)

    n_quantiles = 100
    taus = (np.arange(1, n_quantiles + 1) - 0.5) / 100

    def true_quantile(x, tau):
        mu = 2400 + 15 * x + 0.5 * x**2
        sigma = 10 + 0.5 * x
        return mu + sigma * stats.norm.ppf(tau)

    def generate_data(n, seed=None):
        if seed is not None: np.random.seed(seed)
        x = np.random.uniform(0, 10, n)
        y = np.random.normal(2400 + 15*x + 0.5*x**2, 10 + 0.5*x)
        return x, y

    rmse_indep_all = []
    rmse_alm_all = []

    for rep in range(n_reps):
        if verbose: print(f"Replication {rep + 1}/{n_reps}")
        X, y = generate_data(n_samples, seed=rep * 42 + 123)
        Q_true = np.array([[true_quantile(x_i, tau) for tau in taus] for x_i in X])
        knots, boundaries = calculate_knots_boundaries(X)
        X_reshaped = X.reshape(-1, 1)

        # Indep
        model_indep = IndepQRModel()
        model_indep.fit(X_reshaped, y, [knots], [boundaries], verbose=False)
        preds_indep = model_indep.predict_isotonized(X_reshaped)
        rmse_indep = np.sqrt(np.mean((preds_indep - Q_true) ** 2))
        rmse_indep_all.append(rmse_indep)

        # ALM (Uses FULL PRECISION settings: fast_mode=False)
        model_alm = ALMModel()
        model_alm.fit(X_reshaped, y, [knots], [boundaries], verbose=False)
        preds_alm = model_alm.predict_raw(X_reshaped)
        preds_alm_sorted = np.sort(preds_alm, axis=1)
        rmse_alm = np.sqrt(np.mean((preds_alm_sorted - Q_true) ** 2))
        rmse_alm_all.append(rmse_alm)

        if verbose:
            print(f"    RMSE: Indep={rmse_indep:.3f}, ALM={rmse_alm:.3f}")

    return {'rmse_indep': np.array(rmse_indep_all), 'rmse_alm': np.array(rmse_alm_all)}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: REAL DATA COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def run_real_data_comparison(filepath='training.csv'):
    print("\n" + "=" * 80)
    print("REAL DATA COMPARISON (All Grades/Subjects)")
    print("=" * 80)

    preprocessor = SGPDataPreprocessor()
    df = preprocessor.load_data(filepath)
    if df is None: return None, None

    cohorts = preprocessor.prepare_cohorts(df)
    summary_results = []

    for (subject, grade), data in sorted(cohorts.items()):
        print(f"\n{'─'*60}")
        print(f"{subject} Grade {grade} (N={len(data['y']):,})")
        print(f"{'─'*60}")

        X, y = data['X'], data['y']
        knots_list, bounds_list = data['knots'], data['bounds']
        n = len(y)

        # 1. Indep QR
        print("  Training Indep-QR...")
        m_indep = IndepQRModel()
        m_indep.fit(X, y, knots_list, bounds_list)
        pred_indep_raw = m_indep.predict_raw(X)
        pred_indep_iso = isotonize_predictions(pred_indep_raw)
        cross_indep = m_indep.count_crossings(X)
        print(f"    Crossings: {cross_indep}/{n} ({cross_indep/n*100:.2f}%)")

        # 2. ALM
        print("  Training ALM (Warm-up + Standard)...")
        m_alm = ALMModel()
        m_alm.fit(X, y, knots_list, bounds_list, verbose=True)
        pred_alm = m_alm.predict_raw(X)
        cross_alm = m_alm.count_crossings(X)
        print(f"    Crossings: {cross_alm}/{n} ({cross_alm/n*100:.2f}%)")

        # 3. SGP Calculation & Validity Check
        sgp_indep = calculate_sgp(pred_indep_iso, y)
        sgp_alm = calculate_sgp(pred_alm, y)

        # NEW METRIC: Affected SGPs
        affected = count_affected_students(pred_indep_raw, sgp_indep, window=10)
        affected_pct = affected / n * 100
        print(f"    ⚠️  Affected SGPs (Indep): {affected:,} ({affected_pct:.2f}%)")

        # Agreement
        diff = np.abs(sgp_indep - sgp_alm)
        match = np.mean(diff == 0) * 100
        within_5 = np.mean(diff <= 5) * 100
        print(f"    Agreement: Exact={match:.1f}%, ±5={within_5:.1f}%")

        summary_results.append({
            'Subject': subject, 'Grade': grade, 'N': n,
            'Cross_Indep_%': cross_indep/n*100,
            'Affected_Indep_%': affected_pct,
            'Cross_ALM_%': cross_alm/n*100,
            'Within_5_%': within_5
        })

        # Plot SGP Histograms
        plt.figure(figsize=(10, 4))
        plt.hist(sgp_indep, bins=50, alpha=0.5, label='Indep-QR', density=True)
        plt.hist(sgp_alm, bins=50, alpha=0.5, label='ALM', density=True)
        plt.title(f"{subject} Grade {grade}")
        plt.legend()
        plt.savefig(f"sgp_hist_{subject}_{grade}.png")
        plt.close()

        # NEW: SPECIAL PLOT FOR MATH GRADE 4 (CURVES)
        if subject == "MATHEMATICS" and grade == 4:
             print("    📸 Generating Quantile Curves Plot for Math G4...")
             plot_quantile_curves(m_indep, X, knots_list, bounds_list,
                                 "Indep-QR (Math G4): Visible Crossing", "math_g4_indep_curves.png")
             plot_quantile_curves(m_alm, X, knots_list, bounds_list,
                                 "CJQR-ALM (Math G4): Non-Crossing", "math_g4_alm_curves.png")


    return pd.DataFrame(summary_results), cohorts

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    summary_df, _ = run_real_data_comparison('training.csv')
    if summary_df is not None:
        print("\n" + "="*80)
        print(summary_df.to_string(index=False, float_format="%.2f"))
        summary_df.to_csv('sgp_full_summary.csv', index=False)

    sim_res = run_simulation_study()
    d_mean = sim_res['rmse_indep'].mean() - sim_res['rmse_alm'].mean()
    print(f"\nSimulation RMSE Diff (Indep - ALM) = {d_mean:.4f}")