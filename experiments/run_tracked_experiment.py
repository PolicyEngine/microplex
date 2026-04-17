"""Run a synthesis experiment with full tracking."""

import sys
from pathlib import Path

# Direct import to avoid microplex __init__ pulling in polars
experiments_path = Path(__file__).parent.parent / "src" / "microplex" / "experiments"
sys.path.insert(0, str(experiments_path.parent.parent))

# Import directly from module files
import importlib.util
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Load tracker first
spec = importlib.util.spec_from_file_location("tracker", experiments_path / "tracker.py")
tracker_module = importlib.util.module_from_spec(spec)
sys.modules["tracker"] = tracker_module  # Make available for registry import
spec.loader.exec_module(tracker_module)
ExperimentTracker = tracker_module.ExperimentTracker

# Now load registry (it imports tracker)
spec = importlib.util.spec_from_file_location("registry", experiments_path / "registry.py")
registry_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(registry_module)
ExperimentRegistry = registry_module.ExperimentRegistry


def run_experiment():
    # Initialize tracker
    tracker = ExperimentTracker(Path(__file__).parent / "tracking")

    # Create experiment
    exp = tracker.create_experiment(
        name="Multi-source ZI-QDNN v1",
        description="CPS+SIPP+PSID fusion with zero-inflated quantile DNN. "
                    "Predicts annual_income using demographics, survey dummies, and income lag."
    )

    print("=" * 60)
    print(f"Experiment: {exp.id}")
    print(f"  {exp.name}")
    print("=" * 60)

    # Load data
    df = pd.read_parquet(Path(__file__).parent.parent / "data" / "stacked_comprehensive.parquet")

    # Create harmonized annual income
    df["annual_income"] = np.nan
    cps_mask = df["_survey"] == "cps"
    sipp_mask = df["_survey"] == "sipp"
    psid_mask = df["_survey"] == "psid"

    if "wage_income" in df.columns:
        df.loc[cps_mask, "annual_income"] = df.loc[cps_mask, "wage_income"]
        df.loc[psid_mask, "annual_income"] = df.loc[psid_mask, "wage_income"]
    if "total_income" in df.columns:
        df.loc[sipp_mask, "annual_income"] = df.loc[sipp_mask, "total_income"] * 12

    # Create lag for SIPP
    df["annual_income_lag"] = np.nan
    if "total_income_lag1" in df.columns:
        df.loc[sipp_mask, "annual_income_lag"] = df.loc[sipp_mask, "total_income_lag1"] * 12

    # Survey dummies
    df["survey_cps"] = (df["_survey"] == "cps").astype(int)
    df["survey_sipp"] = (df["_survey"] == "sipp").astype(int)
    df["survey_psid"] = (df["_survey"] == "psid").astype(int)

    # Define variables
    predictors = [
        "age", "is_male", "annual_income_lag",
        "survey_cps", "survey_sipp", "survey_psid",
        "race", "hispanic", "education", "marital_status", "state_fips"
    ]
    target = "annual_income"

    # Track variables
    variable_sources = {
        "age": ["cps", "sipp", "psid"],
        "is_male": ["cps", "sipp", "psid"],
        "annual_income_lag": ["sipp"],
        "survey_cps": ["derived"],
        "survey_sipp": ["derived"],
        "survey_psid": ["derived"],
        "race": ["sipp"],
        "hispanic": ["sipp"],
        "education": ["sipp"],
        "marital_status": ["sipp", "psid"],
        "state_fips": ["cps", "psid"],
    }

    for var in predictors:
        tracker.add_variable(
            exp, var,
            sources=variable_sources.get(var, ["unknown"]),
            role="predictor",
            dtype="continuous" if var in ["age", "annual_income_lag"] else "categorical",
        )
    tracker.add_variable(exp, target, sources=["cps", "sipp", "psid"], role="target", dtype="continuous")
    exp.target_variable = target

    # Prep data
    df_valid = df[df[target].notna()].copy()
    for col in predictors:
        if col in df_valid.columns:
            df_valid[col] = df_valid[col].fillna(0)
        else:
            df_valid[col] = 0

    # Train/test split
    np.random.seed(42)
    n = len(df_valid)
    train_idx = np.random.choice(n, size=int(0.8 * n), replace=False)
    test_idx = np.setdiff1d(np.arange(n), train_idx)
    train_df = df_valid.iloc[train_idx].reset_index(drop=True)
    test_df = df_valid.iloc[test_idx].reset_index(drop=True)

    # Track dataset splits
    for survey in ["cps", "sipp", "psid"]:
        train_mask = train_df["_survey"] == survey
        test_mask = test_df["_survey"] == survey
        n_train = train_mask.sum()
        n_holdout = test_mask.sum()

        available_vars = [c for c in df_valid.columns
                         if df_valid.loc[df_valid["_survey"] == survey, c].notna().mean() > 0.5]

        tracker.add_dataset(
            exp, survey,
            n_train=int(n_train),
            n_holdout=int(n_holdout),
            variables_available=available_vars,
            waves_used=list(train_df.loc[train_mask, "wave"].dropna().unique().astype(int)) if "wave" in train_df.columns else None,
        )

    print("\nDatasets:")
    for ds in exp.datasets:
        print(f"  {ds.survey.upper()}: train={ds.n_train:,}, holdout={ds.n_holdout:,} ({ds.train_share:.0%})")

    # Prepare tensors
    X_train = train_df[predictors].values.astype(np.float32)
    y_train = train_df[target].values.astype(np.float32)
    test_df[predictors].values.astype(np.float32)
    y_test = test_df[target].values.astype(np.float32)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    y_scale = np.abs(y_train).std()
    y_mean = y_train.mean()
    y_train_scaled = (y_train - y_mean) / y_scale
    is_zero_train = (y_train == 0).astype(np.float32)

    X_train_t = torch.from_numpy(X_train_scaled)
    y_train_t = torch.from_numpy(y_train_scaled)
    is_zero_train_t = torch.from_numpy(is_zero_train)

    # Model config
    n_q = 19
    hidden_units = 256
    n_layers = 2
    batch_size = 4096
    n_epochs = 30
    lr = 1e-3

    tracker.set_model_config(
        exp,
        model_type="zi-qdnn",
        architecture={
            "hidden_units": hidden_units,
            "n_layers": n_layers,
            "n_quantiles": n_q,
            "zero_inflation": True,
        },
        training={
            "epochs": n_epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": "adam",
            "grad_clip": 1.0,
        },
        quantiles=list(np.linspace(0.05, 0.95, n_q)),
    )

    # Build model
    class ZIQDNN(nn.Module):
        def __init__(self, n_in, n_q=19):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(n_in, hidden_units), nn.ReLU(),
                nn.Linear(hidden_units, hidden_units), nn.ReLU(),
            )
            self.zero_head = nn.Linear(hidden_units, 1)
            self.quant_head = nn.Linear(hidden_units, n_q)

        def forward(self, x):
            h = self.shared(x)
            return torch.sigmoid(self.zero_head(h)), self.quant_head(h)

    model = ZIQDNN(X_train_scaled.shape[1], n_q)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    quantile_levels = torch.linspace(0.05, 0.95, n_q)

    def pinball_loss(pred, target, tau):
        err = target - pred
        return torch.mean(torch.max(tau * err, (tau - 1) * err))

    # Train
    print(f"\nTraining {exp.model.model_type}...")
    start_time = time.time()

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(len(X_train_t))
        for i in range(0, len(X_train_t), batch_size):
            idx = perm[i:i + batch_size]
            x, z, y = X_train_t[idx], is_zero_train_t[idx], y_train_t[idx]

            p_zero, quantiles = model(x)
            zero_loss = nn.functional.binary_cross_entropy(p_zero.squeeze(), z)

            nonzero_mask = z == 0
            if nonzero_mask.sum() > 0:
                q_loss = sum(pinball_loss(quantiles[nonzero_mask, j], y[nonzero_mask], tau)
                             for j, tau in enumerate(quantile_levels)) / n_q
            else:
                q_loss = torch.tensor(0.0)

            loss = zero_loss + q_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{n_epochs}")

    exp.training_time_seconds = time.time() - start_time
    print(f"Training time: {exp.training_time_seconds:.1f}s")

    # Generate synthetic from train predictors
    print("\nGenerating synthetic...")
    model.eval()
    with torch.no_grad():
        p_zero, quantiles = model(X_train_t)

    quantiles_np = quantiles.numpy() * y_scale + y_mean
    p_zero_np = p_zero.squeeze().numpy()

    np.random.seed(42)
    synth = np.zeros(len(X_train))
    for i in range(len(X_train)):
        if np.random.random() < p_zero_np[i]:
            synth[i] = 0
        else:
            u = np.random.random()
            tau_idx = min(int(u * (n_q - 1)), n_q - 2)
            q_low, q_high = quantiles_np[i, tau_idx], quantiles_np[i, tau_idx + 1]
            synth[i] = q_low + (u * (n_q - 1) - tau_idx) * (q_high - q_low)

    # Save synthetic data
    tracker.save_synthetic_data(exp, synth, X_train, predictors)
    print(f"Saved synthetic: {exp.synthetic_data_path}")

    # Compute coverage
    print("\nComputing coverage...")
    nn_model = NearestNeighbors(n_neighbors=1)
    nn_model.fit(synth.reshape(-1, 1))
    dist, nn_idx = nn_model.kneighbors(y_test.reshape(-1, 1))
    norm_dist = dist.flatten() / np.std(y_test)

    # Coverage by survey
    for survey in ["cps", "sipp", "psid"]:
        mask = test_df["_survey"].values == survey
        if mask.sum() == 0:
            continue
        tracker.add_coverage_result(
            exp, survey,
            y_test[mask],
            norm_dist[mask],
        )

    exp.overall_coverage_median = float(np.median(norm_dist))
    exp.overall_coverage_mean = float(np.mean(norm_dist))

    print("\nCoverage results:")
    for cr in exp.coverage_results:
        print(f"  {cr.survey.upper()}: median={cr.coverage_median:.6f}, p99={cr.coverage_p99:.4f}")
    print(f"  OVERALL: median={exp.overall_coverage_median:.6f}")

    # Save holdout coverage with pointers
    nearest_synth_values = synth[nn_idx.flatten()]
    tracker.save_holdout_coverage(
        exp, test_df, y_test, norm_dist, nn_idx.flatten(), nearest_synth_values
    )
    print(f"Saved holdout coverage: {exp.holdout_coverage_path}")

    # Save model
    model_path = tracker.data_dir / f"{exp.id}_model.pt"
    torch.save({
        "model_state": model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "y_scale": y_scale,
        "y_mean": y_mean,
        "predictors": predictors,
    }, model_path)
    exp.model_path = str(model_path)

    # Save experiment metadata
    exp_path = tracker.save_experiment(exp)
    print(f"\nExperiment saved: {exp_path}")

    # Export for dashboard
    registry = ExperimentRegistry(tracker)
    dashboard_json = tracker.base_dir / "dashboard_data.json"
    registry.export_for_dashboard(dashboard_json)
    print(f"Dashboard data: {dashboard_json}")

    return exp


if __name__ == "__main__":
    exp = run_experiment()
