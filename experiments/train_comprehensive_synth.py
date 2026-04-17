"""Train MixedSynth on comprehensive stacked dataset with runtime monitoring.

Uses:
- ZI-QDNN for continuous variables (wage_income, etc.)
- Categorical heads for discrete variables (education, occupation, etc.)
- Binary heads for boolean variables (is_male, job_loss, job_gain)
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

start_time = time.time()


class MixedSynth(nn.Module):
    """Multi-head synthesizer for mixed variable types.

    Architecture:
    - Shared encoder (3 hidden layers)
    - Per-variable heads:
        - Zero-inflated quantile head for continuous vars
        - Cross-entropy head for categorical vars
        - Binary head for boolean vars
    """

    def __init__(self, n_cond, zi_vars, cat_vars, binary_vars, hidden=512, n_quantiles=19):
        super().__init__()
        self.zi_vars = zi_vars
        self.cat_vars = cat_vars
        self.binary_vars = binary_vars
        self.n_quantiles = n_quantiles

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_cond, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # ZI-QDNN heads (zero indicator + quantile prediction)
        self.zero_heads = nn.ModuleDict({
            name: nn.Linear(hidden, 1) for name in zi_vars
        })
        self.quant_heads = nn.ModuleDict({
            name: nn.Linear(hidden, n_quantiles) for name in zi_vars
        })

        # Categorical heads
        self.cat_heads = nn.ModuleDict({
            name: nn.Linear(hidden, n_classes) for name, n_classes in cat_vars.items()
        })

        # Binary heads
        self.binary_heads = nn.ModuleDict({
            name: nn.Linear(hidden, 1) for name in binary_vars
        })

    def forward(self, x):
        h = self.encoder(x)

        out = {}

        # ZI-QDNN outputs
        for name in self.zi_vars:
            out[f'{name}_zero'] = torch.sigmoid(self.zero_heads[name](h))
            out[f'{name}_quant'] = self.quant_heads[name](h)

        # Categorical outputs
        for name in self.cat_vars:
            out[f'{name}_logits'] = self.cat_heads[name](h)

        # Binary outputs
        for name in self.binary_vars:
            out[f'{name}_prob'] = torch.sigmoid(self.binary_heads[name](h))

        return out


def quantile_loss(pred, target, quantiles):
    """Pinball loss for quantile regression."""
    target = target.unsqueeze(1)  # [B, 1]
    quantiles = torch.tensor(quantiles, dtype=torch.float32, device=pred.device).unsqueeze(0)  # [1, Q]
    errors = target - pred  # [B, Q]
    loss = torch.max((quantiles - 1) * errors, quantiles * errors)
    return loss.mean()


def train_model(df, epochs=50, batch_size=2048, lr=1e-3, device='mps'):
    """Train MixedSynth on comprehensive dataset."""

    print("\n" + "="*60)
    print("Preparing training data...")
    print("="*60)

    # Predictor variables - demographics + lags for transition modeling
    # Core predictors (all surveys)
    predictors = ['age', 'is_male']

    # Add lag predictors (SIPP only, but that's where transitions matter)
    lag_predictors = ['job1_income_lag1', 'total_income_lag1']
    for lag in lag_predictors:
        if lag in df.columns and df[lag].notna().sum() > 10000:
            predictors.append(lag)

    # Filter to rows with valid predictors FIRST
    mask = df[predictors].notna().all(axis=1)
    train_subset = df[mask]
    print(f"  Training subset: {len(train_subset):,} rows (have all predictors)")

    # Identify variable types - only use vars observed in training subset
    # Continuous (ZI-QDNN) - income variables that can be zero or positive
    zi_vars = []
    for col in ['wage_income', 'self_employment_income', 'interest_income',
                'dividend_income', 'rental_income', 'farm_income',
                'total_income', 'job1_income', 'job2_income', 'job3_income',
                'tip_income', 'social_security', 'total_family_income']:
        if col in train_subset.columns and train_subset[col].notna().sum() > 1000:
            zi_vars.append(col)

    # Categorical - discrete with >2 classes (use train_subset)
    cat_vars = {}
    for col in ['education', 'race', 'marital_status', 'relationship',
                'state_fips', 'job1_occ', 'job1_ind', 'job2_occ', 'job2_ind']:
        if col in train_subset.columns and train_subset[col].notna().sum() > 1000:
            n_classes = int(train_subset[col].max()) + 1
            if n_classes > 2 and n_classes < 100:  # Reasonable number of classes
                cat_vars[col] = n_classes

    # Binary (use train_subset) - exclude is_male since it's a predictor
    binary_vars = []
    for col in ['hispanic', 'job_loss', 'job_gain']:
        if col in train_subset.columns and train_subset[col].notna().sum() > 1000:
            binary_vars.append(col)

    print("\nVariable types:")
    print(f"  ZI-QDNN (continuous): {zi_vars}")
    print(f"  Categorical: {list(cat_vars.keys())}")
    print(f"  Binary: {binary_vars}")
    print(f"  Predictors: {predictors}")

    # Use the pre-filtered train_subset
    train_df = train_subset.copy()
    predictors + zi_vars + list(cat_vars.keys()) + binary_vars
    print(f"\n  Training rows: {len(train_df):,} (of {len(df):,})")

    # Normalize predictor variables (including lags)
    pred_data = train_df[predictors].values.astype(np.float32)
    # Log-transform income lags before normalizing
    for i, p in enumerate(predictors):
        if 'income' in p:
            pred_data[:, i] = np.log1p(np.maximum(pred_data[:, i], 0))
    pred_means = np.nanmean(pred_data, axis=0)
    pred_stds = np.nanstd(pred_data, axis=0) + 1e-6
    pred_data = (pred_data - pred_means) / pred_stds

    X = torch.tensor(pred_data, dtype=torch.float32).to(device)

    # Build target tensors
    targets = {}

    # ZI targets (log-transformed for positive values)
    for var in zi_vars:
        vals = train_df[var].values.astype(np.float32)
        is_zero = (vals <= 0) | np.isnan(vals)
        log_vals = np.where(is_zero, 0, np.log1p(np.maximum(vals, 0)))
        targets[f'{var}_zero'] = torch.tensor(is_zero.astype(np.float32), device=device)
        targets[f'{var}_log'] = torch.tensor(log_vals, device=device)
        targets[f'{var}_mask'] = torch.tensor((~np.isnan(train_df[var].values)).astype(np.float32), device=device)

    # Categorical targets
    for var in cat_vars:
        vals = train_df[var].values.astype(np.float32)
        mask_valid = ~np.isnan(vals)
        vals = np.where(mask_valid, vals, 0).astype(np.int64)
        targets[f'{var}_class'] = torch.tensor(vals, device=device)
        targets[f'{var}_mask'] = torch.tensor(mask_valid.astype(np.float32), device=device)

    # Binary targets
    for var in binary_vars:
        vals = train_df[var].values.astype(np.float32)
        mask_valid = ~np.isnan(vals)
        targets[f'{var}_val'] = torch.tensor(np.where(mask_valid, vals, 0), device=device)
        targets[f'{var}_mask'] = torch.tensor(mask_valid.astype(np.float32), device=device)

    # Model
    model = MixedSynth(
        n_cond=len(predictors),
        zi_vars=zi_vars,
        cat_vars=cat_vars,
        binary_vars=binary_vars,
        hidden=512,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Quantile targets
    quantiles = np.linspace(0.05, 0.95, 19)

    # Training
    print(f"\n{'='*60}")
    print(f"Training MixedSynth ({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"{'='*60}")

    n = len(X)
    batch_times = []

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        n_batches = 0

        # Shuffle
        perm = torch.randperm(n)
        X_shuffled = X[perm]
        targets_shuffled = {k: v[perm] for k, v in targets.items()}

        for i in range(0, n, batch_size):
            batch_start = time.time()

            x_batch = X_shuffled[i:i+batch_size]

            optimizer.zero_grad()
            out = model(x_batch)

            loss = 0

            # ZI losses
            for var in zi_vars:
                mask = targets_shuffled[f'{var}_mask'][i:i+batch_size]
                if mask.sum() < 10:
                    continue

                # Zero indicator loss (BCE)
                zero_pred = out[f'{var}_zero'].squeeze()
                zero_target = targets_shuffled[f'{var}_zero'][i:i+batch_size]
                zero_loss = nn.functional.binary_cross_entropy(
                    zero_pred * mask, zero_target * mask, reduction='sum'
                ) / (mask.sum() + 1e-6)

                # Quantile loss (only for non-zero values)
                nonzero_mask = mask * (1 - zero_target)
                if nonzero_mask.sum() > 10:
                    quant_pred = out[f'{var}_quant']
                    log_target = targets_shuffled[f'{var}_log'][i:i+batch_size]
                    q_loss = quantile_loss(quant_pred[nonzero_mask > 0],
                                          log_target[nonzero_mask > 0], quantiles)
                else:
                    q_loss = 0

                loss = loss + zero_loss + q_loss

            # Categorical losses
            for var in cat_vars:
                mask = targets_shuffled[f'{var}_mask'][i:i+batch_size]
                if mask.sum() < 10:
                    continue
                logits = out[f'{var}_logits']
                target = targets_shuffled[f'{var}_class'][i:i+batch_size]
                cat_loss = nn.functional.cross_entropy(
                    logits[mask > 0], target[mask > 0], reduction='mean'
                )
                loss = loss + cat_loss

            # Binary losses
            for var in binary_vars:
                mask = targets_shuffled[f'{var}_mask'][i:i+batch_size]
                if mask.sum() < 10:
                    continue
                pred = out[f'{var}_prob'].squeeze()
                target = targets_shuffled[f'{var}_val'][i:i+batch_size]
                bin_loss = nn.functional.binary_cross_entropy(
                    pred * mask, target * mask, reduction='sum'
                ) / (mask.sum() + 1e-6)
                loss = loss + bin_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            batch_times.append(time.time() - batch_start)

        scheduler.step()
        epoch_time = time.time() - epoch_start

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_batch = np.mean(batch_times[-n_batches:]) * 1000
            print(f"  Epoch {epoch+1:3d}/{epochs}: loss={total_loss/n_batches:.4f} "
                  f"time={epoch_time:.1f}s ({avg_batch:.0f}ms/batch)")

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.1f}s")

    # Save
    model_info = {
        'state_dict': model.state_dict(),
        'zi_vars': zi_vars,
        'cat_vars': cat_vars,
        'binary_vars': binary_vars,
        'predictors': predictors,
        'pred_means': pred_means,
        'pred_stds': pred_stds,
        'quantiles': quantiles,
    }

    save_path = Path(__file__).parent.parent / "models" / "mixed_synth_comprehensive.pt"
    save_path.parent.mkdir(exist_ok=True)
    torch.save(model_info, save_path)
    print(f"Model saved to {save_path}")

    return model, model_info, train_df


def generate_synthetic(model, model_info, n_samples, train_df, device='mps'):
    """Generate synthetic samples by sampling predictors from training data."""
    print(f"\nGenerating {n_samples:,} synthetic samples...")
    gen_start = time.time()

    model.eval()

    # Sample predictor values from training data
    predictors = model_info['predictors']
    train_pred = train_df[predictors].dropna()
    sample_idx = np.random.choice(len(train_pred), n_samples, replace=True)
    pred_data_raw = train_pred.iloc[sample_idx].values.astype(np.float32)

    # Log-transform income predictors, then normalize
    pred_data = pred_data_raw.copy()
    for i, p in enumerate(predictors):
        if 'income' in p:
            pred_data[:, i] = np.log1p(np.maximum(pred_data[:, i], 0))
    pred_data = (pred_data - model_info['pred_means']) / model_info['pred_stds']
    X = torch.tensor(pred_data, dtype=torch.float32).to(device)

    with torch.no_grad():
        out = model(X)

    result = {}

    # Keep original predictor values (for reference)
    for i, var in enumerate(predictors):
        result[var] = pred_data_raw[:, i]

    # Sample ZI vars
    quantiles = model_info['quantiles']
    for var in model_info['zi_vars']:
        is_zero = out[f'{var}_zero'].cpu().numpy().squeeze() > 0.5
        quant_logits = out[f'{var}_quant'].cpu().numpy()

        # Sample quantile index
        q_idx = np.random.randint(0, len(quantiles), n_samples)
        log_vals = quant_logits[np.arange(n_samples), q_idx]

        # Clip to prevent overflow
        log_vals = np.clip(log_vals, 0, 20)  # exp(20) ~ 485M
        vals = np.expm1(log_vals)
        vals = np.where(is_zero, 0, vals)
        result[var] = vals

    # Sample categorical vars
    for var, n_classes in model_info['cat_vars'].items():
        logits = out[f'{var}_logits'].cpu().numpy()
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        vals = np.array([np.random.choice(n_classes, p=p) for p in probs])
        result[var] = vals

    # Sample binary vars
    for var in model_info['binary_vars']:
        probs = out[f'{var}_prob'].cpu().numpy().squeeze()
        vals = (np.random.rand(n_samples) < probs).astype(int)
        result[var] = vals

    df = pd.DataFrame(result)
    gen_time = time.time() - gen_start
    print(f"  Generated in {gen_time:.1f}s")

    return df


def evaluate_coverage(synth, holdout, vars_to_check, core_vars=None):
    """Compute coverage (median NN distance from holdout to synthetic).

    Args:
        synth: Synthetic data
        holdout: Holdout data
        vars_to_check: All variables to consider
        core_vars: Subset of vars that must be present (if None, use minimal set)
    """
    print("\nEvaluating coverage...")

    if core_vars is None:
        # Use variables with good coverage in both datasets
        core_vars = []
        for v in vars_to_check:
            if v in synth.columns and v in holdout.columns:
                synth_obs = synth[v].notna().mean()
                holdout_obs = holdout[v].notna().mean()
                if synth_obs > 0.5 and holdout_obs > 0.3:
                    core_vars.append(v)

    if len(core_vars) < 3:
        print("  Not enough variables with good coverage")
        return np.nan, []

    print(f"  Using {len(core_vars)} variables: {core_vars}")

    # Drop rows with any NaN in core vars
    synth_clean = synth[core_vars].dropna()
    holdout_clean = holdout[core_vars].dropna()

    print(f"  Synth: {len(synth_clean):,}, Holdout: {len(holdout_clean):,}")

    if len(synth_clean) < 100 or len(holdout_clean) < 100:
        print("  Not enough data after dropping NaN")
        return np.nan, core_vars

    # Normalize
    means = synth_clean.mean()
    stds = synth_clean.std() + 1e-6

    synth_norm = (synth_clean - means) / stds
    holdout_norm = (holdout_clean - means) / stds

    # Fit NN on synthetic
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
    nn.fit(synth_norm.values)

    # Find distances from holdout
    n_eval = min(5000, len(holdout_norm))
    distances, _ = nn.kneighbors(holdout_norm.values[:n_eval])

    coverage = np.median(distances)
    print(f"  Coverage (median NN distance): {coverage:.4f}")

    return coverage, core_vars


def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load data
    data_path = Path(__file__).parent.parent / "data" / "stacked_comprehensive.parquet"
    print(f"Loading {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Loaded: {len(df):,} rows x {len(df.columns)} cols")

    # Train
    model, model_info, train_df = train_model(df, epochs=50, batch_size=2048, device=device)

    # Generate synthetic (1M samples for good coverage)
    synth = generate_synthetic(model, model_info, n_samples=1_000_000, train_df=train_df, device=device)

    # Evaluate coverage
    vars_for_coverage = model_info['zi_vars'] + list(model_info['cat_vars'].keys()) + model_info['binary_vars']

    # Use original data as holdout (sample)
    holdout = train_df.sample(n=min(10000, len(train_df)), random_state=42)

    coverage, used_vars = evaluate_coverage(synth, holdout, vars_for_coverage)

    # Report statistics
    print(f"\n{'='*60}")
    print("Synthetic data statistics:")
    print("="*60)
    for var in model_info['zi_vars'][:5]:  # First 5 ZI vars
        if var in synth.columns:
            zero_pct = (synth[var] == 0).mean() * 100
            nonzero = synth[var][synth[var] > 0]
            if len(nonzero) > 0:
                print(f"  {var}: {zero_pct:.0f}% zero, nonzero median={nonzero.median():,.0f}")

    # Save synthetic
    synth_path = Path(__file__).parent.parent / "data" / "synthetic_comprehensive_1m.parquet"
    synth.to_parquet(synth_path, index=False)
    print(f"\nSynthetic data saved to {synth_path}")
    print(f"Size: {synth_path.stat().st_size / 1e6:.1f} MB")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"DONE! Total runtime: {total_time:.1f}s")
    print(f"Coverage: {coverage:.4f} on {len(used_vars)} variables")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
