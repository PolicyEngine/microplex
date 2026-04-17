"""Alternative methods for multi-source population synthesis.

Methods:
1. QRF (baseline): Conditional factorization P(X|shared) per survey
2. Gaussian Copula: Model marginals + dependence structure
3. VAE: Latent variable model with masked training
4. CTGAN: GAN-based tabular synthesis (via SDV)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from microplex.dgp import EvalResult, Survey, compute_prdc


class SynthesisMethod(ABC):
    """Base class for population synthesis methods."""

    name: str = "base"

    @abstractmethod
    def fit(self, surveys: list[Survey], shared_cols: list[str]) -> SynthesisMethod:
        """Learn from multiple partial surveys."""
        pass

    @abstractmethod
    def generate(self, n: int, seed: int | None = None) -> pd.DataFrame:
        """Generate n synthetic records."""
        pass

    def evaluate(
        self,
        holdouts: dict[str, pd.DataFrame],
        n_synthetic: int | None = None,
        k: int = 5,
    ) -> dict[str, EvalResult]:
        """Evaluate against holdouts."""
        if n_synthetic is None:
            n_synthetic = sum(len(h) for h in holdouts.values())

        synthetic = self.generate(n=n_synthetic)
        results = {}

        for name, holdout in holdouts.items():
            eval_cols = [c for c in holdout.columns if c in synthetic.columns]
            if len(eval_cols) < 2:
                continue

            holdout_vals = holdout[eval_cols].values
            synth_vals = synthetic[eval_cols].values

            holdout_mask = ~np.isnan(holdout_vals).any(axis=1)
            synth_mask = ~np.isnan(synth_vals).any(axis=1)

            prdc = compute_prdc(
                holdout_vals[holdout_mask],
                synth_vals[synth_mask],
                k=k,
            )

            results[name] = EvalResult(
                survey_name=name,
                coverage=prdc["coverage"],
                precision=prdc["precision"],
                recall=prdc["recall"],
                density=prdc["density"],
                n_holdout=holdout_mask.sum(),
                n_synthetic=synth_mask.sum(),
                columns_evaluated=eval_cols,
            )

        return results


class MeanImputationMethod(SynthesisMethod):
    """Baseline: Bootstrap shared + mean imputation for other columns."""

    name = "MeanImpute"

    def __init__(self):
        self.shared_cols_: list[str] = []
        self.all_cols_: list[str] = []
        self.col_means_: dict[str, float] = {}
        self.shared_data_: pd.DataFrame | None = None

    def fit(self, surveys: list[Survey], shared_cols: list[str]) -> MeanImputationMethod:
        self.shared_cols_ = list(shared_cols)

        # Collect all columns
        all_cols = set(shared_cols)
        for survey in surveys:
            all_cols.update(survey.columns)
        self.all_cols_ = list(all_cols)

        # Pool shared columns
        shared_dfs = []
        for survey in surveys:
            available = [c for c in shared_cols if c in survey.data.columns]
            if len(available) == len(shared_cols):
                shared_dfs.append(survey.data[shared_cols])
        self.shared_data_ = pd.concat(shared_dfs, ignore_index=True)

        # Compute means for all columns from whichever survey has them
        for survey in surveys:
            for col in survey.columns:
                if col not in self.col_means_:
                    self.col_means_[col] = survey.data[col].mean()

        return self

    def generate(self, n: int, seed: int | None = None) -> pd.DataFrame:
        rng = np.random.RandomState(seed or 42)

        # Bootstrap shared
        idx = rng.choice(len(self.shared_data_), size=n, replace=True)
        synthetic = self.shared_data_.iloc[idx].copy().reset_index(drop=True)

        # Add noise
        for col in self.shared_cols_:
            synthetic[col] += rng.normal(0, 0.1, n)

        # Mean imputation for other columns
        for col in self.all_cols_:
            if col not in synthetic.columns:
                synthetic[col] = self.col_means_.get(col, 0)

        return synthetic


class GaussianCopulaMethod(SynthesisMethod):
    """Gaussian Copula: Model marginals + correlation structure."""

    name = "GaussianCopula"

    def __init__(self):
        self.shared_cols_: list[str] = []
        self.all_cols_: list[str] = []
        self.marginals_: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # col -> (sorted_vals, quantiles)
        self.correlation_matrix_: np.ndarray | None = None
        self.col_order_: list[str] = []

    def fit(self, surveys: list[Survey], shared_cols: list[str]) -> GaussianCopulaMethod:
        from scipy import stats

        self.shared_cols_ = list(shared_cols)

        # Collect all columns and build complete dataset where possible
        all_cols = set(shared_cols)
        for survey in surveys:
            all_cols.update(survey.columns)
        self.all_cols_ = list(all_cols)
        self.col_order_ = list(all_cols)

        # Build pseudo-complete dataset using shared columns to join
        # Stack all data, fill missing with NaN
        all_data = []
        for survey in surveys:
            df = survey.data.copy()
            for col in self.all_cols_:
                if col not in df.columns:
                    df[col] = np.nan
            all_data.append(df[self.all_cols_])

        combined = pd.concat(all_data, ignore_index=True)

        # Fit marginals (empirical CDF for each column)
        for col in self.all_cols_:
            vals = combined[col].dropna().values
            if len(vals) > 0:
                sorted_vals = np.sort(vals)
                quantiles = np.linspace(0, 1, len(sorted_vals))
                self.marginals_[col] = (sorted_vals, quantiles)

        # Estimate correlation on shared columns (which are complete)
        shared_data = combined[shared_cols].dropna()
        if len(shared_data) > 10:
            # Transform to normal via empirical CDF
            normal_data = np.zeros((len(shared_data), len(shared_cols)))
            for i, col in enumerate(shared_cols):
                vals = shared_data[col].values
                # Empirical quantiles
                ranks = stats.rankdata(vals) / (len(vals) + 1)
                normal_data[:, i] = stats.norm.ppf(ranks)

            # Estimate correlation
            self.correlation_matrix_ = np.corrcoef(normal_data.T)
            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(self.correlation_matrix_)
            eigvals = np.maximum(eigvals, 1e-6)
            self.correlation_matrix_ = eigvecs @ np.diag(eigvals) @ eigvecs.T

        return self

    def generate(self, n: int, seed: int | None = None) -> pd.DataFrame:
        from scipy import stats

        rng = np.random.RandomState(seed or 42)

        # Sample from multivariate normal for shared columns
        if self.correlation_matrix_ is not None:
            z = rng.multivariate_normal(
                np.zeros(len(self.shared_cols_)),
                self.correlation_matrix_,
                size=n,
            )
            u = stats.norm.cdf(z)  # Transform to uniform
        else:
            u = rng.uniform(0, 1, (n, len(self.shared_cols_)))

        # Transform to marginals for shared columns
        synthetic = pd.DataFrame()
        for i, col in enumerate(self.shared_cols_):
            if col in self.marginals_:
                sorted_vals, quantiles = self.marginals_[col]
                synthetic[col] = np.interp(u[:, i], quantiles, sorted_vals)

        # For non-shared columns, sample independently from marginal
        for col in self.all_cols_:
            if col not in synthetic.columns and col in self.marginals_:
                sorted_vals, quantiles = self.marginals_[col]
                u_col = rng.uniform(0, 1, n)
                synthetic[col] = np.interp(u_col, quantiles, sorted_vals)

        return synthetic


class VAEMethod(SynthesisMethod):
    """Variational Autoencoder with masked training for partial observations."""

    name = "VAE"

    def __init__(self, latent_dim: int = 16, hidden_dim: int = 64, epochs: int = 100):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.all_cols_: list[str] = []
        self.model_ = None
        self.col_stats_: dict[str, tuple[float, float]] = {}

    def fit(self, surveys: list[Survey], shared_cols: list[str]) -> VAEMethod:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        # Collect all columns
        all_cols = set(shared_cols)
        for survey in surveys:
            all_cols.update(survey.columns)
        self.all_cols_ = sorted(list(all_cols))
        n_features = len(self.all_cols_)

        # Build training data with masks
        all_data = []
        all_masks = []
        for survey in surveys:
            n_rows = len(survey.data)
            data = np.zeros((n_rows, n_features))
            mask = np.zeros((n_rows, n_features))

            for i, col in enumerate(self.all_cols_):
                if col in survey.data.columns:
                    data[:, i] = survey.data[col].values
                    mask[:, i] = 1.0

            all_data.append(data)
            all_masks.append(mask)

        X = np.vstack(all_data).astype(np.float32)
        M = np.vstack(all_masks).astype(np.float32)

        # Normalize
        for i, col in enumerate(self.all_cols_):
            observed = X[:, i][M[:, i] > 0]
            if len(observed) > 0:
                mean, std = observed.mean(), observed.std() + 1e-6
                self.col_stats_[col] = (mean, std)
                X[:, i] = (X[:, i] - mean) / std

        # Simple VAE
        class VAE(nn.Module):
            def __init__(self, n_features, latent_dim, hidden_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(n_features, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
                self.fc_mu = nn.Linear(hidden_dim, latent_dim)
                self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, n_features),
                )

            def encode(self, x):
                h = self.encoder(x)
                return self.fc_mu(h), self.fc_logvar(h)

            def reparameterize(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std

            def decode(self, z):
                return self.decoder(z)

            def forward(self, x):
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                return self.decode(z), mu, logvar

        model = VAE(n_features, self.latent_dim, self.hidden_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        X_tensor = torch.tensor(X)
        M_tensor = torch.tensor(M)
        dataset = TensorDataset(X_tensor, M_tensor)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, batch_m in loader:
                optimizer.zero_grad()

                # Fill unobserved with zeros for input
                x_input = batch_x * batch_m

                recon, mu, logvar = model(x_input)

                # Masked reconstruction loss
                recon_loss = ((recon - batch_x) ** 2 * batch_m).sum() / batch_m.sum()
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

                loss = recon_loss + 0.1 * kl_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        model.eval()
        self.model_ = model
        return self

    def generate(self, n: int, seed: int | None = None) -> pd.DataFrame:
        import torch

        if self.model_ is None:
            raise RuntimeError("Must call fit() first")

        torch.manual_seed(seed or 42)

        with torch.no_grad():
            z = torch.randn(n, self.latent_dim)
            X = self.model_.decode(z).numpy()

        # Denormalize
        synthetic = pd.DataFrame()
        for i, col in enumerate(self.all_cols_):
            if col in self.col_stats_:
                mean, std = self.col_stats_[col]
                synthetic[col] = X[:, i] * std + mean
            else:
                synthetic[col] = X[:, i]

        return synthetic


class CTGANMethod(SynthesisMethod):
    """CTGAN from SDV library."""

    name = "CTGAN"

    def __init__(self, epochs: int = 100):
        self.epochs = epochs
        self.all_cols_: list[str] = []
        self.model_ = None

    def fit(self, surveys: list[Survey], shared_cols: list[str]) -> CTGANMethod:
        try:
            from sdv.metadata import SingleTableMetadata
            from sdv.single_table import CTGANSynthesizer
        except ImportError:
            raise ImportError("sdv required: pip install sdv")

        # Collect all columns
        all_cols = set(shared_cols)
        for survey in surveys:
            all_cols.update(survey.columns)
        self.all_cols_ = list(all_cols)

        # Stack data, fill missing with median
        all_data = []
        for survey in surveys:
            df = survey.data.copy()
            for col in self.all_cols_:
                if col not in df.columns:
                    df[col] = np.nan
            all_data.append(df[self.all_cols_])

        combined = pd.concat(all_data, ignore_index=True)

        # Fill NaN with column median
        for col in self.all_cols_:
            median = combined[col].median()
            combined[col] = combined[col].fillna(median)

        # Fit CTGAN
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(combined)

        self.model_ = CTGANSynthesizer(metadata, epochs=self.epochs, verbose=False)
        self.model_.fit(combined)

        return self

    def generate(self, n: int, seed: int | None = None) -> pd.DataFrame:
        if self.model_ is None:
            raise RuntimeError("Must call fit() first")

        return self.model_.sample(n)


def run_method_comparison(
    surveys: list[Survey],
    shared_cols: list[str],
    methods: list[SynthesisMethod] | None = None,
    holdout_frac: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """Run multiple methods and compare on same train/holdout split.

    Returns DataFrame with coverage/precision/recall per method per survey.
    """
    from microplex.dgp import PopulationDGP

    rng = np.random.RandomState(seed)

    # Split each survey
    train_surveys = []
    holdouts = {}

    for survey in surveys:
        n = len(survey.data)
        n_holdout = int(n * holdout_frac)
        indices = rng.permutation(n)

        train_data = survey.data.iloc[indices[n_holdout:]].reset_index(drop=True)
        holdout_data = survey.data.iloc[indices[:n_holdout]].reset_index(drop=True)

        train_surveys.append(Survey(survey.name, train_data, survey.columns))
        holdouts[survey.name] = holdout_data

    # Default methods
    if methods is None:
        methods = [
            PopulationDGP(random_state=seed),  # QRF baseline
            MeanImputationMethod(),
            GaussianCopulaMethod(),
            VAEMethod(epochs=50),
        ]

        # Try CTGAN if available
        try:
            methods.append(CTGANMethod(epochs=50))
        except ImportError:
            pass

    # Run each method
    all_results = []

    for method in methods:
        method_name = getattr(method, "name", method.__class__.__name__)
        print(f"Running {method_name}...")

        try:
            # Handle PopulationDGP vs other methods
            if hasattr(method, "fit"):
                if isinstance(method, PopulationDGP):
                    method.fit(train_surveys, shared_cols)
                else:
                    method.fit(train_surveys, shared_cols)

            results = method.evaluate(holdouts)

            for survey_name, result in results.items():
                all_results.append({
                    "method": method_name,
                    "survey": survey_name,
                    "coverage": result.coverage,
                    "precision": result.precision,
                    "recall": result.recall,
                    "n_cols": len(result.columns_evaluated),
                })

        except Exception as e:
            print(f"  Error: {e}")
            continue

    df = pd.DataFrame(all_results)

    # Print summary
    print()
    print("=" * 70)
    print(f"{'Method':<20} {'Survey':<15} {'Coverage':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 70)
    for _, row in df.iterrows():
        print(f"{row['method']:<20} {row['survey']:<15} {row['coverage']:>10.1%} "
              f"{row['precision']:>10.1%} {row['recall']:>10.1%}")
    print("=" * 70)

    # Average coverage per method
    print()
    print("Average Coverage by Method:")
    for method_name in df["method"].unique():
        avg = df[df["method"] == method_name]["coverage"].mean()
        print(f"  {method_name}: {avg:.1%}")

    return df
