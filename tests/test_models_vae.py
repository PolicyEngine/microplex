"""
Tests for models/trajectory_vae.py - Variational Autoencoder for trajectories.

TDD: Define expected behavior before implementation.
"""

import numpy as np
import pandas as pd
import pytest

from microplex.models.base import ImputationResult, SyntheticPopulation


class TestTrajectoryVAEInterface:
    """Tests for TrajectoryVAE conforming to BaseSynthesisModel."""

    @pytest.fixture
    def sample_panel_data(self):
        """Create sample panel data for testing."""
        np.random.seed(42)
        n_persons, T, _n_features = 100, 12, 4

        # Create panel DataFrame
        data = []
        for pid in range(n_persons):
            base_age = np.random.randint(20, 60)
            base_income = np.random.lognormal(10, 1)
            base_wealth = np.random.lognormal(11, 1.5)

            for t in range(T):
                data.append({
                    "person_id": pid,
                    "period": t,
                    "age": base_age + t / 12,
                    "income": base_income * (1 + np.random.normal(0, 0.1)),
                    "wealth": base_wealth * (1 + np.random.normal(0, 0.05)),
                    "employed": int(np.random.random() > 0.1),
                })

        return pd.DataFrame(data)

    @pytest.fixture
    def vae_model(self):
        """Create VAE model instance."""
        from microplex.models.trajectory_vae import TrajectoryVAE
        return TrajectoryVAE(
            n_features=4,
            latent_dim=16,
            hidden_dim=64,
            n_layers=2,
        )

    def test_can_instantiate(self, vae_model):
        """TrajectoryVAE can be instantiated."""
        from microplex.models.trajectory_vae import TrajectoryVAE
        assert isinstance(vae_model, TrajectoryVAE)

    def test_implements_base_interface(self, vae_model):
        """TrajectoryVAE implements BaseSynthesisModel interface."""
        from microplex.models.base import BaseSynthesisModel
        assert isinstance(vae_model, BaseSynthesisModel)

    def test_fit_returns_self(self, vae_model, sample_panel_data):
        """fit() returns self for chaining."""
        result = vae_model.fit(sample_panel_data, epochs=1)
        assert result is vae_model

    def test_fit_sets_fitted_flag(self, vae_model, sample_panel_data):
        """fit() sets internal fitted state."""
        vae_model.fit(sample_panel_data, epochs=1)
        assert vae_model.is_fitted

    def test_generate_returns_synthetic_population(self, vae_model, sample_panel_data):
        """generate() returns SyntheticPopulation."""
        vae_model.fit(sample_panel_data, epochs=1)
        result = vae_model.generate(n=50, T=12)
        assert isinstance(result, SyntheticPopulation)

    def test_generate_correct_shape(self, vae_model, sample_panel_data):
        """generate() produces correct number of records."""
        vae_model.fit(sample_panel_data, epochs=1)
        result = vae_model.generate(n=50, T=12)
        assert result.n_persons == 50
        assert result.n_periods == 12

    def test_generate_has_all_columns(self, vae_model, sample_panel_data):
        """generate() includes all training columns."""
        vae_model.fit(sample_panel_data, epochs=1)
        result = vae_model.generate(n=50, T=12)
        for col in ["age", "income", "wealth", "employed"]:
            assert col in result.persons.columns

    def test_impute_returns_imputation_result(self, vae_model, sample_panel_data):
        """impute() returns ImputationResult."""
        vae_model.fit(sample_panel_data, epochs=1)

        # Partial observation: age and income known, wealth and employed missing
        partial = pd.DataFrame({
            "age": [30, 40, 50],
            "income": [50000, 80000, 100000],
            "wealth": [np.nan, np.nan, np.nan],
            "employed": [np.nan, np.nan, np.nan],
        })

        result = vae_model.impute(partial, n_samples=10)
        assert isinstance(result, ImputationResult)
        assert result.n_samples == 10

    def test_log_prob_returns_array(self, vae_model, sample_panel_data):
        """log_prob() returns array of log probabilities."""
        vae_model.fit(sample_panel_data, epochs=1)
        log_probs = vae_model.log_prob(sample_panel_data)
        assert isinstance(log_probs, np.ndarray)
        assert len(log_probs) == len(sample_panel_data)


class TestTrajectoryVAEEncoding:
    """Tests for VAE encoding functionality."""

    @pytest.fixture
    def fitted_vae(self):
        """Create and fit a VAE."""
        from microplex.models.trajectory_vae import TrajectoryVAE

        np.random.seed(42)
        n_persons, T = 50, 8

        data = []
        for pid in range(n_persons):
            for t in range(T):
                data.append({
                    "person_id": pid,
                    "period": t,
                    "age": 30 + t / 12,
                    "income": 50000 + np.random.normal(0, 5000),
                    "wealth": 100000 + np.random.normal(0, 10000),
                })

        df = pd.DataFrame(data)

        vae = TrajectoryVAE(n_features=3, latent_dim=8, hidden_dim=32, n_layers=1)
        vae.fit(df, epochs=2)
        return vae, df

    def test_encode_returns_latent(self, fitted_vae):
        """encode() returns latent representation."""
        vae, df = fitted_vae
        latent = vae.encode(df)

        assert isinstance(latent, np.ndarray)
        # Should have one latent vector per person
        n_persons = df["person_id"].nunique()
        assert latent.shape == (n_persons, vae.latent_dim)

    def test_encode_deterministic_mean(self, fitted_vae):
        """encode() with deterministic=True returns mean (no sampling)."""
        vae, df = fitted_vae

        latent1 = vae.encode(df, deterministic=True)
        latent2 = vae.encode(df, deterministic=True)

        np.testing.assert_array_almost_equal(latent1, latent2)

    def test_encode_stochastic_varies(self, fitted_vae):
        """encode() with deterministic=False samples from posterior."""
        vae, df = fitted_vae

        latent1 = vae.encode(df, deterministic=False)
        latent2 = vae.encode(df, deterministic=False)

        # Should be different (with very high probability)
        assert not np.allclose(latent1, latent2)

    def test_embeddings_for_coverage(self, fitted_vae):
        """Embeddings can be used for coverage computation."""
        from microplex.eval.coverage import compute_prdc

        vae, df = fitted_vae

        # Generate synthetic
        synthetic = vae.generate(n=30, T=8)

        # Get embeddings
        real_emb = vae.encode(df, deterministic=True)
        synth_emb = vae.encode(synthetic.persons, deterministic=True)

        # Compute coverage in embedding space
        result = compute_prdc(real_emb, synth_emb, k=3)

        assert 0 <= result.coverage <= 1
        assert 0 <= result.precision <= 1


class TestTrajectoryVAEReconstruction:
    """Tests for VAE reconstruction quality."""

    @pytest.fixture
    def trained_vae(self):
        """Train VAE for longer to test reconstruction."""
        from microplex.models.trajectory_vae import TrajectoryVAE

        np.random.seed(42)
        n_persons, T = 100, 12

        data = []
        for pid in range(n_persons):
            base = np.random.randn(3) * 10
            for t in range(T):
                data.append({
                    "person_id": pid,
                    "period": t,
                    "x": base[0] + np.random.normal(0, 1),
                    "y": base[1] + np.random.normal(0, 1),
                    "z": base[2] + np.random.normal(0, 1),
                })

        df = pd.DataFrame(data)

        vae = TrajectoryVAE(n_features=3, latent_dim=16, hidden_dim=64, n_layers=2)
        vae.fit(df, epochs=10)
        return vae, df

    def test_reconstruction_quality(self, trained_vae):
        """Reconstruction should be close to original."""
        vae, df = trained_vae

        reconstructed = vae.reconstruct(df)

        # Compute reconstruction error
        original = df[["x", "y", "z"]].values
        recon = reconstructed[["x", "y", "z"]].values

        mse = np.mean((original - recon) ** 2)

        # After training, MSE should be reasonable
        # (not perfect but capturing structure)
        assert mse < np.var(original)  # Better than predicting mean


class TestTrajectoryVAEMaskedInput:
    """Tests for handling missing data / multi-survey fusion."""

    @pytest.fixture
    def multi_survey_data(self):
        """Create data simulating multiple surveys with different columns."""
        np.random.seed(42)

        # Survey A: has age, income (no wealth)
        survey_a = pd.DataFrame({
            "person_id": range(50),
            "period": [0] * 50,
            "age": np.random.randint(20, 60, 50),
            "income": np.random.lognormal(10, 1, 50),
            "wealth": [np.nan] * 50,  # Missing
        })

        # Survey B: has age, wealth (no income)
        survey_b = pd.DataFrame({
            "person_id": range(50, 100),
            "period": [0] * 50,
            "age": np.random.randint(20, 60, 50),
            "income": [np.nan] * 50,  # Missing
            "wealth": np.random.lognormal(11, 1.5, 50),
        })

        return pd.concat([survey_a, survey_b], ignore_index=True)

    def test_fit_with_missing_data(self, multi_survey_data):
        """VAE can fit on data with missing values."""
        from microplex.models.trajectory_vae import TrajectoryVAE

        vae = TrajectoryVAE(n_features=3, latent_dim=8, hidden_dim=32)
        vae.fit(multi_survey_data, epochs=1)

        assert vae.is_fitted

    def test_generate_complete_from_partial(self, multi_survey_data):
        """VAE generates complete data even when trained on partial."""
        from microplex.models.trajectory_vae import TrajectoryVAE

        vae = TrajectoryVAE(n_features=3, latent_dim=8, hidden_dim=32)
        vae.fit(multi_survey_data, epochs=1)

        result = vae.generate(n=20, T=1)

        # Should have no missing values
        assert not result.persons[["age", "income", "wealth"]].isna().any().any()


class TestTrajectoryVAEUncertainty:
    """Tests for uncertainty quantification."""

    @pytest.fixture
    def fitted_vae(self):
        """Create fitted VAE."""
        from microplex.models.trajectory_vae import TrajectoryVAE

        np.random.seed(42)
        data = pd.DataFrame({
            "person_id": list(range(50)) * 6,
            "period": [t for _ in range(50) for t in range(6)],
            "income": np.random.lognormal(10, 1, 300),
            "wealth": np.random.lognormal(11, 1.5, 300),
        })

        vae = TrajectoryVAE(n_features=2, latent_dim=8, hidden_dim=32)
        vae.fit(data, epochs=2)
        return vae

    def test_multiple_samples_vary(self, fitted_vae):
        """Multiple generation samples should vary (uncertainty)."""
        vae = fitted_vae

        sample1 = vae.generate(n=20, T=6, seed=1)
        sample2 = vae.generate(n=20, T=6, seed=2)

        # Different seeds should give different samples
        assert not np.allclose(
            sample1.persons["income"].values,
            sample2.persons["income"].values
        )

    def test_imputation_samples_vary(self, fitted_vae):
        """Imputation samples should show uncertainty."""
        vae = fitted_vae

        partial = pd.DataFrame({
            "income": [50000, 80000],
            "wealth": [np.nan, np.nan],
        })

        result = vae.impute(partial, n_samples=50)

        # Check variance across samples
        wealth_samples = result.samples.groupby("_input_row_id")["wealth"]
        stds = wealth_samples.std()

        # Should have non-zero variance
        assert (stds > 0).all()
