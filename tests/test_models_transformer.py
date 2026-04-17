"""
Tests for models/trajectory_transformer.py - Autoregressive Transformer for trajectories.

TDD: Define expected behavior before implementation.
"""

import numpy as np
import pandas as pd
import pytest

from microplex.models.base import SyntheticPopulation


class TestTrajectoryTransformerInterface:
    """Tests for TrajectoryTransformer conforming to BaseSynthesisModel."""

    @pytest.fixture
    def sample_panel_data(self):
        """Create sample panel data for testing."""
        np.random.seed(42)
        n_persons, T, _n_features = 100, 12, 3

        data = []
        for pid in range(n_persons):
            base_income = np.random.lognormal(10, 1)
            for t in range(T):
                # Autoregressive dynamics
                if t == 0:
                    income = base_income
                else:
                    income = income * (1 + np.random.normal(0.01, 0.05))

                data.append({
                    "person_id": pid,
                    "period": t,
                    "age": 30 + t / 12,
                    "income": income,
                    "wealth": income * 3 + np.random.normal(0, 10000),
                })

        return pd.DataFrame(data)

    @pytest.fixture
    def transformer_model(self):
        """Create Transformer model instance."""
        from microplex.models.trajectory_transformer import TrajectoryTransformer
        return TrajectoryTransformer(
            n_features=3,
            hidden_dim=64,
            n_heads=4,
            n_layers=2,
        )

    def test_can_instantiate(self, transformer_model):
        """TrajectoryTransformer can be instantiated."""
        from microplex.models.trajectory_transformer import TrajectoryTransformer
        assert isinstance(transformer_model, TrajectoryTransformer)

    def test_implements_base_interface(self, transformer_model):
        """TrajectoryTransformer implements BaseSynthesisModel."""
        from microplex.models.base import BaseSynthesisModel
        assert isinstance(transformer_model, BaseSynthesisModel)

    def test_fit_returns_self(self, transformer_model, sample_panel_data):
        """fit() returns self for chaining."""
        result = transformer_model.fit(sample_panel_data, epochs=1)
        assert result is transformer_model

    def test_generate_returns_synthetic_population(self, transformer_model, sample_panel_data):
        """generate() returns SyntheticPopulation."""
        transformer_model.fit(sample_panel_data, epochs=1)
        result = transformer_model.generate(n=50, T=12)
        assert isinstance(result, SyntheticPopulation)

    def test_generate_correct_shape(self, transformer_model, sample_panel_data):
        """generate() produces correct dimensions."""
        transformer_model.fit(sample_panel_data, epochs=1)
        result = transformer_model.generate(n=50, T=8)
        assert result.n_persons == 50
        assert result.n_periods == 8


class TestTrajectoryTransformerAutoregressive:
    """Tests for autoregressive generation."""

    @pytest.fixture
    def fitted_transformer(self):
        """Create and fit a Transformer."""
        from microplex.models.trajectory_transformer import TrajectoryTransformer

        np.random.seed(42)
        n_persons, T = 80, 10

        data = []
        for pid in range(n_persons):
            income = np.random.lognormal(10, 0.5)
            for t in range(T):
                data.append({
                    "person_id": pid,
                    "period": t,
                    "income": income,
                    "wealth": income * 2,
                })
                income = income * (1 + np.random.normal(0.02, 0.03))

        df = pd.DataFrame(data)

        model = TrajectoryTransformer(n_features=2, hidden_dim=32, n_heads=2, n_layers=1)
        model.fit(df, epochs=3)
        return model, df

    def test_generates_trajectories_autoregressively(self, fitted_transformer):
        """Each time step should depend on previous steps."""
        model, _ = fitted_transformer

        # Generate with same initial conditions but different seeds
        pop1 = model.generate(n=20, T=10, seed=1)
        pop2 = model.generate(n=20, T=10, seed=2)

        # Later time steps should diverge more than early ones
        df1 = pop1.persons
        df2 = pop2.persons

        early_diff = np.abs(
            df1[df1["period"] == 1]["income"].values -
            df2[df2["period"] == 1]["income"].values
        ).mean()

        late_diff = np.abs(
            df1[df1["period"] == 9]["income"].values -
            df2[df2["period"] == 9]["income"].values
        ).mean()

        # Later differences should be larger (uncertainty grows)
        assert late_diff >= early_diff * 0.5  # Allow some tolerance

    def test_respects_initial_conditions(self, fitted_transformer):
        """Can condition on initial state."""
        model, _ = fitted_transformer

        # Generate from specific initial condition
        initial = pd.DataFrame({
            "income": [100000, 50000],
            "wealth": [200000, 100000],
        })

        pop = model.generate_from_initial(initial, T=6)

        assert pop.n_persons == 2
        assert pop.n_periods == 6

        # First period should match initial
        first_period = pop.persons[pop.persons["period"] == 0]
        np.testing.assert_array_almost_equal(
            first_period["income"].values,
            initial["income"].values,
            decimal=0
        )


class TestTrajectoryTransformerTemporalDynamics:
    """Tests for capturing temporal patterns."""

    @pytest.fixture
    def trending_data(self):
        """Create data with clear temporal trends."""
        np.random.seed(42)
        n_persons, T = 50, 12

        data = []
        for pid in range(n_persons):
            base = np.random.uniform(100, 200)
            growth_rate = np.random.uniform(0.01, 0.05)

            for t in range(T):
                # Exponential growth with noise
                value = base * np.exp(growth_rate * t) * (1 + np.random.normal(0, 0.02))
                data.append({
                    "person_id": pid,
                    "period": t,
                    "value": value,
                })

        return pd.DataFrame(data)

    def test_captures_growth_trend(self, trending_data):
        """Transformer should capture positive growth trends."""
        from microplex.models.trajectory_transformer import TrajectoryTransformer

        model = TrajectoryTransformer(n_features=1, hidden_dim=32, n_heads=2, n_layers=2)
        model.fit(trending_data, epochs=5)

        # Generate new trajectories
        pop = model.generate(n=30, T=12, seed=42)

        # Check that values tend to increase over time
        df = pop.persons
        mean_by_period = df.groupby("period")["value"].mean()

        # Later periods should have higher values on average
        assert mean_by_period.iloc[-1] > mean_by_period.iloc[0]

    def test_maintains_autocorrelation(self, trending_data):
        """Generated trajectories should have positive autocorrelation."""
        from microplex.models.trajectory_transformer import TrajectoryTransformer

        model = TrajectoryTransformer(n_features=1, hidden_dim=32, n_heads=2, n_layers=2)
        model.fit(trending_data, epochs=5)

        pop = model.generate(n=30, T=12, seed=42)

        # Compute autocorrelation for each person
        autocorrs = []
        for pid in pop.persons["person_id"].unique():
            person = pop.persons[pop.persons["person_id"] == pid].sort_values("period")
            values = person["value"].values
            if len(values) > 1:
                autocorr = np.corrcoef(values[:-1], values[1:])[0, 1]
                if not np.isnan(autocorr):
                    autocorrs.append(autocorr)

        # Should have positive autocorrelation on average
        # (with limited training, may not be strong)
        assert np.mean(autocorrs) > 0.0


class TestTrajectoryTransformerEmbedding:
    """Tests for embedding extraction."""

    @pytest.fixture
    def fitted_model(self):
        """Create fitted transformer."""
        from microplex.models.trajectory_transformer import TrajectoryTransformer

        np.random.seed(42)
        data = pd.DataFrame({
            "person_id": [pid for pid in range(40) for _ in range(8)],
            "period": [t for _ in range(40) for t in range(8)],
            "x": np.random.randn(320),
            "y": np.random.randn(320),
        })

        model = TrajectoryTransformer(n_features=2, hidden_dim=32, n_heads=2, n_layers=1)
        model.fit(data, epochs=2)
        return model, data

    def test_encode_returns_embeddings(self, fitted_model):
        """encode() returns trajectory embeddings."""
        model, df = fitted_model

        embeddings = model.encode(df)

        n_persons = df["person_id"].nunique()
        assert embeddings.shape[0] == n_persons
        assert embeddings.shape[1] == model.hidden_dim

    def test_embeddings_for_coverage(self, fitted_model):
        """Embeddings can be used with PRDC coverage."""
        from microplex.eval.coverage import compute_prdc

        model, df = fitted_model

        # Generate synthetic
        synthetic = model.generate(n=20, T=8)

        # Get embeddings
        real_emb = model.encode(df)
        synth_emb = model.encode(synthetic.persons)

        # Compute coverage
        result = compute_prdc(real_emb, synth_emb, k=3)

        assert 0 <= result.coverage <= 1


class TestTrajectoryTransformerMaskedData:
    """Tests for handling missing data."""

    @pytest.fixture
    def partial_panel(self):
        """Create panel with some missing values."""
        np.random.seed(42)
        n_persons, T = 40, 8

        data = []
        for pid in range(n_persons):
            for t in range(T):
                record = {
                    "person_id": pid,
                    "period": t,
                    "x": np.random.randn(),
                }
                # y missing 30% of the time
                if np.random.random() > 0.3:
                    record["y"] = np.random.randn()
                else:
                    record["y"] = np.nan

                data.append(record)

        return pd.DataFrame(data)

    def test_fit_with_missing(self, partial_panel):
        """Transformer can train on data with missing values."""
        from microplex.models.trajectory_transformer import TrajectoryTransformer

        model = TrajectoryTransformer(n_features=2, hidden_dim=32, n_heads=2, n_layers=1)
        model.fit(partial_panel, epochs=1)

        assert model.is_fitted

    def test_generate_complete(self, partial_panel):
        """Generate complete data even from partial training."""
        from microplex.models.trajectory_transformer import TrajectoryTransformer

        model = TrajectoryTransformer(n_features=2, hidden_dim=32, n_heads=2, n_layers=1)
        model.fit(partial_panel, epochs=1)

        pop = model.generate(n=10, T=8)

        # No missing values in output
        assert not pop.persons[["x", "y"]].isna().any().any()
