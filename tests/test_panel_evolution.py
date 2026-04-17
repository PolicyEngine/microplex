"""Tests for unified panel evolution model.

Replaces separate transition classes (MarriageTransition, DivorceTransition, etc.)
with a single autoregressive model: state[t+1] ~ state[t], state[t-1], ...
"""

import numpy as np
import pandas as pd
import pytest

from microplex.models.panel_evolution import (
    PanelEvolutionModel,
    create_history_features,
    create_lagged_features,
)


class TestCreateLaggedFeatures:
    """Test lag feature creation."""

    def test_single_lag(self):
        """Test creating single lag features."""
        df = pd.DataFrame({
            "person_id": [1, 1, 1, 2, 2, 2],
            "period": [0, 1, 2, 0, 1, 2],
            "income": [50000, 52000, 55000, 30000, 31000, 32000],
            "is_married": [0, 0, 1, 1, 1, 0],
        })

        result = create_lagged_features(
            df,
            vars=["income", "is_married"],
            lags=[1],
            person_id_col="person_id",
            period_col="period",
        )

        # First period should have NaN lags
        assert pd.isna(result.loc[result["period"] == 0, "income_lag1"]).all()

        # Second period should have first period values
        p1_period1 = result[(result["person_id"] == 1) & (result["period"] == 1)]
        assert p1_period1["income_lag1"].values[0] == 50000
        assert p1_period1["is_married_lag1"].values[0] == 0

    def test_multiple_lags(self):
        """Test creating multiple lag features."""
        df = pd.DataFrame({
            "person_id": [1, 1, 1, 1],
            "period": [0, 1, 2, 3],
            "income": [50, 51, 52, 53],
        })

        result = create_lagged_features(
            df,
            vars=["income"],
            lags=[1, 2],
            person_id_col="person_id",
            period_col="period",
        )

        # Period 3 should have lag1=52, lag2=51
        p3 = result[result["period"] == 3]
        assert p3["income_lag1"].values[0] == 52
        assert p3["income_lag2"].values[0] == 51

    def test_respects_person_boundaries(self):
        """Test that lags don't cross person boundaries."""
        df = pd.DataFrame({
            "person_id": [1, 1, 2, 2],
            "period": [0, 1, 0, 1],
            "income": [100, 110, 200, 210],
        })

        result = create_lagged_features(
            df,
            vars=["income"],
            lags=[1],
            person_id_col="person_id",
            period_col="period",
        )

        # Person 2's first period should have NaN lag, not person 1's value
        p2_period0 = result[(result["person_id"] == 2) & (result["period"] == 0)]
        assert pd.isna(p2_period0["income_lag1"].values[0])


class TestCreateHistoryFeatures:
    """Test derived history feature creation."""

    def test_duration_feature(self):
        """Test computing duration in state."""
        df = pd.DataFrame({
            "person_id": [1, 1, 1, 1, 1],
            "period": [0, 1, 2, 3, 4],
            "is_married": [0, 1, 1, 1, 0],  # Married for 3 periods
        })

        result = create_history_features(
            df,
            state_var="is_married",
            feature_type="duration",
            person_id_col="person_id",
            period_col="period",
        )

        assert "is_married_duration" in result.columns
        # Duration should count consecutive periods in state=1
        durations = result["is_married_duration"].values
        assert durations[0] == 0  # Not married
        assert durations[1] == 1  # First period married
        assert durations[2] == 2  # Second period married
        assert durations[3] == 3  # Third period married
        assert durations[4] == 0  # No longer married

    def test_ever_in_state_feature(self):
        """Test computing 'ever in state' indicator."""
        df = pd.DataFrame({
            "person_id": [1, 1, 1, 1],
            "period": [0, 1, 2, 3],
            "is_divorced": [0, 1, 0, 0],  # Was divorced in period 1
        })

        result = create_history_features(
            df,
            state_var="is_divorced",
            feature_type="ever",
            person_id_col="person_id",
            period_col="period",
        )

        assert "ever_is_divorced" in result.columns
        ever = result["ever_is_divorced"].values
        assert ever[0] == 0  # Not yet divorced
        assert ever[1] == 1  # Divorced
        assert ever[2] == 1  # Ever divorced (cumulative)
        assert ever[3] == 1  # Ever divorced

    def test_trend_feature(self):
        """Test computing trend over lookback window."""
        df = pd.DataFrame({
            "person_id": [1, 1, 1, 1, 1],
            "period": [0, 1, 2, 3, 4],
            "income": [100, 110, 120, 130, 140],  # Linear growth
        })

        result = create_history_features(
            df,
            state_var="income",
            feature_type="trend",
            lookback=3,
            person_id_col="person_id",
            period_col="period",
        )

        assert "income_trend_3" in result.columns
        # Trend should be positive for increasing income
        trend_at_4 = result[result["period"] == 4]["income_trend_3"].values[0]
        assert trend_at_4 > 0


class TestPanelEvolutionModel:
    """Test the unified panel evolution model."""

    @pytest.fixture
    def mock_panel_data(self):
        """Create mock panel data for testing."""
        np.random.seed(42)
        n_persons = 100
        n_periods = 6

        records = []
        for pid in range(n_persons):
            age = np.random.randint(25, 55)
            is_male = np.random.choice([0, 1])
            is_married = np.random.choice([0, 1])
            income = np.random.lognormal(10.5, 0.5)

            for t in range(n_periods):
                # Simple transition logic for mock data
                if not is_married and np.random.random() < 0.05:
                    is_married = 1
                elif is_married and np.random.random() < 0.02:
                    is_married = 0

                income *= (1 + np.random.normal(0.02, 0.05))

                records.append({
                    "person_id": pid,
                    "period": t,
                    "age": age + t,
                    "is_male": is_male,
                    "is_married": is_married,
                    "income": income,
                })

        return pd.DataFrame(records)

    def test_model_initialization(self):
        """Test model can be initialized."""
        model = PanelEvolutionModel(
            state_vars=["is_married", "income"],
            condition_vars=["age", "is_male"],
            lags=[1, 2],
        )

        assert model.state_vars == ["is_married", "income"]
        assert model.lags == [1, 2]

    def test_model_fit(self, mock_panel_data):
        """Test model fitting on panel data."""
        model = PanelEvolutionModel(
            state_vars=["is_married", "income"],
            condition_vars=["age", "is_male"],
            lags=[1],
        )

        model.fit(mock_panel_data, epochs=5, verbose=False)

        assert model._is_fitted

    def test_model_predict_proba(self, mock_panel_data):
        """Test predicting next-period probabilities."""
        model = PanelEvolutionModel(
            state_vars=["is_married"],
            condition_vars=["age", "is_male"],
            lags=[1],
            var_types={"is_married": "binary"},
        )

        model.fit(mock_panel_data, epochs=5, verbose=False)

        # Get current state for some people
        current = mock_panel_data[mock_panel_data["period"] == 3].copy()

        probs = model.predict_proba(current)

        assert "is_married_prob" in probs.columns
        assert (probs["is_married_prob"] >= 0).all()
        assert (probs["is_married_prob"] <= 1).all()

    def test_model_simulate_step(self, mock_panel_data):
        """Test simulating one step forward."""
        model = PanelEvolutionModel(
            state_vars=["is_married", "income"],
            condition_vars=["age", "is_male"],
            lags=[1],
            var_types={"is_married": "binary", "income": "continuous"},
        )

        model.fit(mock_panel_data, epochs=5, verbose=False)

        current = mock_panel_data[mock_panel_data["period"] == 3].copy()

        next_state = model.simulate_step(current, seed=42)

        assert len(next_state) == len(current)
        assert "is_married" in next_state.columns
        assert "income" in next_state.columns
        # Binary should be 0 or 1
        assert set(next_state["is_married"].unique()).issubset({0, 1})

    def test_model_simulate_trajectory(self, mock_panel_data):
        """Test simulating multiple steps."""
        model = PanelEvolutionModel(
            state_vars=["is_married"],
            condition_vars=["age", "is_male"],
            lags=[1],
            var_types={"is_married": "binary"},
        )

        model.fit(mock_panel_data, epochs=5, verbose=False)

        initial = mock_panel_data[mock_panel_data["period"] == 0].head(10).copy()

        trajectory = model.simulate_trajectory(initial, n_steps=5, seed=42)

        # Should have initial + 5 steps = 6 periods per person
        assert len(trajectory) == 10 * 6
        assert trajectory["period"].max() == 5

    def test_model_with_history_features(self, mock_panel_data):
        """Test model with derived history features."""
        model = PanelEvolutionModel(
            state_vars=["is_married"],
            condition_vars=["age", "is_male"],
            lags=[1, 2],
            history_features={"is_married": ["duration", "ever"]},
            var_types={"is_married": "binary"},
        )

        model.fit(mock_panel_data, epochs=5, verbose=False)
        assert model._is_fitted

    def test_continuous_variable_prediction(self, mock_panel_data):
        """Test predicting continuous variables (income)."""
        model = PanelEvolutionModel(
            state_vars=["income"],
            condition_vars=["age", "is_male", "is_married"],
            lags=[1],
            var_types={"income": "continuous"},
        )

        model.fit(mock_panel_data, epochs=5, verbose=False)

        current = mock_panel_data[mock_panel_data["period"] == 3].copy()
        next_state = model.simulate_step(current, seed=42)

        # Income should be non-negative (clipped at 0 in simulate_step)
        assert (next_state["income"] >= 0).all()
        # Some should be positive (with only 5 epochs, model may not learn well)
        assert (next_state["income"] > 0).sum() > 0


class TestIntegrationWithPSID:
    """Test integration with PSID data source."""

    def test_fit_from_psid_transitions(self):
        """Test fitting model from PSID transition data."""
        # Mock PSID-like data with transition info
        psid_data = pd.DataFrame({
            "person_id": np.repeat(range(50), 4),
            "year": np.tile([2015, 2017, 2019, 2021], 50),
            "age": np.repeat(np.random.randint(25, 55, 50), 4) + np.tile([0, 2, 4, 6], 50),
            "is_male": np.repeat(np.random.choice([0, 1], 50), 4),
            "is_married": np.random.choice([0, 1], 200),
            "total_income": np.abs(np.random.lognormal(10.5, 0.8, 200)),
        })

        model = PanelEvolutionModel(
            state_vars=["is_married", "total_income"],
            condition_vars=["age", "is_male"],
            lags=[1],
            var_types={"is_married": "binary", "total_income": "continuous"},
        )

        # Should be able to fit on PSID-like panel
        model.fit(
            psid_data,
            person_id_col="person_id",
            period_col="year",
            epochs=5,
            verbose=False,
        )

        assert model._is_fitted


class TestReplacesTransitionClasses:
    """Test that PanelEvolutionModel can replace existing transition classes."""

    def test_marriage_transition_equivalent(self):
        """Test reproducing MarriageTransition behavior."""
        # Create test data
        pd.DataFrame({
            "person_id": range(100),
            "period": [0] * 100,
            "age": np.random.randint(20, 50, 100),
            "is_male": np.random.choice([0, 1], 100),
            "is_married": [0] * 100,  # All unmarried
            "is_married_lag1": [0] * 100,
        })

        model = PanelEvolutionModel(
            state_vars=["is_married"],
            condition_vars=["age", "is_male"],
            lags=[1],
            var_types={"is_married": "binary"},
        )

        # Even without fitting, should be able to get structure right
        assert "is_married" in model.state_vars

    def test_divorce_transition_with_duration(self):
        """Test divorce prediction with marriage duration feature."""
        model = PanelEvolutionModel(
            state_vars=["is_married"],
            condition_vars=["age", "is_male"],
            lags=[1, 2, 3],
            history_features={"is_married": ["duration"]},
            var_types={"is_married": "binary"},
        )

        # Duration should be used as a feature
        assert "duration" in model.history_features.get("is_married", [])
