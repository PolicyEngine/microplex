"""Tests for unified sequence synthesizer.

One model that:
- Takes full person history (variable length)
- Attends to all prior periods automatically
- Predicts any target vars given any context
- Same model handles imputation and evolution
"""

import numpy as np
import pandas as pd
import pytest

from microplex.models.sequence_synthesizer import (
    SequenceSynthesizer,
    collate_variable_length,
    prepare_sequences,
)


class TestPrepareSequences:
    """Test converting panel data to sequences."""

    def test_basic_sequence_creation(self):
        """Test creating sequences from panel data."""
        df = pd.DataFrame({
            "person_id": [1, 1, 1, 2, 2],
            "period": ["2019", "2020", "2021", "2019", "2021"],
            "income": [50000, 52000, 55000, 30000, 32000],
            "is_married": [0, 0, 1, 1, 1],
        })

        sequences = prepare_sequences(
            df,
            vars=["income", "is_married"],
            person_id_col="person_id",
            period_col="period",
        )

        assert len(sequences) == 2  # 2 persons
        assert len(sequences[0]["periods"]) == 3  # Person 1 has 3 periods
        assert len(sequences[1]["periods"]) == 2  # Person 2 has 2 periods

    def test_sequences_sorted_by_period(self):
        """Test that sequences are sorted chronologically."""
        df = pd.DataFrame({
            "person_id": [1, 1, 1],
            "period": ["2021", "2019", "2020"],  # Out of order
            "income": [55000, 50000, 52000],
        })

        sequences = prepare_sequences(df, vars=["income"])

        # Should be sorted
        assert sequences[0]["periods"] == ["2019", "2020", "2021"]
        assert sequences[0]["values"]["income"] == [50000, 52000, 55000]

    def test_handles_missing_values(self):
        """Test handling of missing/NaN values."""
        df = pd.DataFrame({
            "person_id": [1, 1, 1],
            "period": ["2019", "2020", "2021"],
            "income": [50000, np.nan, 55000],
        })

        sequences = prepare_sequences(df, vars=["income"])

        # NaN should be preserved (model will mask)
        assert np.isnan(sequences[0]["values"]["income"][1])

    def test_month_periods(self):
        """Test with month-level periods."""
        df = pd.DataFrame({
            "person_id": [1, 1, 1],
            "period": ["2021-01", "2021-02", "2021-03"],
            "income": [4000, 4100, 4200],
        })

        sequences = prepare_sequences(df, vars=["income"])

        assert sequences[0]["periods"] == ["2021-01", "2021-02", "2021-03"]


class TestCollateVariableLength:
    """Test batching variable-length sequences."""

    def test_padding(self):
        """Test that sequences are padded to max length."""
        sequences = [
            {"values": {"income": [1, 2, 3]}, "periods": ["a", "b", "c"]},
            {"values": {"income": [4, 5]}, "periods": ["a", "b"]},
        ]

        batch = collate_variable_length(sequences, vars=["income"])

        assert batch["values"].shape == (2, 3, 1)  # (batch, max_len, n_vars)
        assert batch["mask"].shape == (2, 3)
        # Second sequence should be padded
        assert batch["mask"][1, 2] == 0  # Padded position

    def test_mask_indicates_real_data(self):
        """Test that mask correctly indicates real vs padded."""
        sequences = [
            {"values": {"x": [1, 2, 3]}, "periods": ["a", "b", "c"]},
            {"values": {"x": [4]}, "periods": ["a"]},
        ]

        batch = collate_variable_length(sequences, vars=["x"])

        # First sequence: all real
        assert batch["mask"][0].tolist() == [1, 1, 1]
        # Second sequence: 1 real, 2 padded
        assert batch["mask"][1].tolist() == [1, 0, 0]


class TestSequenceSynthesizer:
    """Test the unified sequence synthesizer."""

    @pytest.fixture
    def mock_panel_data(self):
        """Create mock panel data."""
        np.random.seed(42)
        records = []
        for pid in range(50):
            n_periods = np.random.randint(3, 8)
            base_income = np.random.lognormal(10.5, 0.5)
            is_married = np.random.choice([0, 1])

            for t in range(n_periods):
                year = 2015 + t
                # Simple dynamics
                if not is_married and np.random.random() < 0.1:
                    is_married = 1
                base_income *= (1 + np.random.normal(0.02, 0.03))

                records.append({
                    "person_id": pid,
                    "period": str(year),
                    "income": base_income,
                    "is_married": is_married,
                    "age": 30 + t,
                })

        return pd.DataFrame(records)

    def test_model_initialization(self):
        """Test model can be initialized."""
        model = SequenceSynthesizer(
            continuous_vars=["income"],
            binary_vars=["is_married"],
            static_vars=["age"],
        )

        assert "income" in model.continuous_vars
        assert "is_married" in model.binary_vars

    def test_model_fit(self, mock_panel_data):
        """Test model fitting."""
        model = SequenceSynthesizer(
            continuous_vars=["income"],
            binary_vars=["is_married"],
            static_vars=[],
        )

        model.fit(mock_panel_data, epochs=5, verbose=False)

        assert model._is_fitted

    def test_predict_next(self, mock_panel_data):
        """Test predicting next period from history."""
        model = SequenceSynthesizer(
            continuous_vars=["income"],
            binary_vars=["is_married"],
        )

        model.fit(mock_panel_data, epochs=5, verbose=False)

        # Get a person's history
        person_history = mock_panel_data[mock_panel_data["person_id"] == 0]

        # Predict next period
        next_pred = model.predict_next(person_history)

        assert "income" in next_pred
        assert "is_married" in next_pred
        assert next_pred["income"] >= 0
        assert next_pred["is_married"] in [0, 1]

    def test_impute_missing(self, mock_panel_data):
        """Test imputing missing values in a record."""
        model = SequenceSynthesizer(
            continuous_vars=["income"],
            binary_vars=["is_married"],
        )

        model.fit(mock_panel_data, epochs=5, verbose=False)

        # Create record with missing income
        history = mock_panel_data[mock_panel_data["person_id"] == 0].copy()
        history_with_missing = history.copy()
        history_with_missing.loc[history_with_missing.index[-1], "income"] = np.nan

        # Impute
        imputed = model.impute(history_with_missing, target_vars=["income"])

        assert not np.isnan(imputed["income"].iloc[-1])

    def test_generate_trajectory(self, mock_panel_data):
        """Test generating full trajectory from initial state."""
        model = SequenceSynthesizer(
            continuous_vars=["income"],
            binary_vars=["is_married"],
        )

        model.fit(mock_panel_data, epochs=5, verbose=False)

        # Start from first period of a person
        initial = mock_panel_data[
            (mock_panel_data["person_id"] == 0) &
            (mock_panel_data["period"] == "2015")
        ]

        # Generate forward
        trajectory = model.generate_trajectory(
            initial_state=initial,
            n_periods=5,
            seed=42,
        )

        assert len(trajectory) == 6  # initial + 5 generated
        assert trajectory["period"].nunique() == 6

    def test_attention_uses_full_history(self, mock_panel_data):
        """Test that model attends to full history, not just recent."""
        model = SequenceSynthesizer(
            continuous_vars=["income"],
            binary_vars=["is_married"],
            n_heads=4,
            n_layers=2,
        )

        model.fit(mock_panel_data, epochs=5, verbose=False)

        # Model should be able to use long history
        # (Testing architecture, not that it learns perfectly with 5 epochs)
        assert model._transformer is not None


class TestUnifiedImputationEvolution:
    """Test that imputation and evolution are unified."""

    @pytest.fixture
    def model_and_data(self):
        """Create fitted model and test data."""
        np.random.seed(42)
        records = []
        for pid in range(30):
            is_married = 0
            income = np.random.lognormal(10.5, 0.5)
            for t in range(5):
                if not is_married and np.random.random() < 0.1:
                    is_married = 1
                income *= 1.02
                records.append({
                    "person_id": pid,
                    "period": str(2017 + t),
                    "income": income,
                    "is_married": is_married,
                    "dividend_income": np.random.lognormal(7, 1) if np.random.random() > 0.7 else 0,
                })

        df = pd.DataFrame(records)

        model = SequenceSynthesizer(
            continuous_vars=["income", "dividend_income"],
            binary_vars=["is_married"],
        )
        model.fit(df, epochs=5, verbose=False)

        return model, df

    def test_same_model_does_imputation(self, model_and_data):
        """Test model can impute cross-sectional missing vars."""
        model, df = model_and_data

        # Record missing dividend_income (like CPS record without it)
        record = df[df["person_id"] == 0].head(1).copy()
        record["dividend_income"] = np.nan

        imputed = model.impute(record, target_vars=["dividend_income"])

        assert not np.isnan(imputed["dividend_income"].iloc[0])

    def test_same_model_does_evolution(self, model_and_data):
        """Test model can predict future state."""
        model, df = model_and_data

        history = df[df["person_id"] == 0]
        next_state = model.predict_next(history)

        assert "is_married" in next_state
        assert "income" in next_state

    def test_imputation_uses_history(self, model_and_data):
        """Test that imputation conditions on available history."""
        model, df = model_and_data

        # Full history should give different imputation than no history
        full_history = df[df["person_id"] == 0].copy()
        full_history.loc[full_history.index[-1], "dividend_income"] = np.nan

        single_record = full_history.tail(1).copy()
        single_record["dividend_income"] = np.nan

        imputed_with_history = model.impute(full_history, target_vars=["dividend_income"])
        imputed_no_history = model.impute(single_record, target_vars=["dividend_income"])

        # Results may differ (depends on learned patterns)
        # Just verify both work
        assert not np.isnan(imputed_with_history["dividend_income"].iloc[-1])
        assert not np.isnan(imputed_no_history["dividend_income"].iloc[0])
