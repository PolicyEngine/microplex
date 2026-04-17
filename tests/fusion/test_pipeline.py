"""Tests for fusion pipeline."""

import numpy as np
import pandas as pd
import pytest

from microplex.fusion.harmonize import COMMON_SCHEMA
from microplex.fusion.pipeline import (
    FusionConfig,
    FusionResult,
    FusionSynthesizer,
)


class TestFusionConfig:
    """Test FusionConfig dataclass."""

    def test_default_values(self):
        config = FusionConfig()
        assert config.n_layers == 6
        assert config.hidden_dim == 128
        assert config.epochs == 100
        assert config.device == "cpu"

    def test_custom_values(self):
        config = FusionConfig(n_layers=4, epochs=50, device="mps")
        assert config.n_layers == 4
        assert config.epochs == 50
        assert config.device == "mps"


class TestFusionSynthesizer:
    """Test FusionSynthesizer class."""

    @pytest.fixture
    def sample_cps(self):
        return pd.DataFrame({
            "age": [35, 42, 28, 55, 62],
            "sex": [1, 2, 1, 2, 1],
            "employment_income": [50000, 75000, 40000, 80000, 0],
            "self_employment_income": [0, 10000, 0, 0, 5000],
            "social_security": [0, 0, 0, 0, 24000],
            "state_fips": [6, 36, 48, 12, 6],
            "weight": [1000, 1500, 800, 1200, 900],
        })

    @pytest.fixture
    def sample_puf(self):
        return pd.DataFrame({
            "age": [45, 55, 38],
            "filing_status": ["JOINT", "SINGLE", "JOINT"],
            "employment_income": [150000, 80000, 200000],
            "self_employment_income": [0, 25000, 0],
            "long_term_capital_gains": [50000, 0, 100000],
            "partnership_s_corp_income": [0, 25000, 75000],
            "charitable_cash": [5000, 0, 10000],
            "weight": [500, 300, 400],
        })

    def test_init_with_default_config(self):
        synth = FusionSynthesizer()
        assert synth.config is not None
        assert synth.surveys == {}

    def test_add_survey(self, sample_cps):
        synth = FusionSynthesizer()
        synth.add_survey("cps", sample_cps)
        assert "cps" in synth.surveys
        assert len(synth.surveys["cps"]) == len(sample_cps)

    def test_add_survey_chaining(self, sample_cps, sample_puf):
        synth = FusionSynthesizer()
        result = synth.add_survey("cps", sample_cps).add_survey("puf", sample_puf)
        assert result is synth
        assert len(synth.surveys) == 2

    def test_harmonize(self, sample_cps, sample_puf):
        synth = FusionSynthesizer()
        synth.add_survey("cps", sample_cps)
        synth.add_survey("puf", sample_puf)

        stacked, mask = synth.harmonize()

        assert len(stacked) == len(sample_cps) + len(sample_puf)
        assert mask.shape[0] == len(stacked)
        assert mask.shape[1] == len(COMMON_SCHEMA)

    def test_fit(self, sample_cps, sample_puf):
        config = FusionConfig(n_layers=2, hidden_dim=16, epochs=5)
        synth = FusionSynthesizer(config=config)
        synth.add_survey("cps", sample_cps)
        synth.add_survey("puf", sample_puf)

        model = synth.fit(verbose=False)

        assert model is not None
        assert synth.model is not None

    def test_generate(self, sample_cps, sample_puf):
        config = FusionConfig(n_layers=2, hidden_dim=16, epochs=5)
        synth = FusionSynthesizer(config=config)
        synth.add_survey("cps", sample_cps)
        synth.add_survey("puf", sample_puf)
        synth.fit(verbose=False)

        synthetic = synth.generate(n_samples=10)

        assert len(synthetic) == 10
        # Should have schema variables
        for var in COMMON_SCHEMA.keys():
            assert var in synthetic.columns

    def test_generate_values_in_range(self, sample_cps, sample_puf):
        config = FusionConfig(n_layers=2, hidden_dim=16, epochs=10)
        synth = FusionSynthesizer(config=config)
        synth.add_survey("cps", sample_cps)
        synth.add_survey("puf", sample_puf)
        synth.fit(verbose=False)

        synthetic = synth.generate(n_samples=100)

        # Age should be reasonable
        assert synthetic["age"].min() >= 0
        assert synthetic["age"].max() <= 120

        # Binary variables should be 0 or 1
        assert synthetic["is_male"].isin([0, 1]).all()
        assert synthetic["is_married"].isin([0, 1]).all()

    def test_fit_generate(self, sample_cps, sample_puf):
        config = FusionConfig(n_layers=2, hidden_dim=16, epochs=5)
        synth = FusionSynthesizer(config=config)
        synth.add_survey("cps", sample_cps)
        synth.add_survey("puf", sample_puf)

        result = synth.fit_generate(n_samples=10, verbose=False)

        assert isinstance(result, FusionResult)
        assert len(result.synthetic) == 10
        assert result.model is not None
        assert result.training_time > 0


class TestFusionResult:
    """Test FusionResult dataclass."""

    @pytest.fixture
    def sample_result(self, tmp_path):
        from microplex.fusion.masked_maf import MaskedMAF

        synthetic = pd.DataFrame({
            "age": [35, 42, 28],
            "employment_income": [50000, 75000, 40000],
            "weight": [1.0, 1.0, 1.0],
        })
        model = MaskedMAF(n_features=2, n_layers=2, hidden_dim=16)
        # Initialize model attributes
        model.feature_means_ = np.zeros(2)
        model.feature_stds_ = np.ones(2)
        model.dim_weights_ = np.ones(2)

        return FusionResult(
            synthetic=synthetic,
            model=model,
            config=FusionConfig(),
            training_time=10.5,
            variable_names=["age", "employment_income"],
            observation_rates={"age": 1.0, "employment_income": 0.8},
        )

    def test_save(self, sample_result, tmp_path):
        save_path = tmp_path / "fusion_result"
        sample_result.save(save_path)

        assert (save_path / "synthetic.parquet").exists()
        assert (save_path / "model.pkl").exists()
        assert (save_path / "metadata.json").exists()


class TestFusionSynthesizerErrors:
    """Test error handling."""

    def test_fit_without_surveys_raises(self):
        synth = FusionSynthesizer()
        with pytest.raises(ValueError, match="No surveys added"):
            synth.harmonize()

    def test_generate_without_fit_raises(self):
        synth = FusionSynthesizer()
        synth.add_survey("cps", pd.DataFrame({"age": [35]}))
        with pytest.raises(ValueError, match="Model not fitted"):
            synth.generate(10)
