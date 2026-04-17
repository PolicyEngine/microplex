"""Tests for survey harmonization."""

import numpy as np
import pandas as pd
import pytest

from microplex.fusion.harmonize import (
    COMMON_SCHEMA,
    apply_inverse_transform,
    apply_transform,
    harmonize_cps,
    harmonize_puf,
    harmonize_surveys,
    inverse_signed_log,
    signed_log,
    stack_surveys,
)


class TestTransforms:
    """Test variable transformations."""

    def test_signed_log_positive(self):
        x = np.array([0, 1, 10, 100, 1000])
        y = signed_log(x)
        assert np.allclose(y, np.log1p(x))

    def test_signed_log_negative(self):
        x = np.array([-1, -10, -100])
        y = signed_log(x)
        expected = -np.log1p(np.abs(x))
        assert np.allclose(y, expected)

    def test_signed_log_invertible(self):
        x = np.array([-1000, -10, 0, 10, 1000])
        y = signed_log(x)
        x_recovered = inverse_signed_log(y)
        assert np.allclose(x, x_recovered)

    def test_apply_transform_none(self):
        x = np.array([1, 2, 3])
        assert np.allclose(apply_transform(x, "none"), x)

    def test_apply_transform_log1p(self):
        x = np.array([0, 1, 10, 100])
        assert np.allclose(apply_transform(x, "log1p"), np.log1p(x))

    def test_apply_inverse_transform_log1p(self):
        x = np.array([0, 1, 10, 100])
        y = apply_transform(x, "log1p")
        x_recovered = apply_inverse_transform(y, "log1p")
        assert np.allclose(x, x_recovered)


class TestCommonSchema:
    """Test common variable schema."""

    def test_schema_has_demographics(self):
        assert "age" in COMMON_SCHEMA
        assert "is_male" in COMMON_SCHEMA
        assert "is_married" in COMMON_SCHEMA

    def test_schema_has_income_vars(self):
        assert "employment_income" in COMMON_SCHEMA
        assert "self_employment_income" in COMMON_SCHEMA
        assert "long_term_capital_gains" in COMMON_SCHEMA

    def test_schema_specifies_transforms(self):
        for var, spec in COMMON_SCHEMA.items():
            assert "type" in spec
            assert spec["type"] in ["continuous", "binary", "discrete"]
            if spec["type"] == "continuous":
                assert "transform" in spec


class TestHarmonizeCPS:
    """Test CPS harmonization."""

    @pytest.fixture
    def sample_cps(self):
        return pd.DataFrame({
            "age": [35, 42, 28],
            "sex": [1, 2, 1],
            "employment_income": [50000, 75000, 40000],
            "state_fips": [6, 36, 48],
            "weight": [1000, 1500, 800],
        })

    def test_harmonize_preserves_rows(self, sample_cps):
        result = harmonize_cps(sample_cps)
        assert len(result) == len(sample_cps)

    def test_harmonize_adds_survey_marker(self, sample_cps):
        result = harmonize_cps(sample_cps)
        assert "_survey" in result.columns
        assert (result["_survey"] == "cps").all()

    def test_harmonize_preserves_age(self, sample_cps):
        result = harmonize_cps(sample_cps)
        assert "age" in result.columns
        assert np.allclose(result["age"], sample_cps["age"])

    def test_harmonize_maps_sex_to_is_male(self, sample_cps):
        result = harmonize_cps(sample_cps)
        assert "is_male" in result.columns
        # sex=1 -> is_male=1, sex=2 -> is_male=0
        expected = [1.0, 0.0, 1.0]
        assert np.allclose(result["is_male"], expected)

    def test_harmonize_puf_vars_are_nan(self, sample_cps):
        result = harmonize_cps(sample_cps)
        # Capital gains only in PUF
        assert "long_term_capital_gains" in result.columns
        assert result["long_term_capital_gains"].isna().all()


class TestHarmonizePUF:
    """Test PUF harmonization."""

    @pytest.fixture
    def sample_puf(self):
        return pd.DataFrame({
            "age": [45, 55],
            "filing_status": ["JOINT", "SINGLE"],
            "employment_income": [150000, 80000],
            "long_term_capital_gains": [50000, 0],
            "partnership_s_corp_income": [0, 25000],
            "weight": [500, 300],
        })

    def test_harmonize_preserves_rows(self, sample_puf):
        result = harmonize_puf(sample_puf)
        assert len(result) == len(sample_puf)

    def test_harmonize_adds_survey_marker(self, sample_puf):
        result = harmonize_puf(sample_puf)
        assert (result["_survey"] == "puf").all()

    def test_harmonize_maps_filing_status(self, sample_puf):
        result = harmonize_puf(sample_puf)
        assert "is_joint_filer" in result.columns
        expected = [1.0, 0.0]  # JOINT -> 1, SINGLE -> 0
        assert np.allclose(result["is_joint_filer"], expected)

    def test_harmonize_cps_vars_are_nan(self, sample_puf):
        result = harmonize_puf(sample_puf)
        # State FIPS only in CPS
        assert "state_fips" in result.columns
        assert result["state_fips"].isna().all()


class TestStackSurveys:
    """Test survey stacking."""

    @pytest.fixture
    def harmonized_surveys(self):
        # Use raw surveys and harmonize them properly
        cps = pd.DataFrame({
            "age": [35, 42],
            "sex": [1, 2],
            "employment_income": [50000, 75000],
            "state_fips": [6, 36],
            "weight": [1000, 1500],
        })
        puf = pd.DataFrame({
            "age": [45],
            "filing_status": ["JOINT"],
            "employment_income": [150000],
            "long_term_capital_gains": [50000],
            "weight": [500],
        })
        return harmonize_surveys({"cps": cps, "puf": puf})

    def test_stack_concatenates(self, harmonized_surveys):
        stacked, mask = stack_surveys(harmonized_surveys)
        assert len(stacked) == 3  # 2 CPS + 1 PUF

    def test_stack_creates_mask(self, harmonized_surveys):
        stacked, mask = stack_surveys(harmonized_surveys)
        # Mask should have shape (n_records, n_schema_vars)
        assert mask.shape[0] == len(stacked)
        assert mask.shape[1] == len(COMMON_SCHEMA)

    def test_mask_true_for_observed(self, harmonized_surveys):
        stacked, mask = stack_surveys(harmonized_surveys)
        schema_vars = list(COMMON_SCHEMA.keys())
        age_idx = schema_vars.index("age")
        # Age is in both surveys - all observed
        assert mask[:, age_idx].all()

    def test_mask_false_for_missing(self, harmonized_surveys):
        stacked, mask = stack_surveys(harmonized_surveys)
        schema_vars = list(COMMON_SCHEMA.keys())
        ltcg_idx = schema_vars.index("long_term_capital_gains")
        # First 2 rows are CPS (NaN), last is PUF (observed)
        assert not mask[0, ltcg_idx]  # CPS - missing
        assert not mask[1, ltcg_idx]  # CPS - missing
        assert mask[2, ltcg_idx]      # PUF - observed


class TestHarmonizeSurveys:
    """Test full harmonization pipeline."""

    @pytest.fixture
    def raw_surveys(self):
        cps = pd.DataFrame({
            "age": [35, 42, 28],
            "sex": [1, 2, 1],
            "employment_income": [50000, 75000, 40000],
            "state_fips": [6, 36, 48],
            "weight": [1000, 1500, 800],
        })
        puf = pd.DataFrame({
            "age": [45, 55],
            "filing_status": ["JOINT", "SINGLE"],
            "employment_income": [150000, 80000],
            "long_term_capital_gains": [50000, 0],
            "weight": [500, 300],
        })
        return {"cps": cps, "puf": puf}

    def test_harmonize_both_surveys(self, raw_surveys):
        result = harmonize_surveys(raw_surveys)
        assert "cps" in result
        assert "puf" in result

    def test_harmonized_have_same_columns(self, raw_surveys):
        result = harmonize_surveys(raw_surveys)
        cps_cols = set(result["cps"].columns)
        puf_cols = set(result["puf"].columns)
        # Should have same schema columns plus metadata
        schema_cols = set(COMMON_SCHEMA.keys())
        assert schema_cols.issubset(cps_cols)
        assert schema_cols.issubset(puf_cols)
