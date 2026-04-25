"""
TDD tests for loading calibration targets from Supabase.

These tests verify that:
1. Targets can be loaded from Supabase with proper filtering
2. Target variables are mapped correctly to CPS columns
3. Calibration constraints can be built from targets
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import responses

# Direct import to avoid torch dependency in __init__.py
src_path = Path(__file__).parent.parent / "src" / "microplex"
sys.path.insert(0, str(src_path.parent))

# Import directly to avoid package __init__.py
spec = importlib.util.spec_from_file_location("supabase_targets", src_path / "supabase_targets.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
SupabaseTargetLoader = module.SupabaseTargetLoader


SUPABASE_URL = "https://test.supabase.co"
SUPABASE_KEY = "test-key"


class TestSupabaseTargetLoader:
    """Tests for loading targets from Supabase."""

    @pytest.fixture
    def loader(self):
        return SupabaseTargetLoader(SUPABASE_URL, SUPABASE_KEY)

    def test_missing_service_key_raises(self, monkeypatch):
        """Should never fall back to an embedded service-role key."""
        monkeypatch.delenv("COSILICO_SUPABASE_SERVICE_KEY", raising=False)

        with pytest.raises(ValueError, match="COSILICO_SUPABASE_SERVICE_KEY"):
            SupabaseTargetLoader(SUPABASE_URL)

    @responses.activate
    def test_load_all_targets(self, loader):
        """Should load all targets with source and stratum info."""
        # Mock the targets query with joined data
        responses.add(
            responses.GET,
            f"{SUPABASE_URL}/rest/v1/targets",
            json=[
                {
                    "id": "t1",
                    "variable": "employment_income",
                    "value": 9022400000000,
                    "target_type": "amount",
                    "period": 2024,
                    "source": {"name": "IRS SOI", "institution": "IRS"},
                    "stratum": {"name": "National", "jurisdiction": "us"},
                },
                {
                    "id": "t2",
                    "variable": "snap_spending",
                    "value": 103100000000,
                    "target_type": "amount",
                    "period": 2024,
                    "source": {"name": "USDA SNAP", "institution": "USDA"},
                    "stratum": {"name": "National", "jurisdiction": "us"},
                },
            ],
            status=200,
        )

        targets = loader.load_all()

        assert len(targets) == 2
        assert targets[0]["variable"] == "employment_income"
        assert targets[0]["value"] == 9022400000000
        assert targets[1]["variable"] == "snap_spending"

    @responses.activate
    def test_load_by_institution(self, loader):
        """Should filter targets by source institution."""
        # Mock the sources query first
        responses.add(
            responses.GET,
            f"{SUPABASE_URL}/rest/v1/sources",
            json=[{"id": "src-1", "institution": "IRS", "name": "IRS SOI"}],
            status=200,
        )
        # Then mock the targets query
        responses.add(
            responses.GET,
            f"{SUPABASE_URL}/rest/v1/targets",
            json=[
                {
                    "id": "t1",
                    "variable": "employment_income",
                    "value": 9022400000000,
                    "target_type": "amount",
                    "period": 2024,
                    "source": {"name": "IRS SOI", "institution": "IRS"},
                    "stratum": {"name": "National", "jurisdiction": "us"},
                },
            ],
            status=200,
        )

        targets = loader.load_by_institution("IRS")

        assert len(targets) == 1
        assert targets[0]["source"]["institution"] == "IRS"

    @responses.activate
    def test_load_by_period(self, loader):
        """Should filter targets by period/year."""
        responses.add(
            responses.GET,
            f"{SUPABASE_URL}/rest/v1/targets",
            json=[
                {
                    "id": "t1",
                    "variable": "employment_income",
                    "value": 9022400000000,
                    "target_type": "amount",
                    "period": 2024,
                    "source": {"name": "IRS SOI", "institution": "IRS"},
                    "stratum": {"name": "National", "jurisdiction": "us"},
                },
            ],
            status=200,
        )

        targets = loader.load_by_period(2024)

        assert len(targets) == 1
        assert targets[0]["period"] == 2024


class TestTargetToCPSMapping:
    """Tests for mapping Supabase targets to CPS columns."""

    @pytest.fixture
    def loader(self):
        return SupabaseTargetLoader(SUPABASE_URL, SUPABASE_KEY)

    def test_income_variable_mapping(self, loader):
        """Should map PE income variables to CPS columns."""
        mapping = loader.get_cps_column_map()

        # IRS income targets should map to CPS columns
        assert mapping["employment_income"] == "employment_income"
        assert mapping["self_employment_income"] == "self_employment_income"
        assert mapping["dividend_income"] == "dividend_income"
        assert mapping["interest_income"] == "interest_income"
        assert mapping["social_security"] == "social_security"
        assert mapping["unemployment_compensation"] == "unemployment_compensation"

    def test_benefit_variable_mapping(self, loader):
        """Should map benefit targets to CPS columns."""
        mapping = loader.get_cps_column_map()

        # Benefit spending targets
        assert mapping["snap_spending"] == "snap"
        assert mapping["ssi_spending"] == "ssi"
        assert mapping["eitc_spending"] == "eitc"


class TestBuildCalibrationConstraints:
    """Tests for building calibration constraints from targets."""

    @pytest.fixture
    def loader(self):
        return SupabaseTargetLoader(SUPABASE_URL, SUPABASE_KEY)

    @responses.activate
    def test_build_continuous_targets(self, loader):
        """Should build continuous calibration targets dict."""
        responses.add(
            responses.GET,
            f"{SUPABASE_URL}/rest/v1/targets",
            json=[
                {
                    "id": "t1",
                    "variable": "employment_income",
                    "value": 9022400000000,
                    "target_type": "amount",
                    "period": 2024,
                    "source": {"name": "IRS SOI", "institution": "IRS"},
                    "stratum": {"name": "National", "jurisdiction": "us"},
                },
                {
                    "id": "t2",
                    "variable": "snap_spending",
                    "value": 103100000000,
                    "target_type": "amount",
                    "period": 2024,
                    "source": {"name": "USDA SNAP", "institution": "USDA"},
                    "stratum": {"name": "National", "jurisdiction": "us"},
                },
            ],
            status=200,
        )

        constraints = loader.build_calibration_constraints()

        # Should return dict with CPS column names as keys
        assert "employment_income" in constraints
        assert constraints["employment_income"] == 9022400000000
        assert "snap" in constraints
        assert constraints["snap"] == 103100000000

    @responses.activate
    def test_build_state_targets(self, loader):
        """Should build state-level calibration targets."""
        responses.add(
            responses.GET,
            f"{SUPABASE_URL}/rest/v1/targets",
            json=[
                {
                    "id": "t1",
                    "variable": "medicaid_enrollment",
                    "value": 14000000,
                    "target_type": "count",
                    "period": 2024,
                    "source": {"name": "CMS Medicaid", "institution": "HHS"},
                    "stratum": {"name": "California", "jurisdiction": "us-ca"},
                },
            ],
            status=200,
        )

        constraints = loader.build_calibration_constraints(include_states=True)

        # State targets should include state code
        assert "medicaid_ca" in constraints or "medicaid_enrollment_ca" in constraints


class TestIntegrationWithCalibrator:
    """Integration tests with the Calibrator."""

    @pytest.fixture
    def loader(self):
        return SupabaseTargetLoader(SUPABASE_URL, SUPABASE_KEY)

    @pytest.mark.skip(reason="Integration test requires real Supabase connection")
    def test_calibration_with_supabase_targets(self, loader):
        """End-to-end test: load targets from Supabase and run calibration."""
        import numpy as np
        import pandas as pd
        try:
            # Direct import to avoid torch dependency
            import importlib.util
            cal_spec = importlib.util.spec_from_file_location(
                "calibration",
                Path(__file__).parent.parent / "src" / "microplex" / "calibration.py"
            )
            cal_module = importlib.util.module_from_spec(cal_spec)
            cal_spec.loader.exec_module(cal_module)
            calibrator_cls = cal_module.Calibrator
        except Exception as e:
            pytest.skip(f"Cannot import Calibrator: {e}")

        # Create mock CPS data
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            "weight": np.ones(n) * 100,
            "employment_income": np.random.lognormal(10, 1, n),
            "snap": np.random.choice([0, 500], n, p=[0.9, 0.1]),
        })

        # Load targets from Supabase (uses live connection)
        targets = loader.build_calibration_constraints()

        if not targets:
            pytest.skip("No targets loaded from Supabase")

        # Filter to available columns
        available = {k: v for k, v in targets.items() if k in df.columns}

        if not available:
            pytest.skip("No matching targets for test data")

        # Run calibration
        calibrator = calibrator_cls(method="ipf", max_iter=100)
        calibrator.fit(df, marginal_targets={}, continuous_targets=available, weight_col="weight")

        assert calibrator.weights_ is not None
        assert len(calibrator.weights_) == n
