"""
End-to-end tests for hierarchical calibration pipeline.

Tests the full flow from:
1. Loading hierarchical microdata (households + persons)
2. Building constraints that aggregate person-level targets to household level
3. Running IPF calibration
4. Verifying calibrated weights match targets

Key insight: Weights are ONLY at household level. Person-level targets like
"count people aged 18-64" must be aggregated to household level first:
    sum(hh_weight * count_matching_persons_in_hh) = target_total
"""

import numpy as np
import pandas as pd
import pytest


# Test fixtures
@pytest.fixture
def mock_households():
    """Create 500 mock households with state distribution."""
    np.random.seed(42)
    n_hh = 500

    return pd.DataFrame({
        "household_id": range(n_hh),
        "state_fips": np.random.choice(
            ["06", "36", "48", "12", "17"],  # CA, NY, TX, FL, IL
            n_hh,
            p=[0.30, 0.15, 0.20, 0.20, 0.15],
        ),
        "weight": np.random.uniform(100, 500, n_hh),
        "tenure": np.random.choice([1, 2], n_hh),  # 1=own, 2=rent
        "n_persons": np.random.choice([1, 2, 3, 4, 5], n_hh, p=[0.25, 0.30, 0.25, 0.15, 0.05]),
    })


@pytest.fixture
def mock_persons(mock_households):
    """Create ~1500 mock persons linked to households."""
    np.random.seed(43)
    persons = []
    person_id = 0

    for _, hh in mock_households.iterrows():
        n_persons = hh["n_persons"]
        for i in range(n_persons):
            age = np.random.randint(0, 95)
            persons.append({
                "person_id": person_id,
                "household_id": hh["household_id"],
                "age": age,
                "is_employed": 1 if 18 <= age < 65 and np.random.random() > 0.3 else 0,
                "income": np.random.lognormal(10, 1) if 18 <= age < 65 else 0,
            })
            person_id += 1

    return pd.DataFrame(persons)


class TestHierarchicalConstraintBuilding:
    """Test building constraints that aggregate person-level to household-level."""

    def test_person_count_aggregated_to_household(self, mock_households, mock_persons):
        """Person-level count targets should be aggregated to household indicators."""

        # Target: count of people aged 18-64
        # This should produce an indicator at household level where
        # indicator[i] = count of people aged 18-64 in household i
        working_age_mask = (mock_persons["age"] >= 18) & (mock_persons["age"] < 65)
        working_age_persons = mock_persons[working_age_mask]

        # Count per household
        counts_per_hh = working_age_persons.groupby("household_id").size()
        expected_indicator = mock_households["household_id"].map(counts_per_hh).fillna(0).values

        # Verify the aggregation math:
        # sum(hh_weight * count_per_hh) should give weighted total
        weighted_total = (mock_households["weight"].values * expected_indicator).sum()

        # This should equal the unweighted count times some average weight
        assert weighted_total > 0
        assert len(expected_indicator) == len(mock_households)

    def test_household_count_is_direct_indicator(self, mock_households, mock_persons):
        """Household-level targets should be direct 0/1 indicators."""
        # Target: count of California households
        ca_mask = mock_households["state_fips"] == "06"
        indicator = ca_mask.astype(float).values

        # All values should be 0 or 1
        assert set(np.unique(indicator)).issubset({0.0, 1.0})

        # Weighted count
        weighted_ca_hh = (mock_households["weight"].values * indicator).sum()
        assert weighted_ca_hh > 0


class TestIPFCalibration:
    """Test IPF calibration on hierarchical data."""

    def test_ipf_converges_on_feasible_targets(self, mock_households, mock_persons):
        """IPF should converge when targets are feasible."""
        from microplex.calibration import Calibrator

        # Simple household-level targets
        targets = {
            "state_fips": {
                "06": 50000,  # CA
                "36": 25000,  # NY
                "48": 33000,  # TX
                "12": 33000,  # FL
                "17": 25000,  # IL
            }
        }

        calibrator = Calibrator(method="ipf", max_iter=100, tol=1e-4)
        calibrator.fit(mock_households, targets)

        assert calibrator.is_fitted_
        assert calibrator.weights_ is not None
        assert len(calibrator.weights_) == len(mock_households)

    def test_calibrated_weights_match_targets(self, mock_households, mock_persons):
        """Calibrated weights should match target totals within tolerance."""
        from microplex.calibration import Calibrator

        targets = {
            "state_fips": {
                "06": 50000,  # CA
                "36": 25000,  # NY
                "48": 33000,  # TX
                "12": 33000,  # FL
                "17": 25000,  # IL
            }
        }

        calibrator = Calibrator(method="ipf", max_iter=100, tol=1e-6)
        calibrator.fit(mock_households, targets)

        # Check weighted counts match targets
        mock_households["calibrated_weight"] = calibrator.weights_
        for state, target in targets["state_fips"].items():
            weighted_count = mock_households[
                mock_households["state_fips"] == state
            ]["calibrated_weight"].sum()
            np.testing.assert_allclose(weighted_count, target, rtol=0.01)

    def test_weights_are_positive(self, mock_households, mock_persons):
        """All calibrated weights should be positive."""
        from microplex.calibration import Calibrator

        targets = {"tenure": {1: 100000, 2: 66000}}

        calibrator = Calibrator(method="ipf")
        calibrator.fit(mock_households, targets)

        assert np.all(calibrator.weights_ > 0)


class TestHierarchicalPersonTargets:
    """Test calibrating to person-level targets using household weights."""

    def test_person_count_via_aggregation(self, mock_households, mock_persons):
        """Should calibrate to person-level count via household aggregation."""
        # Create aggregated indicator: count of working-age adults per household
        working_age_mask = (mock_persons["age"] >= 18) & (mock_persons["age"] < 65)
        working_age_per_hh = (
            mock_persons[working_age_mask]
            .groupby("household_id")
            .size()
            .reindex(mock_households["household_id"], fill_value=0)
        )

        mock_households = mock_households.copy()
        mock_households["n_working_age"] = working_age_per_hh.values

        # Now calibrate at household level
        from microplex.calibration import Calibrator

        target_working_age = 100_000_000  # 100M working-age adults

        # Use continuous target for the sum
        calibrator = Calibrator(method="ipf")
        calibrator.fit(
            mock_households,
            marginal_targets={},
            continuous_targets={"n_working_age": target_working_age}
        )

        # Verify the weighted sum matches target
        mock_households["calibrated_weight"] = calibrator.weights_
        weighted_sum = (
            mock_households["calibrated_weight"] * mock_households["n_working_age"]
        ).sum()

        np.testing.assert_allclose(weighted_sum, target_working_age, rtol=0.01)


class TestE2EPipeline:
    """End-to-end pipeline tests."""

    def test_full_pipeline_with_mixed_targets(self, mock_households, mock_persons):
        """Test full pipeline with both household and person-level targets."""
        from microplex.calibration import Calibrator

        # Step 1: Aggregate person-level features to household level
        hh_df = mock_households.copy()

        # Count children (age < 18) per household
        child_mask = mock_persons["age"] < 18
        children_per_hh = (
            mock_persons[child_mask]
            .groupby("household_id")
            .size()
            .reindex(hh_df["household_id"], fill_value=0)
        )
        hh_df["n_children"] = children_per_hh.values

        # Count working-age adults per household
        adult_mask = (mock_persons["age"] >= 18) & (mock_persons["age"] < 65)
        adults_per_hh = (
            mock_persons[adult_mask]
            .groupby("household_id")
            .size()
            .reindex(hh_df["household_id"], fill_value=0)
        )
        hh_df["n_working_age"] = adults_per_hh.values

        # Step 2: Define targets
        # Use consistent scale - household counts as base
        # We have 500 sample households, target scaled appropriately
        total_hh = 500_000  # 500k households (1000x sample)

        # Household-level targets by state
        targets_household = {
            "state_fips": {
                "06": total_hh * 0.30,  # CA: 150k
                "36": total_hh * 0.15,  # NY: 75k
                "48": total_hh * 0.20,  # TX: 100k
                "12": total_hh * 0.20,  # FL: 100k
                "17": total_hh * 0.15,  # IL: 75k
            }
        }

        # Person-level targets (aggregated) - should be consistent with HH scale
        # Average ~2.5 persons per HH, ~0.5 children, ~1.5 working age
        avg_children = hh_df["n_children"].mean()
        avg_working_age = hh_df["n_working_age"].mean()

        targets_person_aggregated = {
            "n_children": total_hh * avg_children,
            "n_working_age": total_hh * avg_working_age,
        }

        # Step 3: Calibrate
        calibrator = Calibrator(method="ipf", max_iter=200, tol=1e-6)
        calibrator.fit(
            hh_df,
            marginal_targets=targets_household,
            continuous_targets=targets_person_aggregated,
        )

        assert calibrator.is_fitted_
        hh_df["calibrated_weight"] = calibrator.weights_

        # Step 4: Verify household-level targets
        for state, target in targets_household["state_fips"].items():
            weighted_count = hh_df[hh_df["state_fips"] == state]["calibrated_weight"].sum()
            np.testing.assert_allclose(weighted_count, target, rtol=0.02)

        # Step 5: Verify person-level targets (via aggregation)
        weighted_children = (hh_df["calibrated_weight"] * hh_df["n_children"]).sum()
        weighted_adults = (hh_df["calibrated_weight"] * hh_df["n_working_age"]).sum()

        np.testing.assert_allclose(
            weighted_children, targets_person_aggregated["n_children"], rtol=0.02
        )
        np.testing.assert_allclose(
            weighted_adults, targets_person_aggregated["n_working_age"], rtol=0.02
        )

    def test_weights_propagate_to_persons(self, mock_households, mock_persons):
        """Household weights should propagate to all persons in that household."""
        from microplex.calibration import Calibrator

        hh_df = mock_households.copy()

        # Simple calibration
        calibrator = Calibrator(method="ipf")
        calibrator.fit(hh_df, {"tenure": {1: 100000, 2: 66000}})

        hh_df["calibrated_weight"] = calibrator.weights_

        # Propagate weights to persons
        persons_df = mock_persons.copy()
        persons_df = persons_df.merge(
            hh_df[["household_id", "calibrated_weight"]],
            on="household_id",
        )

        # Verify all persons in same household have same weight
        for hh_id in hh_df["household_id"].head(10):
            hh_weight = hh_df[hh_df["household_id"] == hh_id]["calibrated_weight"].iloc[0]
            person_weights = persons_df[persons_df["household_id"] == hh_id]["calibrated_weight"]
            assert (person_weights == hh_weight).all()

    def test_total_person_weight_respects_household_structure(
        self, mock_households, mock_persons
    ):
        """Total weighted person count should equal sum(hh_weight * n_persons_in_hh)."""
        from microplex.calibration import Calibrator

        hh_df = mock_households.copy()

        # Calibrate
        calibrator = Calibrator(method="ipf")
        calibrator.fit(hh_df, {"tenure": {1: 100000, 2: 66000}})
        hh_df["calibrated_weight"] = calibrator.weights_

        # Compute expected total persons
        expected_total_persons = (hh_df["calibrated_weight"] * hh_df["n_persons"]).sum()

        # Propagate and sum
        persons_df = mock_persons.copy()
        persons_df = persons_df.merge(
            hh_df[["household_id", "calibrated_weight"]],
            on="household_id",
        )
        actual_total_persons = persons_df["calibrated_weight"].sum()

        np.testing.assert_allclose(actual_total_persons, expected_total_persons, rtol=1e-10)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_infeasible_targets_handled(self, mock_households, mock_persons):
        """Should handle infeasible targets gracefully."""
        from microplex.calibration import Calibrator

        # Targets that sum to more than total households can support
        # (e.g., all states need 100% of households)
        infeasible_targets = {
            "state_fips": {
                "06": 1_000_000,  # More than total
                "36": 1_000_000,
                "48": 1_000_000,
                "12": 1_000_000,
                "17": 1_000_000,
            }
        }

        calibrator = Calibrator(method="ipf", max_iter=50)

        # Should not raise, but may not converge perfectly
        calibrator.fit(mock_households, infeasible_targets)

        # Weights should still be positive
        assert np.all(calibrator.weights_ > 0)

    def test_empty_stratum_handled(self, mock_households, mock_persons):
        """Should handle strata with no matching records."""
        from microplex.calibration import Calibrator

        # Add target for state that doesn't exist in data
        targets = {
            "state_fips": {
                "06": 50000,
                "36": 25000,
                "99": 0,  # No such state in data
            }
        }

        calibrator = Calibrator(method="ipf")

        # Should handle gracefully or skip empty stratum
        with pytest.raises(ValueError):
            calibrator.fit(mock_households, targets)
