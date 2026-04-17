"""
Integration tests for block-based synthesis pipeline.

Tests the full flow from synthesis to calibration with block-level geography.

US block geography is provided by `microplex-us`; these integration tests
skip when it is not installed so the core package stays test-clean on its
own.
"""

import importlib.util

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("microplex_us") is None,
    reason="Block synthesis integration requires US helpers from microplex-us",
)

from microplex.hierarchical import HierarchicalSynthesizer, HouseholdSchema
from microplex.geography import BlockGeography, load_block_probabilities, derive_geographies


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def block_probabilities():
    """Load block probabilities if available."""
    data_path = Path(__file__).parent.parent / "data" / "block_probabilities.parquet"
    if not data_path.exists():
        pytest.skip("Block probabilities data not available")
    return pd.read_parquet(data_path)


@pytest.fixture
def sample_cps_data():
    """Create sample CPS-like data for testing."""
    np.random.seed(42)
    n_hh = 100

    # Households
    hh = pd.DataFrame({
        "household_id": range(n_hh),
        "n_persons": np.random.choice([1, 2, 3, 4, 5], size=n_hh, p=[0.3, 0.3, 0.2, 0.15, 0.05]),
        "state_fips": np.random.choice([6, 48, 12, 36], size=n_hh),  # CA, TX, FL, NY
        "tenure": np.random.choice([1, 2], size=n_hh),
        "hh_weight": np.random.uniform(100, 1000, size=n_hh),
    })
    hh["n_adults"] = np.clip(hh["n_persons"] - np.random.randint(0, 2, size=n_hh), 1, None)
    hh["n_children"] = hh["n_persons"] - hh["n_adults"]

    # Persons
    persons = []
    for _, hh_row in hh.iterrows():
        for person_idx in range(int(hh_row["n_persons"])):
            is_adult = person_idx < hh_row["n_adults"]
            persons.append({
                "household_id": hh_row["household_id"],
                "person_id": len(persons),
                "age": np.random.randint(25, 65) if is_adult else np.random.randint(0, 18),
                "sex": np.random.choice([1, 2]),
                "income": np.random.uniform(0, 100000) if is_adult else 0,
                "employment_status": np.random.choice([1, 2, 3]) if is_adult else 0,
                "education": np.random.choice([1, 2, 3, 4]) if is_adult else 0,
                "relationship_to_head": 0 if person_idx == 0 else np.random.choice([1, 2, 3]),
            })

    persons = pd.DataFrame(persons)

    return hh, persons


# =============================================================================
# Block Assignment Tests
# =============================================================================

class TestBlockAssignmentIntegration:
    """Test block assignment integration with synthesizer."""

    def test_synthesizer_accepts_block_probabilities(self, block_probabilities):
        """Synthesizer initializes with block_probabilities parameter."""
        schema = HouseholdSchema(
            hh_vars=["n_persons", "n_adults", "n_children", "state_fips", "tenure"],
            person_vars=["age", "sex", "income"],
        )
        synth = HierarchicalSynthesizer(
            schema=schema,
            block_probabilities=block_probabilities,
            random_state=42,
        )
        assert synth._block_lookup is not None

    def test_generate_includes_block_geoid(self, sample_cps_data, block_probabilities):
        """Generated data includes block_geoid column."""
        hh, persons = sample_cps_data

        schema = HouseholdSchema(
            hh_vars=["n_persons", "n_adults", "n_children", "state_fips", "tenure"],
            person_vars=["age", "sex", "income", "employment_status", "education", "relationship_to_head"],
        )

        synth = HierarchicalSynthesizer(
            schema=schema,
            block_probabilities=block_probabilities,
            random_state=42,
        )

        synth.fit(hh, persons, hh_weight_col="hh_weight", epochs=5, verbose=False)
        syn_hh, syn_persons = synth.generate(n_households=50, verbose=False)

        assert "block_geoid" in syn_hh.columns
        assert all(syn_hh["block_geoid"].str.len() == 15)

    def test_derive_geographies_post_hoc(self, sample_cps_data, block_probabilities):
        """Parent geographies can be derived from block_geoid post-hoc."""
        hh, persons = sample_cps_data

        schema = HouseholdSchema(
            hh_vars=["n_persons", "n_adults", "n_children", "state_fips", "tenure"],
            person_vars=["age", "sex", "income", "employment_status", "education", "relationship_to_head"],
        )

        synth = HierarchicalSynthesizer(
            schema=schema,
            block_probabilities=block_probabilities,
            random_state=42,
        )

        synth.fit(hh, persons, hh_weight_col="hh_weight", epochs=5, verbose=False)
        syn_hh, syn_persons = synth.generate(n_households=50, verbose=False)

        # Synthesizer only sets block_geoid
        assert "block_geoid" in syn_hh.columns
        assert "tract_geoid" not in syn_hh.columns  # Not set during synthesis

        # Derive parent geographies post-hoc
        geos = derive_geographies(
            syn_hh["block_geoid"],
            include_cd=True,
            include_sld=True,
            block_data=block_probabilities
        )

        # Check derived columns exist
        assert "tract_geoid" in geos.columns
        assert "county_fips" in geos.columns
        assert "cd_id" in geos.columns
        assert "sldu_id" in geos.columns
        assert "sldl_id" in geos.columns

        # Check lengths
        assert all(geos["tract_geoid"].str.len() == 11)
        assert all(geos["county_fips"].str.len() == 5)

    def test_block_assignment_respects_state(self, sample_cps_data, block_probabilities):
        """Assigned blocks are in valid states and consistent with state_fips."""
        hh, persons = sample_cps_data

        schema = HouseholdSchema(
            hh_vars=["n_persons", "n_adults", "n_children", "state_fips", "tenure"],
            person_vars=["age", "sex", "income", "employment_status", "education", "relationship_to_head"],
        )

        synth = HierarchicalSynthesizer(
            schema=schema,
            block_probabilities=block_probabilities,
            random_state=42,
        )

        synth.fit(hh, persons, hh_weight_col="hh_weight", epochs=5, verbose=False)
        syn_hh, syn_persons = synth.generate(n_households=100, verbose=False)

        # State from block_geoid should match state_fips column
        block_states = syn_hh["block_geoid"].str[:2].astype(int)
        state_fips_col = syn_hh["state_fips"].astype(int)

        # Block state and state_fips should be consistent
        assert all(block_states == state_fips_col)

        # States should be valid (in our block data)
        valid_states = block_probabilities["state_fips"].astype(int).unique()
        assert all(block_states.isin(valid_states))

    def test_cd_derived_from_block(self, sample_cps_data, block_probabilities):
        """CD is correctly derived from block lookup post-hoc."""
        hh, persons = sample_cps_data

        schema = HouseholdSchema(
            hh_vars=["n_persons", "n_adults", "n_children", "state_fips", "tenure"],
            person_vars=["age", "sex", "income", "employment_status", "education", "relationship_to_head"],
        )

        synth = HierarchicalSynthesizer(
            schema=schema,
            block_probabilities=block_probabilities,
            random_state=42,
        )

        synth.fit(hh, persons, hh_weight_col="hh_weight", epochs=5, verbose=False)
        syn_hh, _ = synth.generate(n_households=50, verbose=False)

        # Derive CD post-hoc
        geos = derive_geographies(
            syn_hh["block_geoid"],
            include_cd=True,
            block_data=block_probabilities
        )

        # CD should be in proper format (e.g., CA-12, TX-AL)
        cd_pattern = r"^[A-Z]{2}-(\d{2}|AL)$"
        valid_cds = geos["cd_id"].str.match(cd_pattern, na=False)
        assert valid_cds.sum() > 0


# =============================================================================
# Geography Module Integration Tests
# =============================================================================

class TestBlockGeographyIntegration:
    """Test BlockGeography with real data."""

    def test_sample_blocks_distribution(self, block_probabilities):
        """Sampled blocks follow population distribution."""
        geo = BlockGeography(lazy_load=True)
        geo._data = block_probabilities

        # Sample California blocks
        ca_blocks = geo.sample_blocks("06", n=1000, random_state=42)

        # Should cover multiple counties
        counties = set(BlockGeography.get_county(b) for b in ca_blocks)
        assert len(counties) > 10  # LA County has many blocks, so should have many counties

    def test_get_all_geographies_consistent(self, block_probabilities):
        """All geography derivations are consistent."""
        geo = BlockGeography(lazy_load=True)
        geo._data = block_probabilities

        sample_block = block_probabilities["geoid"].iloc[0]
        geos = geo.get_all_geographies(sample_block)

        # State from county should match state from block
        assert geos["county_fips"][:2] == geos["state_fips"]

        # Tract should start with county
        assert geos["tract_geoid"].startswith(geos["county_fips"])


# =============================================================================
# Calibration Integration Tests
# =============================================================================

class TestBlockCalibrationIntegration:
    """Test calibration with block-level geography."""

    def test_cd_indicators_can_be_built(self, sample_cps_data, block_probabilities):
        """CD indicator columns can be built for calibration (post-hoc derivation)."""
        hh, persons = sample_cps_data

        schema = HouseholdSchema(
            hh_vars=["n_persons", "n_adults", "n_children", "state_fips", "tenure"],
            person_vars=["age", "sex", "income", "employment_status", "education", "relationship_to_head"],
        )

        synth = HierarchicalSynthesizer(
            schema=schema,
            block_probabilities=block_probabilities,
            random_state=42,
        )

        synth.fit(hh, persons, hh_weight_col="hh_weight", epochs=5, verbose=False)
        syn_hh, _ = synth.generate(n_households=100, verbose=False)

        # Derive CD post-hoc (this is part of the pipeline before calibration)
        geos = derive_geographies(
            syn_hh["block_geoid"],
            include_cd=True,
            block_data=block_probabilities
        )
        syn_hh["cd_id"] = geos["cd_id"].values

        # Build CD indicators
        for cd_id in syn_hh["cd_id"].dropna().unique():
            col_name = f"n_persons_{cd_id}"
            syn_hh[col_name] = np.where(
                syn_hh["cd_id"] == cd_id,
                syn_hh["n_persons"],
                0
            )

        # Check indicators sum to total persons
        indicator_cols = [c for c in syn_hh.columns if c.startswith("n_persons_") and "-" in c]
        indicator_sum = syn_hh[indicator_cols].sum().sum()
        total_persons = syn_hh["n_persons"].sum()

        assert indicator_sum == total_persons
