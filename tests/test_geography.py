"""
Tests for the geography module.

Tests follow TDD principles - written to define expected behavior.

These tests exercise the US-specific block-geography helpers that moved to
`microplex-us`. When `microplex-us` is not installed, they skip rather than
fail — the bare `microplex` install should remain valid on its own.
"""

import importlib.util

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("microplex_us") is None,
    reason="US block geography lives in microplex-us; install it to run these tests",
)

from microplex.geography import (
    BlockGeography,
    load_block_probabilities,
    derive_geographies,
    STATE_LEN,
    COUNTY_LEN,
    TRACT_LEN,
    BLOCK_LEN,
    STATE_GEOID_LEN,
    COUNTY_GEOID_LEN,
    TRACT_GEOID_LEN,
    BLOCK_GEOID_LEN,
)


# =============================================================================
# Constants Tests
# =============================================================================

class TestGEOIDConstants:
    """Test GEOID structure constants are correct."""

    def test_state_len(self):
        assert STATE_LEN == 2

    def test_county_len(self):
        assert COUNTY_LEN == 3

    def test_tract_len(self):
        assert TRACT_LEN == 6

    def test_block_len(self):
        assert BLOCK_LEN == 4

    def test_cumulative_lengths(self):
        assert STATE_GEOID_LEN == 2
        assert COUNTY_GEOID_LEN == 5
        assert TRACT_GEOID_LEN == 11
        assert BLOCK_GEOID_LEN == 15


# =============================================================================
# Static Geography Extraction Tests (no data needed)
# =============================================================================

class TestStaticGeographyExtraction:
    """Test GEOID string extraction methods (no data loading needed)."""

    # Example block GEOID: 060372073021001
    # State: 06 (California)
    # County: 037 (Los Angeles)
    # Tract: 207302
    # Block: 1001
    SAMPLE_BLOCK = "060372073021001"

    def test_get_state_from_block_geoid(self):
        """Extract state FIPS from block GEOID."""
        result = BlockGeography.get_state(self.SAMPLE_BLOCK)
        assert result == "06"

    def test_get_county_from_block_geoid(self):
        """Extract county FIPS (state+county) from block GEOID."""
        result = BlockGeography.get_county(self.SAMPLE_BLOCK)
        assert result == "06037"

    def test_get_tract_from_block_geoid(self):
        """Extract tract GEOID from block GEOID."""
        result = BlockGeography.get_tract(self.SAMPLE_BLOCK)
        assert result == "06037207302"

    def test_geoid_length_validation(self):
        """Verify extracted lengths match constants."""
        assert len(BlockGeography.get_state(self.SAMPLE_BLOCK)) == STATE_GEOID_LEN
        assert len(BlockGeography.get_county(self.SAMPLE_BLOCK)) == COUNTY_GEOID_LEN
        assert len(BlockGeography.get_tract(self.SAMPLE_BLOCK)) == TRACT_GEOID_LEN

    def test_multiple_blocks_different_states(self):
        """Test extraction from blocks in different states."""
        blocks = {
            "010010201001000": ("01", "01001", "01001020100"),  # Alabama
            "060372073021001": ("06", "06037", "06037207302"),  # California
            "481131234001234": ("48", "48113", "48113123400"),  # Texas
        }
        for block, (state, county, tract) in blocks.items():
            assert BlockGeography.get_state(block) == state
            assert BlockGeography.get_county(block) == county
            assert BlockGeography.get_tract(block) == tract


# =============================================================================
# Data Loading Tests
# =============================================================================

class TestLoadBlockProbabilities:
    """Test block probabilities data loading."""

    @pytest.fixture
    def data_path(self):
        """Path to block probabilities data."""
        return Path(__file__).parent.parent / "data" / "block_probabilities.parquet"

    def test_load_default_path(self, data_path):
        """Load from default location."""
        if not data_path.exists():
            pytest.skip("Block probabilities data not available")
        df = load_block_probabilities()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_explicit_path(self, data_path):
        """Load from explicit path."""
        if not data_path.exists():
            pytest.skip("Block probabilities data not available")
        df = load_block_probabilities(data_path)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns_present(self, data_path):
        """Verify required columns exist."""
        if not data_path.exists():
            pytest.skip("Block probabilities data not available")
        df = load_block_probabilities(data_path)
        required_cols = ["geoid", "state_fips", "population", "prob"]
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"

    def test_file_not_found_raises(self):
        """Raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_block_probabilities("/nonexistent/path/file.parquet")


# =============================================================================
# BlockGeography Class Tests
# =============================================================================

class TestBlockGeography:
    """Test BlockGeography class functionality."""

    @pytest.fixture
    def data_path(self):
        return Path(__file__).parent.parent / "data" / "block_probabilities.parquet"

    @pytest.fixture
    def geo(self, data_path):
        """BlockGeography instance with data."""
        if not data_path.exists():
            pytest.skip("Block probabilities data not available")
        return BlockGeography(data_path, lazy_load=False)

    def test_lazy_load_default(self, data_path):
        """Default is lazy loading."""
        if not data_path.exists():
            pytest.skip("Block probabilities data not available")
        geo = BlockGeography(data_path)
        assert geo._data is None

    def test_eager_load(self, data_path):
        """Can force eager loading."""
        if not data_path.exists():
            pytest.skip("Block probabilities data not available")
        geo = BlockGeography(data_path, lazy_load=False)
        assert geo._data is not None

    def test_data_property_loads(self, data_path):
        """Accessing data property triggers load."""
        if not data_path.exists():
            pytest.skip("Block probabilities data not available")
        geo = BlockGeography(data_path)
        assert geo._data is None
        _ = geo.data
        assert geo._data is not None

    def test_get_cd_requires_lookup(self, geo):
        """Congressional district requires data lookup."""
        # Get a real block from the data
        sample_block = geo.data["geoid"].iloc[0]
        expected_cd = geo.data[geo.data["geoid"] == sample_block]["cd_id"].iloc[0]
        result = geo.get_cd(sample_block)
        assert result == expected_cd

    def test_get_cd_unknown_block(self, geo):
        """Unknown block returns None for CD."""
        result = geo.get_cd("000000000000000")
        assert result is None

    def test_get_all_geographies(self, geo):
        """Get all geographies returns dict with all keys."""
        sample_block = geo.data["geoid"].iloc[0]
        result = geo.get_all_geographies(sample_block)

        assert isinstance(result, dict)
        assert "state_fips" in result
        assert "county_fips" in result
        assert "tract_geoid" in result
        assert "cd_id" in result

    def test_states_property(self, geo):
        """States property returns sorted list."""
        states = geo.states
        assert isinstance(states, list)
        assert len(states) > 0
        assert states == sorted(states)

    def test_n_blocks_property(self, geo):
        """n_blocks property returns count."""
        n = geo.n_blocks
        assert isinstance(n, int)
        assert n > 0
        assert n == len(geo.data)


# =============================================================================
# Sampling Tests
# =============================================================================

class TestBlockSampling:
    """Test block sampling functionality."""

    @pytest.fixture
    def geo(self):
        data_path = Path(__file__).parent.parent / "data" / "block_probabilities.parquet"
        if not data_path.exists():
            pytest.skip("Block probabilities data not available")
        return BlockGeography(data_path, lazy_load=False)

    def test_sample_blocks_returns_array(self, geo):
        """Sample blocks returns numpy array."""
        blocks = geo.sample_blocks("06", n=10, random_state=42)
        assert isinstance(blocks, np.ndarray)
        assert len(blocks) == 10

    def test_sample_blocks_from_correct_state(self, geo):
        """All sampled blocks are from requested state."""
        blocks = geo.sample_blocks("06", n=100, random_state=42)
        for block in blocks:
            assert BlockGeography.get_state(block) == "06"

    def test_sample_blocks_reproducible(self, geo):
        """Same seed gives same results."""
        blocks1 = geo.sample_blocks("06", n=10, random_state=42)
        blocks2 = geo.sample_blocks("06", n=10, random_state=42)
        np.testing.assert_array_equal(blocks1, blocks2)

    def test_sample_blocks_invalid_state(self, geo):
        """Invalid state raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            geo.sample_blocks("99", n=10)

    def test_sample_blocks_national(self, geo):
        """Sample blocks nationally."""
        blocks = geo.sample_blocks_national(n=100, random_state=42)
        assert len(blocks) == 100

        # Should have blocks from multiple states
        states = set(BlockGeography.get_state(b) for b in blocks)
        assert len(states) > 1

    def test_sample_blocks_weighted(self, geo):
        """Sampling is population-weighted."""
        # Sample many times and check distribution roughly matches population
        blocks = geo.sample_blocks("06", n=10000, random_state=42)

        # More populous blocks should appear more often
        counts = pd.Series(blocks).value_counts()
        top_sampled = counts.index[0]

        # Top sampled block should be in top 10% by population
        ca_data = geo.get_blocks_in_state("06")
        pop_threshold = ca_data["population"].quantile(0.90)
        top_block_pop = ca_data[ca_data["geoid"] == top_sampled]["population"].iloc[0]

        assert top_block_pop >= pop_threshold


# =============================================================================
# Query Tests
# =============================================================================

class TestBlockQueries:
    """Test block query methods."""

    @pytest.fixture
    def geo(self):
        data_path = Path(__file__).parent.parent / "data" / "block_probabilities.parquet"
        if not data_path.exists():
            pytest.skip("Block probabilities data not available")
        return BlockGeography(data_path, lazy_load=False)

    def test_get_blocks_in_state(self, geo):
        """Get all blocks in a state."""
        ca_blocks = geo.get_blocks_in_state("06")
        assert len(ca_blocks) > 0
        assert all(ca_blocks["state_fips"] == "06")

    def test_get_blocks_in_county(self, geo):
        """Get all blocks in a county."""
        # Los Angeles County
        la_blocks = geo.get_blocks_in_county("06037")
        assert len(la_blocks) > 0
        assert all(la_blocks["state_fips"] == "06")
        assert all(la_blocks["county"] == "037")

    def test_get_blocks_in_cd(self, geo):
        """Get all blocks in a congressional district."""
        # Get a valid CD from the data
        sample_cd = geo.data["cd_id"].dropna().iloc[0]
        cd_blocks = geo.get_blocks_in_cd(sample_cd)
        assert len(cd_blocks) > 0
        assert all(cd_blocks["cd_id"] == sample_cd)


# =============================================================================
# derive_geographies Function Tests
# =============================================================================

class TestDeriveGeographies:
    """Test the derive_geographies helper function."""

    def test_derive_basic_geographies(self):
        """Derive state, county, tract from block GEOIDs."""
        geoids = ["060372073021001", "010010201001000"]
        result = derive_geographies(geoids)

        assert len(result) == 2
        assert "block_geoid" in result.columns
        assert "state_fips" in result.columns
        assert "county_fips" in result.columns
        assert "tract_geoid" in result.columns

        assert result["state_fips"].iloc[0] == "06"
        assert result["state_fips"].iloc[1] == "01"

    def test_derive_with_cd(self):
        """Derive geographies including CD requires data."""
        data_path = Path(__file__).parent.parent / "data" / "block_probabilities.parquet"
        if not data_path.exists():
            pytest.skip("Block probabilities data not available")

        block_data = load_block_probabilities(data_path)
        sample_geoid = block_data["geoid"].iloc[0]

        result = derive_geographies([sample_geoid], include_cd=True, block_data=block_data)

        assert "cd_id" in result.columns
        assert result["cd_id"].iloc[0] is not None

    def test_derive_handles_series(self):
        """Works with pandas Series input."""
        geoids = pd.Series(["060372073021001", "010010201001000"])
        result = derive_geographies(geoids)
        assert len(result) == 2

    def test_derive_handles_array(self):
        """Works with numpy array input."""
        geoids = np.array(["060372073021001", "010010201001000"])
        result = derive_geographies(geoids)
        assert len(result) == 2


# =============================================================================
# State Legislative District (SLD) Tests
# =============================================================================

class TestSLDSupport:
    """Test State Legislative District (SLD) support.

    SLDs come in two types:
    - SLDU: Upper chamber (State Senate)
    - SLDL: Lower chamber (State House/Assembly)

    Nebraska has unicameral legislature, so only SLDU.
    """

    @pytest.fixture
    def data_path(self):
        return Path(__file__).parent.parent / "data" / "block_probabilities.parquet"

    @pytest.fixture
    def geo(self, data_path):
        """BlockGeography instance with data."""
        if not data_path.exists():
            pytest.skip("Block probabilities data not available")
        return BlockGeography(data_path, lazy_load=False)

    def test_block_data_has_sld_columns(self, geo):
        """Block data includes SLDU and SLDL columns."""
        assert "sldu_id" in geo.data.columns, "Missing sldu_id column"
        assert "sldl_id" in geo.data.columns, "Missing sldl_id column"

    def test_sldu_id_format(self, geo):
        """SLDU IDs follow format: STATE-SLDU-XXX."""
        sample_sldu = geo.data["sldu_id"].dropna().iloc[0]
        assert "-SLDU-" in sample_sldu, f"Invalid SLDU format: {sample_sldu}"
        # Format: XX-SLDU-YYY where XX is state abbrev
        parts = sample_sldu.split("-")
        assert len(parts) == 3
        assert len(parts[0]) == 2  # State abbreviation

    def test_sldl_id_format(self, geo):
        """SLDL IDs follow format: STATE-SLDL-XXX."""
        # Skip if no SLDL data (e.g., Nebraska is unicameral)
        sldl_data = geo.data["sldl_id"].dropna()
        if len(sldl_data) == 0:
            pytest.skip("No SLDL data available")
        sample_sldl = sldl_data.iloc[0]
        assert "-SLDL-" in sample_sldl, f"Invalid SLDL format: {sample_sldl}"
        parts = sample_sldl.split("-")
        assert len(parts) == 3
        assert len(parts[0]) == 2  # State abbreviation

    def test_get_sldu(self, geo):
        """Can lookup SLDU for a block."""
        sample_block = geo.data[geo.data["sldu_id"].notna()]["geoid"].iloc[0]
        result = geo.get_sldu(sample_block)
        assert result is not None
        assert "-SLDU-" in result

    def test_get_sldl(self, geo):
        """Can lookup SLDL for a block."""
        sldl_blocks = geo.data[geo.data["sldl_id"].notna()]
        if len(sldl_blocks) == 0:
            pytest.skip("No SLDL data available")
        sample_block = sldl_blocks["geoid"].iloc[0]
        result = geo.get_sldl(sample_block)
        assert result is not None
        assert "-SLDL-" in result

    def test_get_sld_unknown_block(self, geo):
        """Unknown block returns None for SLD."""
        assert geo.get_sldu("000000000000000") is None
        assert geo.get_sldl("000000000000000") is None

    def test_get_all_geographies_includes_sld(self, geo):
        """get_all_geographies includes SLD columns."""
        sample_block = geo.data[geo.data["sldu_id"].notna()]["geoid"].iloc[0]
        result = geo.get_all_geographies(sample_block)
        assert "sldu_id" in result
        assert "sldl_id" in result

    def test_get_blocks_in_sldu(self, geo):
        """Get all blocks in a state senate district."""
        sample_sldu = geo.data["sldu_id"].dropna().iloc[0]
        blocks = geo.get_blocks_in_sldu(sample_sldu)
        assert len(blocks) > 0
        assert all(blocks["sldu_id"] == sample_sldu)

    def test_get_blocks_in_sldl(self, geo):
        """Get all blocks in a state house district."""
        sldl_data = geo.data["sldl_id"].dropna()
        if len(sldl_data) == 0:
            pytest.skip("No SLDL data available")
        sample_sldl = sldl_data.iloc[0]
        blocks = geo.get_blocks_in_sldl(sample_sldl)
        assert len(blocks) > 0
        assert all(blocks["sldl_id"] == sample_sldl)

    def test_sldu_count_reasonable(self, geo):
        """Total SLDU districts should be ~2000 (50 states * ~40 each)."""
        n_sldu = geo.data["sldu_id"].dropna().nunique()
        # Should be between 1000 and 3000 (varies by state senate size)
        assert 1000 < n_sldu < 3000, f"Unexpected SLDU count: {n_sldu}"

    def test_sldl_count_reasonable(self, geo):
        """Total SLDL districts should be ~5000 (varies widely by state)."""
        sldl_unique = geo.data["sldl_id"].dropna().nunique()
        # Nebraska is unicameral, so might have 0 SLDL for some states
        # Total should be between 3000 and 10000
        if sldl_unique > 0:
            assert 3000 < sldl_unique < 10000, f"Unexpected SLDL count: {sldl_unique}"

    def test_derive_with_sld(self, data_path):
        """derive_geographies can include SLD columns."""
        if not data_path.exists():
            pytest.skip("Block probabilities data not available")

        block_data = load_block_probabilities(data_path)
        sample_geoid = block_data[block_data["sldu_id"].notna()]["geoid"].iloc[0]

        result = derive_geographies(
            [sample_geoid],
            include_cd=True,
            include_sld=True,
            block_data=block_data
        )

        assert "sldu_id" in result.columns
        assert "sldl_id" in result.columns
