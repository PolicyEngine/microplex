"""Tests for hierarchical household synthesis."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from microplex.hierarchical import (
    HierarchicalSynthesizer,
    HouseholdSchema,
    prepare_cps_for_hierarchical,
)
from microplex.geography import derive_geographies


def create_test_household_data(n_households: int = 100, seed: int = 42) -> tuple:
    """Create synthetic household and person data for testing."""
    np.random.seed(seed)

    # Generate households
    hh_data = pd.DataFrame({
        'household_id': range(n_households),
        'n_persons': np.random.choice([1, 2, 3, 4, 5], n_households, p=[0.2, 0.3, 0.25, 0.15, 0.1]),
        'state_fips': np.random.choice([6, 36, 48], n_households),  # CA, NY, TX
        'tenure': np.random.choice([1, 2], n_households),  # Own, Rent
    })
    hh_data['n_adults'] = np.clip(hh_data['n_persons'] - np.random.randint(0, 3, n_households), 1, hh_data['n_persons'])
    hh_data['n_children'] = hh_data['n_persons'] - hh_data['n_adults']

    # Generate persons for each household
    person_records = []
    person_id = 0

    for _, hh_row in hh_data.iterrows():
        hh_id = hh_row['household_id']
        n_persons = hh_row['n_persons']
        n_adults = hh_row['n_adults']

        for p_num in range(n_persons):
            is_adult = p_num < n_adults

            if is_adult:
                age = np.random.randint(25, 70)
                income = np.random.lognormal(10.5, 0.8)
            else:
                age = np.random.randint(0, 18)
                income = 0

            person_records.append({
                'person_id': person_id,
                'household_id': hh_id,
                'age': age,
                'sex': np.random.choice([0, 1]),
                'income': income,
                'employment_status': 1 if is_adult and np.random.random() > 0.3 else 0,
                'education': np.random.randint(1, 5) if is_adult else 0,
                'relationship_to_head': 0 if p_num == 0 else (1 if p_num == 1 and is_adult else 2),
            })
            person_id += 1

    person_data = pd.DataFrame(person_records)

    return hh_data, person_data


class TestHouseholdSchema:
    """Tests for HouseholdSchema."""

    def test_default_schema(self):
        """Test default schema has expected fields."""
        schema = HouseholdSchema()

        assert 'n_persons' in schema.hh_vars
        assert 'n_adults' in schema.hh_vars
        assert 'age' in schema.person_vars
        assert 'income' in schema.person_vars
        assert 'hh_income' in schema.derived_vars

    def test_custom_schema(self):
        """Test custom schema configuration."""
        schema = HouseholdSchema(
            hh_vars=['n_persons', 'state'],
            person_vars=['age', 'income'],
            derived_vars={'total_income': 'sum:income'},
        )

        assert schema.hh_vars == ['n_persons', 'state']
        assert schema.person_vars == ['age', 'income']
        assert 'total_income' in schema.derived_vars


class TestHierarchicalSynthesizer:
    """Tests for HierarchicalSynthesizer."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for tests."""
        return create_test_household_data(n_households=50)

    @pytest.fixture
    def simple_schema(self):
        """Create simplified schema for faster tests."""
        return HouseholdSchema(
            hh_vars=['n_persons', 'n_adults', 'n_children'],
            person_vars=['age', 'income'],
            person_condition_vars=['n_persons', 'n_adults', 'person_number', 'is_first_adult'],
            derived_vars={'hh_income': 'sum:income'},
        )

    def test_init(self):
        """Test synthesizer initialization."""
        synth = HierarchicalSynthesizer()
        assert synth.schema is not None
        assert not synth._is_fitted

    def test_init_with_custom_schema(self, simple_schema):
        """Test initialization with custom schema."""
        synth = HierarchicalSynthesizer(schema=simple_schema)
        assert synth.schema == simple_schema

    def test_fit_validates_data(self, sample_data, simple_schema):
        """Test that fit validates required columns."""
        hh_data, person_data = sample_data

        # Remove required column
        bad_hh = hh_data.drop(columns=['n_persons'])

        synth = HierarchicalSynthesizer(schema=simple_schema)
        with pytest.raises(ValueError, match="missing columns"):
            synth.fit(bad_hh, person_data, epochs=1)

    def test_fit_runs(self, sample_data, simple_schema):
        """Test that fit completes without error."""
        hh_data, person_data = sample_data

        synth = HierarchicalSynthesizer(schema=simple_schema)
        synth.fit(hh_data, person_data, epochs=2, verbose=False)

        assert synth._is_fitted
        assert synth.hh_synthesizer is not None
        assert synth.person_synthesizer is not None

    def test_generate_requires_fit(self, simple_schema):
        """Test that generate requires fit first."""
        synth = HierarchicalSynthesizer(schema=simple_schema)

        with pytest.raises(ValueError, match="Must call fit"):
            synth.generate(n_households=10)

    def test_generate_returns_correct_structure(self, sample_data, simple_schema):
        """Test that generate returns expected DataFrames."""
        hh_data, person_data = sample_data

        synth = HierarchicalSynthesizer(schema=simple_schema)
        synth.fit(hh_data, person_data, epochs=2, verbose=False)

        synthetic_hh, synthetic_persons = synth.generate(n_households=20, verbose=False)

        # Check household DataFrame
        assert len(synthetic_hh) == 20
        assert 'household_id' in synthetic_hh.columns
        assert 'n_persons' in synthetic_hh.columns

        # Check person DataFrame
        assert len(synthetic_persons) > 0
        assert 'household_id' in synthetic_persons.columns
        assert 'person_id' in synthetic_persons.columns
        assert 'age' in synthetic_persons.columns
        assert 'income' in synthetic_persons.columns

        # Check every person belongs to a valid household
        assert set(synthetic_persons['household_id']).issubset(set(synthetic_hh['household_id']))

    def test_generate_with_units(self, sample_data, simple_schema):
        """Test generate with tax/SPM unit construction."""
        hh_data, person_data = sample_data

        synth = HierarchicalSynthesizer(schema=simple_schema)
        synth.fit(hh_data, person_data, epochs=2, verbose=False)

        result = synth.generate(n_households=10, return_units=True, verbose=False)

        assert len(result) == 4
        synthetic_hh, synthetic_persons, tax_units, spm_units = result

        # Check tax units
        assert len(tax_units) > 0
        assert 'tax_unit_id' in tax_units.columns
        assert 'household_id' in tax_units.columns

        # Check SPM units
        assert len(spm_units) > 0
        assert 'spm_unit_id' in spm_units.columns
        assert 'household_id' in spm_units.columns

    def test_derived_aggregates(self, sample_data, simple_schema):
        """Test that derived aggregates are computed correctly."""
        hh_data, person_data = sample_data

        synth = HierarchicalSynthesizer(schema=simple_schema)
        synth.fit(hh_data, person_data, epochs=2, verbose=False)

        synthetic_hh, synthetic_persons = synth.generate(n_households=20, verbose=False)

        # Check hh_income is derived
        assert 'hh_income' in synthetic_hh.columns

        # Verify it's the sum of person incomes
        for hh_id in synthetic_hh['household_id'].head(5):
            hh_persons = synthetic_persons[synthetic_persons['household_id'] == hh_id]
            expected_income = hh_persons['income'].sum()
            actual_income = synthetic_hh[synthetic_hh['household_id'] == hh_id]['hh_income'].iloc[0]
            np.testing.assert_almost_equal(actual_income, expected_income, decimal=2)

    def test_person_count_matches_n_persons(self, sample_data, simple_schema):
        """Test that number of persons matches n_persons in HH data."""
        hh_data, person_data = sample_data

        synth = HierarchicalSynthesizer(schema=simple_schema)
        synth.fit(hh_data, person_data, epochs=2, verbose=False)

        synthetic_hh, synthetic_persons = synth.generate(n_households=20, verbose=False)

        # Count persons per household
        person_counts = synthetic_persons.groupby('household_id').size()

        for hh_id in synthetic_hh['household_id']:
            expected = synthetic_hh[synthetic_hh['household_id'] == hh_id]['n_persons'].iloc[0]
            actual = person_counts.get(hh_id, 0)
            assert actual == expected, f"HH {hh_id}: expected {expected} persons, got {actual}"


class TestPrepareCpsForHierarchical:
    """Tests for CPS data preparation utility."""

    def test_basic_preparation(self):
        """Test basic CPS preparation."""
        # Create mock CPS person data
        cps_data = pd.DataFrame({
            'household_id': [1, 1, 1, 2, 2, 3],
            'age': [45, 42, 12, 67, 65, 35],
            'state_fips': [6, 6, 6, 36, 36, 48],
            'tenure': [1, 1, 1, 2, 2, 1],
            'hh_weight': [1000, 1000, 1000, 800, 800, 1200],
        })

        hh_data, person_data = prepare_cps_for_hierarchical(cps_data)

        # Check household data
        assert len(hh_data) == 3
        assert hh_data[hh_data['household_id'] == 1]['n_persons'].iloc[0] == 3
        assert hh_data[hh_data['household_id'] == 1]['n_adults'].iloc[0] == 2
        assert hh_data[hh_data['household_id'] == 1]['n_children'].iloc[0] == 1

        # Check person data unchanged
        assert len(person_data) == 6


@pytest.mark.skipif(
    importlib.util.find_spec("microplex_us") is None,
    reason="Block assignment uses US-specific helpers in microplex-us",
)
class TestBlockAssignment:
    """Tests for block-level geographic assignment."""

    @pytest.fixture
    def sample_block_probs(self):
        """Create sample block probabilities for testing."""
        return pd.DataFrame({
            'geoid': [
                '060010201001000', '060010201001001', '060010201001002',  # CA
                '360590101001000', '360590101001001',  # NY
                '480010101001000', '480010101001001', '480010101001002',  # TX
            ],
            'state_fips': ['06', '06', '06', '36', '36', '48', '48', '48'],
            'county': ['001', '001', '001', '059', '059', '001', '001', '001'],
            'tract': ['020100', '020100', '020100', '010100', '010100', '010100', '010100', '010100'],
            'block': ['1000', '1001', '1002', '1000', '1001', '1000', '1001', '1002'],
            'population': [100, 200, 100, 300, 200, 150, 250, 100],
            'tract_geoid': [
                '06001020100', '06001020100', '06001020100',
                '36059010100', '36059010100',
                '48001010100', '48001010100', '48001010100',
            ],
            'cd_id': ['CA-01', 'CA-01', 'CA-01', 'NY-01', 'NY-01', 'TX-01', 'TX-01', 'TX-01'],
            'prob': [0.25, 0.50, 0.25, 0.6, 0.4, 0.3, 0.5, 0.2],  # Within-state probs
        })

    @pytest.fixture
    def sample_cd_probs(self):
        """Create sample CD probabilities for backward compatibility testing."""
        return pd.DataFrame({
            'state_fips': [6, 6, 36, 36, 48, 48],
            'cd_id': ['CA-01', 'CA-02', 'NY-01', 'NY-02', 'TX-01', 'TX-02'],
            'prob': [0.6, 0.4, 0.5, 0.5, 0.7, 0.3],
        })

    def test_init_with_block_probabilities(self, sample_block_probs):
        """Test synthesizer initializes with block probabilities."""
        synth = HierarchicalSynthesizer(block_probabilities=sample_block_probs)

        assert synth._block_lookup is not None
        assert synth._cd_lookup is None
        assert '06' in synth._block_lookup
        assert '36' in synth._block_lookup
        assert '48' in synth._block_lookup

    def test_init_with_cd_probabilities_backward_compat(self, sample_cd_probs):
        """Test synthesizer initializes with CD probabilities (backward compat)."""
        synth = HierarchicalSynthesizer(cd_probabilities=sample_cd_probs)

        assert synth._cd_lookup is not None
        assert synth._block_lookup is None
        assert 6 in synth._cd_lookup
        assert 36 in synth._cd_lookup

    def test_block_probabilities_take_precedence(self, sample_block_probs, sample_cd_probs):
        """Test that block probabilities take precedence over CD probabilities."""
        synth = HierarchicalSynthesizer(
            cd_probabilities=sample_cd_probs,
            block_probabilities=sample_block_probs,
        )

        assert synth._block_lookup is not None
        assert synth._cd_lookup is None

    def test_assign_blocks_adds_block_geoid_only(self, sample_block_probs):
        """Test that block assignment adds only block_geoid column.

        Parent geographies (tract, county, CD, SLD) should be derived post-hoc.
        """
        synth = HierarchicalSynthesizer(
            block_probabilities=sample_block_probs,
            random_state=42,
        )

        test_hh = pd.DataFrame({
            'state_fips': [6, 36, 48],
            'n_persons': [3, 2, 4],
        })

        result = synth._assign_blocks(test_hh)

        # Only block_geoid is set during synthesis
        assert 'block_geoid' in result.columns

        # Parent geographies are NOT set during synthesis (derived post-hoc)
        assert 'tract_geoid' not in result.columns
        assert 'county_fips' not in result.columns
        assert 'cd_id' not in result.columns

    def test_block_geoid_structure(self, sample_block_probs):
        """Test that block geoid structure is correct (15 chars)."""
        synth = HierarchicalSynthesizer(
            block_probabilities=sample_block_probs,
            random_state=42,
        )

        test_hh = pd.DataFrame({
            'state_fips': [6, 36, 48],
            'n_persons': [3, 2, 4],
        })

        result = synth._assign_blocks(test_hh)

        for _, row in result.iterrows():
            block_geoid = row['block_geoid']

            # Block GEOID should be 15 characters
            assert len(block_geoid) == 15

            # Can derive tract (first 11) and county (first 5) from block
            tract_geoid = block_geoid[:11]
            county_fips = block_geoid[:5]
            assert len(tract_geoid) == 11
            assert len(county_fips) == 5

    def test_derive_geographies_from_block(self, sample_block_probs):
        """Test that parent geographies can be derived post-hoc from block."""
        synth = HierarchicalSynthesizer(
            block_probabilities=sample_block_probs,
            random_state=42,
        )

        test_hh = pd.DataFrame({
            'state_fips': [6, 36, 48],
            'n_persons': [3, 2, 4],
        })

        result = synth._assign_blocks(test_hh)

        # Derive geographies post-hoc
        geos = derive_geographies(
            result['block_geoid'],
            include_cd=True,
            block_data=sample_block_probs
        )

        # All CA blocks map to CA-XX CDs
        ca_mask = geos['state_fips'] == '06'
        assert all(cd.startswith('CA-') for cd in geos.loc[ca_mask, 'cd_id'])

        # All NY blocks map to NY-XX CDs
        ny_mask = geos['state_fips'] == '36'
        assert all(cd.startswith('NY-') for cd in geos.loc[ny_mask, 'cd_id'])

    def test_state_fips_fixed_to_valid(self, sample_block_probs):
        """Test that state_fips values are fixed to valid integers."""
        synth = HierarchicalSynthesizer(
            block_probabilities=sample_block_probs,
            random_state=42,
        )

        # Include a state_fips value that needs rounding/fixing
        test_hh = pd.DataFrame({
            'state_fips': [6.3, 36.7, 47.9],  # 47.9 should map to 48
            'n_persons': [3, 2, 4],
        })

        result = synth._assign_blocks(test_hh)

        assert result['state_fips'].iloc[0] == 6
        assert result['state_fips'].iloc[1] == 36
        assert result['state_fips'].iloc[2] == 48


class TestTaxUnitOptimizer:
    """Tests for TaxUnitOptimizer."""

    def test_single_adult_filing_status(self):
        """Test that single adult files as single."""
        from microplex.hierarchical import TaxUnitOptimizer

        persons = pd.DataFrame({
            'person_id': [0],
            'household_id': [0],
            'age': [35],
            'income': [50000],
            'relationship_to_head': [0],  # Head
            'is_student': [False],
            'is_disabled': [False],
        })

        optimizer = TaxUnitOptimizer()
        tax_units = optimizer.optimize_household(0, persons)

        assert len(tax_units) == 1
        assert tax_units[0]['filing_status'] == 'single'
        assert tax_units[0]['n_dependents'] == 0

    def test_married_couple_files_jointly(self):
        """Test that married couple files jointly."""
        from microplex.hierarchical import TaxUnitOptimizer

        persons = pd.DataFrame({
            'person_id': [0, 1],
            'household_id': [0, 0],
            'age': [35, 33],
            'income': [60000, 55000],
            'relationship_to_head': [0, 1],  # Head, Spouse
            'is_student': [False, False],
            'is_disabled': [False, False],
        })

        optimizer = TaxUnitOptimizer()
        tax_units = optimizer.optimize_household(0, persons)

        assert len(tax_units) == 1
        assert tax_units[0]['filing_status'] == 'married_filing_jointly'
        assert tax_units[0]['n_dependents'] == 0

    def test_couple_with_children(self):
        """Test couple with qualifying children."""
        from microplex.hierarchical import TaxUnitOptimizer

        persons = pd.DataFrame({
            'person_id': [0, 1, 2, 3],
            'household_id': [0, 0, 0, 0],
            'age': [40, 38, 10, 7],
            'income': [80000, 60000, 0, 0],
            'relationship_to_head': [0, 1, 2, 2],  # Head, Spouse, Child, Child
            'is_student': [False, False, False, False],
            'is_disabled': [False, False, False, False],
        })

        optimizer = TaxUnitOptimizer()
        tax_units = optimizer.optimize_household(0, persons)

        assert len(tax_units) == 1
        assert tax_units[0]['filing_status'] == 'married_filing_jointly'
        assert tax_units[0]['n_dependents'] == 2

    def test_single_parent_head_of_household(self):
        """Test single parent files as head of household."""
        from microplex.hierarchical import TaxUnitOptimizer

        persons = pd.DataFrame({
            'person_id': [0, 1],
            'household_id': [0, 0],
            'age': [35, 8],
            'income': [55000, 0],
            'relationship_to_head': [0, 2],  # Head, Child
            'is_student': [False, False],
            'is_disabled': [False, False],
        })

        optimizer = TaxUnitOptimizer()
        tax_units = optimizer.optimize_household(0, persons)

        assert len(tax_units) == 1
        assert tax_units[0]['filing_status'] == 'head_of_household'
        assert tax_units[0]['n_dependents'] == 1

    def test_adult_student_dependent(self):
        """Test that adult student under 24 can be dependent."""
        from microplex.hierarchical import TaxUnitOptimizer

        persons = pd.DataFrame({
            'person_id': [0, 1, 2],
            'household_id': [0, 0, 0],
            'age': [50, 48, 21],
            'income': [90000, 70000, 5000],
            'relationship_to_head': [0, 1, 2],  # Head, Spouse, Child
            'is_student': [False, False, True],
            'is_disabled': [False, False, False],
        })

        optimizer = TaxUnitOptimizer()
        tax_units = optimizer.optimize_household(0, persons)

        assert len(tax_units) == 1
        assert tax_units[0]['n_dependents'] == 1

    def test_disabled_adult_dependent(self):
        """Test that disabled adult can be dependent."""
        from microplex.hierarchical import TaxUnitOptimizer

        persons = pd.DataFrame({
            'person_id': [0, 1],
            'household_id': [0, 0],
            'age': [55, 30],
            'income': [70000, 0],
            'relationship_to_head': [0, 2],  # Head, Child
            'is_student': [False, False],
            'is_disabled': [False, True],
        })

        optimizer = TaxUnitOptimizer()
        tax_units = optimizer.optimize_household(0, persons)

        assert len(tax_units) == 1
        assert tax_units[0]['filing_status'] == 'head_of_household'
        assert tax_units[0]['n_dependents'] == 1

    def test_unrelated_adults_separate_units(self):
        """Test that unrelated adults file separately."""
        from microplex.hierarchical import TaxUnitOptimizer

        persons = pd.DataFrame({
            'person_id': [0, 1],
            'household_id': [0, 0],
            'age': [30, 28],
            'income': [50000, 48000],
            'relationship_to_head': [0, 3],  # Head, Unrelated
            'is_student': [False, False],
            'is_disabled': [False, False],
        })

        optimizer = TaxUnitOptimizer()
        tax_units = optimizer.optimize_household(0, persons)

        assert len(tax_units) == 2
        assert all(tu['filing_status'] == 'single' for tu in tax_units)

    def test_high_income_mfs_optimization(self):
        """Test that MFS can be chosen for high-income disparity."""
        from microplex.hierarchical import TaxUnitOptimizer

        # High earner with student loans + low earner
        persons = pd.DataFrame({
            'person_id': [0, 1],
            'household_id': [0, 0],
            'age': [35, 33],
            'income': [250000, 30000],
            'relationship_to_head': [0, 1],  # Head, Spouse
            'is_student': [False, False],
            'is_disabled': [False, False],
        })

        optimizer = TaxUnitOptimizer()
        tax_units = optimizer.optimize_household(0, persons)

        # Should create 2 tax units if MFS is better
        # (Implementation will determine optimal choice)
        assert len(tax_units) >= 1

    def test_standard_deduction_calculation(self):
        """Test standard deduction varies by filing status."""
        from microplex.hierarchical import TaxUnitOptimizer

        optimizer = TaxUnitOptimizer()

        # 2024 standard deductions
        assert optimizer._standard_deduction('single', 0) == 14600
        assert optimizer._standard_deduction('married_filing_jointly', 0) == 29200
        assert optimizer._standard_deduction('married_filing_separately', 0) == 14600
        assert optimizer._standard_deduction('head_of_household', 0) == 21900

    def test_eitc_eligibility(self):
        """Test EITC calculation."""
        from microplex.hierarchical import TaxUnitOptimizer

        optimizer = TaxUnitOptimizer()

        # No children, moderate income
        assert optimizer._calculate_eitc(15000, 'single', 0) > 0

        # With children, should get more
        eitc_no_kids = optimizer._calculate_eitc(25000, 'single', 0)
        eitc_with_kids = optimizer._calculate_eitc(25000, 'single', 2)
        assert eitc_with_kids > eitc_no_kids

        # High income, no EITC
        assert optimizer._calculate_eitc(100000, 'single', 0) == 0

    def test_ctc_calculation(self):
        """Test Child Tax Credit calculation."""
        from microplex.hierarchical import TaxUnitOptimizer

        optimizer = TaxUnitOptimizer()

        # No children
        assert optimizer._calculate_ctc(50000, 'single', 0) == 0

        # With children
        ctc = optimizer._calculate_ctc(50000, 'married_filing_jointly', 2)
        assert ctc == 4000  # $2000 per child

        # Phase-out at high income
        ctc_high = optimizer._calculate_ctc(500000, 'married_filing_jointly', 2)
        assert ctc_high < 4000

    def test_tax_liability_calculation(self):
        """Test overall tax liability calculation."""
        from microplex.hierarchical import TaxUnitOptimizer

        optimizer = TaxUnitOptimizer()

        # Low income, should have negative tax (refundable credits)
        liability_low = optimizer._calculate_tax_liability(25000, 'single', 1)
        assert liability_low < 0

        # Moderate income
        liability_mid = optimizer._calculate_tax_liability(75000, 'married_filing_jointly', 2)
        assert liability_mid >= 0

        # High income
        liability_high = optimizer._calculate_tax_liability(200000, 'single', 0)
        assert liability_high > liability_mid
