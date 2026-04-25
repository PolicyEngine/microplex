"""
Hierarchical synthesis for household microdata.

Two-pass approach:
1. Synthesize household skeleton (composition, location, tenure)
2. Synthesize person attributes conditioned on household context

Then derive aggregates (HH income = sum of person incomes) and
construct tax units / SPM units algorithmically.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from .geography import (
    AtomicGeographyCrosswalk,
    GeographyAssignmentPlan,
    GeographyProvider,
    ProbabilisticAtomicGeographyAssigner,
    StaticGeographyProvider,
    nearest_numeric_partition_key,
    normalize_us_state_fips,
)
from .synthesizer import Synthesizer


@dataclass
class HouseholdSchema:
    """Schema defining household and person variables."""

    # Household-level variables (Pass 1)
    hh_vars: List[str] = field(default_factory=lambda: [
        'n_persons', 'n_adults', 'n_children', 'state_fips', 'tenure'
    ])

    # Person-level variables to synthesize (Pass 2)
    person_vars: List[str] = field(default_factory=lambda: [
        'age', 'sex', 'income', 'employment_status', 'education',
        'relationship_to_head'
    ])

    # Person-level conditioning variables (from HH + position)
    person_condition_vars: List[str] = field(default_factory=lambda: [
        'n_persons', 'n_adults', 'n_children', 'state_fips', 'tenure',
        'person_number', 'is_first_adult', 'is_child_slot'
    ])

    # Variables to derive by aggregation (not modeled)
    derived_vars: Dict[str, str] = field(default_factory=lambda: {
        'hh_income': 'sum:income',
        'hh_benefits': 'sum:benefits',
        'n_workers': 'count:employment_status==1',
    })

    # ID columns
    hh_id_col: str = 'household_id'
    person_id_col: str = 'person_id'


class HierarchicalSynthesizer:
    """
    Two-pass hierarchical synthesizer for household microdata.

    Pass 1: Learn P(household_features) from data
    Pass 2: Learn P(person_attributes | household_features) from data

    Then:
    - Generate households
    - Generate persons per household
    - Derive aggregates (HH income = sum of person incomes)
    - Construct tax units / SPM units algorithmically

    Example:
        >>> synth = HierarchicalSynthesizer()
        >>> synth.fit(hh_data, person_data)
        >>> synthetic_hh, synthetic_persons = synth.generate(n_households=10000)
    """

    def __init__(
        self,
        schema: Optional[HouseholdSchema] = None,
        hh_flow_kwargs: Optional[Dict] = None,
        person_flow_kwargs: Optional[Dict] = None,
        random_state: Optional[int] = None,
        cd_probabilities: Optional[pd.DataFrame] = None,
        block_probabilities: Optional[pd.DataFrame] = None,
        geography_provider: Optional[GeographyProvider] = None,
        geography_assignment: Optional[GeographyAssignmentPlan] = None,
    ):
        """
        Initialize hierarchical synthesizer.

        Args:
            schema: HouseholdSchema defining variables at each level
            hh_flow_kwargs: Kwargs passed to household-level Synthesizer
            person_flow_kwargs: Kwargs passed to person-level Synthesizer
            random_state: Random seed for reproducibility
            cd_probabilities: DataFrame with state_fips, cd_id, prob columns
                for assigning congressional districts during synthesis.
                Ignored if block_probabilities is provided.
            block_probabilities: DataFrame with geoid, state_fips, prob, cd_id
                columns for assigning census blocks during synthesis. When
                provided, tract_geoid, county_fips, and cd_id are derived
                from the assigned block.
            geography_provider: Generic atomic geography provider. Takes
                precedence over probability DataFrames when supplied.
            geography_assignment: Assignment plan used with geography_provider.
        """
        self.schema = schema or HouseholdSchema()
        self.hh_flow_kwargs = hh_flow_kwargs or {}
        self.person_flow_kwargs = person_flow_kwargs or {}
        self.random_state = random_state
        self.cd_probabilities = cd_probabilities
        self.block_probabilities = block_probabilities

        self.hh_synthesizer: Optional[Synthesizer] = None
        self.person_synthesizer: Optional[Synthesizer] = None

        self._hh_data: Optional[pd.DataFrame] = None
        self._person_data: Optional[pd.DataFrame] = None
        self._is_fitted = False

        # Precompute lookups for fast assignment
        self._cd_lookup: Optional[Dict] = None
        self._block_lookup: Optional[Dict] = None
        self.geography_assignment = geography_assignment
        self._geography_provider = geography_provider
        self._geography_assigner: Optional[ProbabilisticAtomicGeographyAssigner] = None

        # Prefer block probabilities if provided
        if geography_provider is not None:
            if geography_assignment is None:
                raise ValueError(
                    "geography_assignment is required when geography_provider is supplied"
                )
            self._geography_assigner = geography_provider.load_assigner(
                geography_assignment.to_query()
            )
        elif block_probabilities is not None:
            self._configure_block_geography_assignment(block_probabilities)
            self._build_block_lookup(block_probabilities)
        elif cd_probabilities is not None:
            self._configure_cd_geography_assignment(cd_probabilities)
            self._build_cd_lookup(cd_probabilities)

    def _configure_block_geography_assignment(self, block_probs: pd.DataFrame) -> None:
        """Configure generic geography assignment from US Census block probabilities."""
        crosswalk_data = block_probs.copy()
        if "geoid" in crosswalk_data.columns and "block_geoid" not in crosswalk_data.columns:
            crosswalk_data = crosswalk_data.rename(columns={"geoid": "block_geoid"})
        self.geography_assignment = GeographyAssignmentPlan(
            partition_columns=("state_fips",),
            atomic_id_column="block_geoid",
            probability_column="prob",
            partition_normalizers={"state_fips": normalize_us_state_fips},
            fallback_resolver=nearest_numeric_partition_key,
        )
        self._geography_provider = StaticGeographyProvider(
            crosswalk=AtomicGeographyCrosswalk(
                data=crosswalk_data,
                atomic_id_column="block_geoid",
                probability_column="prob",
            ),
            default_partition_columns=("state_fips",),
            default_partition_normalizers={"state_fips": normalize_us_state_fips},
            default_fallback_resolver=nearest_numeric_partition_key,
        )
        self._geography_assigner = self._geography_provider.load_assigner(
            self.geography_assignment.to_query()
        )

    def _configure_cd_geography_assignment(self, cd_probs: pd.DataFrame) -> None:
        """Configure generic geography assignment from congressional district probabilities."""
        crosswalk_data = cd_probs.copy()
        atomic_id_column = "cd_id"
        geography_columns: tuple[str, ...] = ()
        if crosswalk_data["cd_id"].duplicated().any():
            atomic_id_column = "_microplex_cd_atomic_id"
            crosswalk_data[atomic_id_column] = (
                crosswalk_data["state_fips"].map(normalize_us_state_fips).astype(str)
                + "::"
                + crosswalk_data["cd_id"].astype(str)
            )
            geography_columns = ("cd_id",)
        self.geography_assignment = GeographyAssignmentPlan(
            partition_columns=("state_fips",),
            atomic_id_column=atomic_id_column,
            geography_columns=geography_columns,
            probability_column="prob",
            fallback_resolver=nearest_numeric_partition_key,
        )
        self._geography_provider = StaticGeographyProvider(
            crosswalk=AtomicGeographyCrosswalk(
                data=crosswalk_data,
                atomic_id_column=atomic_id_column,
                probability_column="prob",
            ),
            default_partition_columns=("state_fips",),
            default_fallback_resolver=nearest_numeric_partition_key,
        )
        self._geography_assigner = self._geography_provider.load_assigner(
            self.geography_assignment.to_query()
        )

    def _apply_geography_assignment(self, hh: pd.DataFrame) -> pd.DataFrame:
        """Assign atomic geography ids and sync requested parent columns."""
        if self._geography_assigner is None or self.geography_assignment is None:
            return hh
        assigned = self._geography_assigner.assign(
            hh,
            atomic_id_column=self.geography_assignment.atomic_id_column,
            random_state=self.random_state,
        )
        if self._geography_provider is None:
            return assigned
        requested_columns = self.geography_assignment.requested_geography_columns()
        if not requested_columns:
            return assigned
        crosswalk = self._geography_provider.load_crosswalk(
            self.geography_assignment.to_query()
        )
        materialized = crosswalk.materialize(
            assigned,
            columns=requested_columns,
            atomic_id_column=self.geography_assignment.atomic_id_column,
            overwrite=True,
        )
        if self.geography_assignment.atomic_id_column.startswith("_microplex_"):
            materialized = materialized.drop(
                columns=[self.geography_assignment.atomic_id_column],
                errors="ignore",
            )
        return materialized

    def _build_cd_lookup(self, cd_probs: pd.DataFrame) -> None:
        """Build lookup dict for fast CD assignment by state.

        Args:
            cd_probs: DataFrame with columns: state_fips, cd_id, prob
        """
        self._cd_lookup = {}
        for state_fips in cd_probs['state_fips'].unique():
            state_cds = cd_probs[cd_probs['state_fips'] == state_fips]
            self._cd_lookup[int(state_fips)] = {
                'cd_ids': state_cds['cd_id'].values,
                'probs': state_cds['prob'].values,
            }

    def _build_block_lookup(self, block_probs: pd.DataFrame) -> None:
        """Build lookup dict for fast block assignment by state.

        Precomputes block geoids and probabilities for each state.
        Only stores minimal data needed for block assignment - all parent
        geographies (tract, county, CD, SLD) should be derived post-hoc
        using BlockGeography.

        Args:
            block_probs: DataFrame with columns: geoid, state_fips, prob
        """
        self._block_lookup = {}

        # Ensure state_fips is string for consistent handling
        block_probs = block_probs.copy()
        block_probs['state_fips'] = block_probs['state_fips'].astype(str)

        for state_fips in block_probs['state_fips'].unique():
            state_blocks = block_probs[block_probs['state_fips'] == state_fips]

            # Normalize probabilities within state (should already sum to 1)
            probs = state_blocks['prob'].values
            probs = probs / probs.sum()

            self._block_lookup[state_fips] = {
                'geoids': state_blocks['geoid'].values,
                'probs': probs,
            }

    def _assign_blocks(self, hh: pd.DataFrame) -> pd.DataFrame:
        """Assign census blocks to households based on state.

        Uses pseudorandom assignment weighted by block population shares.
        Only assigns block_geoid - all parent geographies should be derived
        post-hoc using BlockGeography or derive_geographies().

        Args:
            hh: Household DataFrame with state_fips column

        Returns:
            DataFrame with block_geoid added
        """
        if self._block_lookup is None:
            return hh

        hh = hh.copy()
        rng = np.random.default_rng(self.random_state)

        # Get valid state FIPS codes (as strings)
        valid_fips = np.array(list(self._block_lookup.keys()))

        # Convert state_fips to string for lookup
        state_fips_values = hh['state_fips'].values

        # Block assignment
        block_geoids = []
        fixed_state_fips = []

        for state_fips in state_fips_values:
            # Convert to padded string (e.g., 6 -> "06")
            state_fips_str = str(int(round(state_fips))).zfill(2)

            # If not valid, find nearest valid FIPS
            if state_fips_str not in self._block_lookup:
                # Find closest valid state FIPS by numeric distance
                valid_int = np.array([int(f) for f in valid_fips])
                diffs = np.abs(valid_int - int(state_fips_str))
                state_fips_str = valid_fips[np.argmin(diffs)]

            lookup = self._block_lookup[state_fips_str]

            # Random weighted selection of block
            idx = rng.choice(len(lookup['geoids']), p=lookup['probs'])

            block_geoids.append(lookup['geoids'][idx])
            fixed_state_fips.append(int(state_fips_str))

        hh['block_geoid'] = block_geoids
        hh['state_fips'] = fixed_state_fips

        return hh

    def _assign_cds(self, hh: pd.DataFrame) -> pd.DataFrame:
        """Assign congressional districts to households based on state.

        Uses pseudorandom assignment weighted by CD population shares.

        Args:
            hh: Household DataFrame with state_fips column

        Returns:
            DataFrame with cd_id column added
        """
        if self._cd_lookup is None:
            return hh

        hh = hh.copy()
        rng = np.random.default_rng(self.random_state)

        # Get valid state FIPS codes
        valid_fips = np.array(list(self._cd_lookup.keys()))

        # Vectorized CD assignment by state
        cd_assignments = []
        for state_fips in hh['state_fips'].values:
            # Round to nearest valid state FIPS
            state_fips_rounded = int(round(state_fips))

            # If not valid, find nearest valid FIPS
            if state_fips_rounded not in self._cd_lookup:
                # Find closest valid state FIPS
                diffs = np.abs(valid_fips - state_fips)
                state_fips_rounded = valid_fips[np.argmin(diffs)]

            lookup = self._cd_lookup[state_fips_rounded]
            cd_id = rng.choice(lookup['cd_ids'], p=lookup['probs'])
            cd_assignments.append(cd_id)

        hh['cd_id'] = cd_assignments

        # Also fix the state_fips to be a valid integer
        fixed_fips = []
        for state_fips in hh['state_fips'].values:
            rounded = int(round(state_fips))
            if rounded not in self._cd_lookup:
                diffs = np.abs(valid_fips - state_fips)
                rounded = valid_fips[np.argmin(diffs)]
            fixed_fips.append(rounded)
        hh['state_fips'] = fixed_fips

        return hh

    def fit(
        self,
        hh_data: pd.DataFrame,
        person_data: pd.DataFrame,
        hh_weight_col: Optional[str] = None,
        person_weight_col: Optional[str] = None,
        epochs: int = 100,
        verbose: bool = True,
    ) -> 'HierarchicalSynthesizer':
        """
        Fit the two-pass hierarchical model.

        Args:
            hh_data: Household-level data (one row per household)
            person_data: Person-level data (one row per person, with HH ID)
            hh_weight_col: Weight column for households
            person_weight_col: Weight column for persons
            epochs: Training epochs for each flow
            verbose: Print progress

        Returns:
            self
        """
        self._hh_data = hh_data.copy()
        self._person_data = person_data.copy()

        # Validate schema
        self._validate_data()

        # Prepare person data with position features
        person_with_position = self._add_position_features(person_data, hh_data)

        # Pass 1: Fit household-level synthesizer
        if verbose:
            print("=" * 60)
            print("PASS 1: Fitting household-level model")
            print("=" * 60)
            print(f"  Variables: {self.schema.hh_vars}")
            print(f"  N households: {len(hh_data):,}")

        self.hh_synthesizer = Synthesizer(
            target_vars=self.schema.hh_vars,
            condition_vars=[],  # Unconditional for now
            **self.hh_flow_kwargs
        )
        self.hh_synthesizer.fit(
            hh_data,
            weight_col=hh_weight_col,
            epochs=epochs,
        )

        # Pass 2: Fit person-level synthesizer
        if verbose:
            print("\n" + "=" * 60)
            print("PASS 2: Fitting person-level model")
            print("=" * 60)
            print(f"  Target vars: {self.schema.person_vars}")
            print(f"  Condition vars: {self.schema.person_condition_vars}")
            print(f"  N persons: {len(person_data):,}")

        # Filter to available condition vars
        available_condition_vars = [
            v for v in self.schema.person_condition_vars
            if v in person_with_position.columns
        ]

        self.person_synthesizer = Synthesizer(
            target_vars=self.schema.person_vars,
            condition_vars=available_condition_vars,
            **self.person_flow_kwargs
        )
        self.person_synthesizer.fit(
            person_with_position,
            weight_col=person_weight_col,
            epochs=epochs,
        )

        self._is_fitted = True

        if verbose:
            print("\n" + "=" * 60)
            print("HIERARCHICAL MODEL FITTED")
            print("=" * 60)

        return self

    def generate(
        self,
        n_households: int,
        return_units: bool = False,
        verbose: bool = True,
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame],
               Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Generate synthetic households and persons.

        Args:
            n_households: Number of households to generate
            return_units: If True, also return tax_units and spm_units DataFrames
            verbose: Print progress

        Returns:
            (households, persons) DataFrames, or
            (households, persons, tax_units, spm_units) if return_units=True
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() before generate()")

        # Pass 1: Generate households
        if verbose:
            print(f"Generating {n_households:,} synthetic households...")

        # Generate household features
        # For unconditional generation, we need dummy conditioning data
        dummy_context = pd.DataFrame({'_dummy': np.zeros(n_households)})
        synthetic_hh = self.hh_synthesizer.generate(dummy_context)

        # Add household IDs
        synthetic_hh[self.schema.hh_id_col] = np.arange(n_households)

        # Ensure integer counts
        for col in ['n_persons', 'n_adults', 'n_children']:
            if col in synthetic_hh.columns:
                synthetic_hh[col] = np.clip(
                    np.round(synthetic_hh[col]).astype(int), 1, 20
                )

        # Assign geographic identifiers based on state
        # Block assignment only sets block_geoid - parent geographies (tract, county, CD, SLD)
        # should be derived post-hoc using BlockGeography or derive_geographies()
        if self._geography_assigner is not None:
            if verbose:
                print("Assigning geography...")
            synthetic_hh = self._apply_geography_assignment(synthetic_hh)
        elif self._block_lookup is not None:
            if verbose:
                print("Assigning census blocks...")
            synthetic_hh = self._assign_blocks(synthetic_hh)
            n_with_block = synthetic_hh['block_geoid'].notna().sum()
            if verbose:
                print(f"  Assigned blocks to {n_with_block:,} households ({n_with_block/n_households:.1%})")
                n_unique_blocks = synthetic_hh['block_geoid'].nunique()
                print(f"  Unique blocks: {n_unique_blocks:,}")
        elif self._cd_lookup is not None:
            if verbose:
                print("Assigning congressional districts...")
            synthetic_hh = self._assign_cds(synthetic_hh)
            n_with_cd = synthetic_hh['cd_id'].notna().sum()
            if verbose:
                print(f"  Assigned CDs to {n_with_cd:,} households ({n_with_cd/n_households:.1%})")

        # Pass 2: Generate persons for each household
        if verbose:
            print("Generating persons for each household...")

        person_records = []
        person_id = 0

        for hh_idx, hh_row in synthetic_hh.iterrows():
            n_persons = int(hh_row.get('n_persons', 1))
            n_adults = int(hh_row.get('n_adults', 1))

            # Create conditioning context for each person in this HH
            for p_num in range(n_persons):
                context = {
                    self.schema.hh_id_col: hh_row[self.schema.hh_id_col],
                    self.schema.person_id_col: person_id,
                    'person_number': p_num,
                    'is_first_adult': p_num == 0,
                    'is_child_slot': p_num >= n_adults,
                }
                # Add HH-level features to context
                for var in self.schema.hh_vars:
                    if var in hh_row.index:
                        value = hh_row[var]
                        if var == 'state_fips':
                            value = int(round(float(value)))
                        context[var] = value

                person_records.append(context)
                person_id += 1

        # Convert to DataFrame
        person_context = pd.DataFrame(person_records)

        if verbose:
            print(f"  Total persons: {len(person_context):,}")
            print(f"  Avg HH size: {len(person_context) / n_households:.2f}")

        # Generate person attributes
        synthetic_persons = self.person_synthesizer.generate(person_context)

        # Add IDs and context back
        synthetic_persons[self.schema.hh_id_col] = person_context[self.schema.hh_id_col].values
        synthetic_persons[self.schema.person_id_col] = person_context[self.schema.person_id_col].values

        # Derive aggregates
        if verbose:
            print("Deriving household aggregates...")
        synthetic_hh = self._derive_aggregates(synthetic_hh, synthetic_persons)

        if return_units:
            if verbose:
                print("Constructing tax units and SPM units...")
            tax_units = self._construct_tax_units(synthetic_hh, synthetic_persons)
            spm_units = self._construct_spm_units(synthetic_hh, synthetic_persons)
            return synthetic_hh, synthetic_persons, tax_units, spm_units

        return synthetic_hh, synthetic_persons

    def reweight(
        self,
        hh_data: pd.DataFrame,
        person_data: pd.DataFrame,
        targets: Dict[str, Dict],
        continuous_targets: Optional[Dict[str, float]] = None,
        **reweighter_kwargs,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Reweight households and persons to match population targets.

        Weights are computed at the household level and then propagated
        to all persons in each household.

        Args:
            hh_data: Household DataFrame (from generate())
            person_data: Person DataFrame (from generate())
            targets: Marginal targets {variable: {category: count}}
            continuous_targets: Optional continuous totals {variable: sum}
            **reweighter_kwargs: Additional kwargs for Calibrator

        Returns:
            (hh_weighted, persons_weighted) tuple with 'weight' column added
        """
        from .calibration import Calibrator

        hh_df = hh_data.copy()
        persons_df = person_data.copy()

        # Initialize calibrator with provided method/backend
        method = reweighter_kwargs.pop('method', 'ipf')
        calibrator = Calibrator(method=method, **reweighter_kwargs)

        # Fit calibration to household data
        calibrator.fit(
            hh_df,
            marginal_targets=targets,
            continuous_targets=continuous_targets,
        )

        # Add weights to households
        hh_df['weight'] = calibrator.weights_

        # Propagate household weights to persons
        weight_map = hh_df.set_index(self.schema.hh_id_col)['weight']
        persons_df['weight'] = persons_df[self.schema.hh_id_col].map(weight_map)

        return hh_df, persons_df

    def generate_and_reweight(
        self,
        n_households: int,
        targets: Dict[str, Dict],
        continuous_targets: Optional[Dict[str, float]] = None,
        return_units: bool = False,
        verbose: bool = True,
        **reweighter_kwargs,
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame],
               Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Generate synthetic data and reweight to match targets in one call.

        Convenience method that combines generate() and reweight().

        Args:
            n_households: Number of households to generate
            targets: Marginal targets {variable: {category: count}}
            continuous_targets: Optional continuous totals {variable: sum}
            return_units: If True, also return tax_units and spm_units
            verbose: Print progress
            **reweighter_kwargs: Additional kwargs for Calibrator

        Returns:
            (hh_weighted, persons_weighted) or
            (hh_weighted, persons_weighted, tax_units, spm_units)
        """
        # Generate synthetic data
        if return_units:
            hh, persons, tax_units, spm_units = self.generate(
                n_households=n_households,
                return_units=True,
                verbose=verbose,
            )
        else:
            hh, persons = self.generate(
                n_households=n_households,
                return_units=False,
                verbose=verbose,
            )

        # Reweight to match targets
        hh_weighted, persons_weighted = self.reweight(
            hh, persons,
            targets=targets,
            continuous_targets=continuous_targets,
            **reweighter_kwargs,
        )

        if return_units:
            return hh_weighted, persons_weighted, tax_units, spm_units
        return hh_weighted, persons_weighted

    def _validate_data(self) -> None:
        """Validate that data has required columns."""
        hh_missing = set(self.schema.hh_vars) - set(self._hh_data.columns)
        if hh_missing:
            raise ValueError(f"Household data missing columns: {hh_missing}")

        person_missing = set(self.schema.person_vars) - set(self._person_data.columns)
        if person_missing:
            raise ValueError(f"Person data missing columns: {person_missing}")

        if self.schema.hh_id_col not in self._person_data.columns:
            raise ValueError(
                f"Person data must have household ID column: {self.schema.hh_id_col}"
            )

    def _add_position_features(
        self,
        person_data: pd.DataFrame,
        hh_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add position-within-household features to person data."""
        df = person_data.copy()

        # Person number within household
        df['person_number'] = df.groupby(self.schema.hh_id_col).cumcount()

        # Merge HH features
        hh_features = hh_data[[self.schema.hh_id_col] + [
            v for v in self.schema.hh_vars if v in hh_data.columns
        ]].copy()

        if self.schema.hh_id_col in hh_features.columns:
            df = df.merge(hh_features, on=self.schema.hh_id_col, how='left')

        # Compute position features
        n_adults = df.get('n_adults', 1)
        df['is_first_adult'] = df['person_number'] == 0
        df['is_child_slot'] = df['person_number'] >= n_adults

        return df

    def _derive_aggregates(
        self,
        hh_data: pd.DataFrame,
        person_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Derive household aggregates from person data."""
        hh = hh_data.copy()

        for derived_name, formula in self.schema.derived_vars.items():
            if ':' in formula:
                agg_type, var_expr = formula.split(':', 1)

                if agg_type == 'sum':
                    if var_expr in person_data.columns:
                        agg = person_data.groupby(self.schema.hh_id_col)[var_expr].sum()
                        hh[derived_name] = hh[self.schema.hh_id_col].map(agg).fillna(0)

                elif agg_type == 'count':
                    if '==' in var_expr:
                        var, val = var_expr.split('==')
                        mask = person_data[var.strip()] == int(val)
                        counts = person_data[mask].groupby(self.schema.hh_id_col).size()
                        hh[derived_name] = hh[self.schema.hh_id_col].map(counts).fillna(0)

        return hh

    def _construct_tax_units(
        self,
        hh_data: pd.DataFrame,
        person_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Construct tax units from persons.

        Simple heuristic for now:
        - Each married couple is a tax unit
        - Each single adult is a tax unit
        - Children assigned to parent tax units

        TODO: Optimize for minimum tax liability
        """
        tax_units = []
        tu_id = 0

        for hh_id in hh_data[self.schema.hh_id_col].unique():
            hh_persons = person_data[
                person_data[self.schema.hh_id_col] == hh_id
            ].copy()

            # Simple: first person is head, spouse if exists, rest are dependents
            # This is a placeholder - real logic would be more sophisticated
            n_persons = len(hh_persons)

            if n_persons == 0:
                continue

            # For now, one tax unit per household (simplified)
            tax_units.append({
                'tax_unit_id': tu_id,
                self.schema.hh_id_col: hh_id,
                'n_members': n_persons,
                'filing_status': 'married_joint' if n_persons >= 2 else 'single',
            })
            tu_id += 1

        return pd.DataFrame(tax_units)

    def _construct_spm_units(
        self,
        hh_data: pd.DataFrame,
        person_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Construct SPM (Supplemental Poverty Measure) units from persons.

        SPM unit rules (simplified):
        - All related individuals in household
        - Includes cohabiting partners and their relatives
        - Foster children included with foster families

        For now: SPM unit = household (simplified)
        """
        spm_units = []

        for idx, hh_row in hh_data.iterrows():
            hh_id = hh_row[self.schema.hh_id_col]
            hh_persons = person_data[
                person_data[self.schema.hh_id_col] == hh_id
            ]

            spm_units.append({
                'spm_unit_id': idx,
                self.schema.hh_id_col: hh_id,
                'n_members': len(hh_persons),
            })

        return pd.DataFrame(spm_units)


class TaxUnitOptimizer:
    """
    Optimizer for constructing tax units within households.

    Determines optimal filing status and dependent assignments
    to minimize overall tax liability for the household.

    Uses 2024 tax parameters.
    """

    # 2024 Standard Deductions
    STANDARD_DEDUCTIONS = {
        'single': 14600,
        'married_filing_jointly': 29200,
        'married_filing_separately': 14600,
        'head_of_household': 21900,
    }

    # 2024 Tax Brackets (single filer)
    TAX_BRACKETS_SINGLE = [
        (11600, 0.10),
        (47150, 0.12),
        (100525, 0.22),
        (191950, 0.24),
        (243725, 0.32),
        (609350, 0.35),
        (float('inf'), 0.37),
    ]

    # 2024 Tax Brackets (married filing jointly)
    TAX_BRACKETS_MFJ = [
        (23200, 0.10),
        (94300, 0.12),
        (201050, 0.22),
        (383900, 0.24),
        (487450, 0.32),
        (731200, 0.35),
        (float('inf'), 0.37),
    ]

    # 2024 Tax Brackets (head of household)
    TAX_BRACKETS_HOH = [
        (16550, 0.10),
        (63100, 0.12),
        (100500, 0.22),
        (191950, 0.24),
        (243700, 0.32),
        (609350, 0.35),
        (float('inf'), 0.37),
    ]

    # 2024 EITC Parameters
    EITC_PARAMS = {
        # (max_credit, phase_in_rate, phase_in_end, phase_out_start_single, phase_out_start_mfj, phase_out_rate, phase_out_end_single, phase_out_end_mfj)
        0: (632, 0.0765, 8260, 9800, 16370, 0.0765, 18591, 25511),
        1: (4213, 0.34, 12390, 22720, 29640, 0.1598, 49084, 56004),
        2: (6960, 0.40, 17400, 22720, 29640, 0.2106, 55768, 62688),
        3: (7830, 0.45, 17400, 22720, 29640, 0.2106, 59899, 66819),
    }

    # 2024 CTC Parameters
    CTC_PER_CHILD = 2000
    CTC_PHASE_OUT_START_SINGLE = 200000
    CTC_PHASE_OUT_START_MFJ = 400000
    CTC_PHASE_OUT_RATE = 0.05  # $50 per $1000 over threshold

    def __init__(self):
        """Initialize the TaxUnitOptimizer."""
        pass

    def optimize_household(
        self,
        hh_id: int,
        persons_df: pd.DataFrame,
    ) -> List[Dict]:
        """
        Construct optimal tax units for a household.

        Args:
            hh_id: Household ID
            persons_df: DataFrame with person data for this household

        Returns:
            List of tax unit dictionaries
        """
        # Filter to this household's persons
        hh_persons = persons_df[persons_df['household_id'] == hh_id].copy()

        if len(hh_persons) == 0:
            return []

        # Identify adults vs potential dependents
        head = hh_persons[hh_persons['relationship_to_head'] == 0]
        spouse = hh_persons[hh_persons['relationship_to_head'] == 1]
        children = hh_persons[hh_persons['relationship_to_head'] == 2]
        others = hh_persons[hh_persons['relationship_to_head'] == 3]

        tax_units = []
        tu_id = 0

        # Determine qualifying dependents among children
        qualifying_dependents = self._get_qualifying_dependents(children)
        n_qualifying_children = len([d for d in qualifying_dependents if d['age'] < 17])

        # Case 1: Married couple
        if len(head) > 0 and len(spouse) > 0:
            head_row = head.iloc[0]
            spouse_row = spouse.iloc[0]
            combined_income = head_row['income'] + spouse_row['income']

            # Calculate MFJ liability
            mfj_liability = self._calculate_tax_liability(
                combined_income,
                'married_filing_jointly',
                len(qualifying_dependents)
            )

            # Calculate MFS liability (each spouse files separately)
            mfs_head_liability = self._calculate_tax_liability(
                head_row['income'],
                'married_filing_separately',
                0  # No dependents for MFS comparison (simplified)
            )
            mfs_spouse_liability = self._calculate_tax_liability(
                spouse_row['income'],
                'married_filing_separately',
                0
            )
            mfs_total = mfs_head_liability + mfs_spouse_liability

            # Choose optimal filing status
            if mfs_total < mfj_liability:
                # File separately
                tax_units.append({
                    'tax_unit_id': tu_id,
                    'household_id': hh_id,
                    'filing_status': 'married_filing_separately',
                    'filer_ids': [int(head_row['person_id'])],
                    'dependent_ids': [],
                    'n_dependents': 0,
                    'total_income': head_row['income'],
                    'tax_liability': mfs_head_liability,
                })
                tu_id += 1
                tax_units.append({
                    'tax_unit_id': tu_id,
                    'household_id': hh_id,
                    'filing_status': 'married_filing_separately',
                    'filer_ids': [int(spouse_row['person_id'])],
                    'dependent_ids': [],
                    'n_dependents': 0,
                    'total_income': spouse_row['income'],
                    'tax_liability': mfs_spouse_liability,
                })
                tu_id += 1
            else:
                # File jointly
                tax_units.append({
                    'tax_unit_id': tu_id,
                    'household_id': hh_id,
                    'filing_status': 'married_filing_jointly',
                    'filer_ids': [int(head_row['person_id']), int(spouse_row['person_id'])],
                    'dependent_ids': [int(d['person_id']) for d in qualifying_dependents],
                    'n_dependents': len(qualifying_dependents),
                    'total_income': combined_income,
                    'tax_liability': mfj_liability,
                })
                tu_id += 1

        # Case 2: Single head with dependents (Head of Household)
        elif len(head) > 0 and len(qualifying_dependents) > 0:
            head_row = head.iloc[0]
            income = head_row['income']

            liability = self._calculate_tax_liability(
                income,
                'head_of_household',
                len(qualifying_dependents)
            )

            tax_units.append({
                'tax_unit_id': tu_id,
                'household_id': hh_id,
                'filing_status': 'head_of_household',
                'filer_ids': [int(head_row['person_id'])],
                'dependent_ids': [int(d['person_id']) for d in qualifying_dependents],
                'n_dependents': len(qualifying_dependents),
                'total_income': income,
                'tax_liability': liability,
            })
            tu_id += 1

        # Case 3: Single head without dependents
        elif len(head) > 0:
            head_row = head.iloc[0]
            income = head_row['income']

            liability = self._calculate_tax_liability(income, 'single', 0)

            tax_units.append({
                'tax_unit_id': tu_id,
                'household_id': hh_id,
                'filing_status': 'single',
                'filer_ids': [int(head_row['person_id'])],
                'dependent_ids': [],
                'n_dependents': 0,
                'total_income': income,
                'tax_liability': liability,
            })
            tu_id += 1

        # Handle unrelated adults - each files separately
        for _, other_row in others.iterrows():
            income = other_row['income']
            liability = self._calculate_tax_liability(income, 'single', 0)

            tax_units.append({
                'tax_unit_id': tu_id,
                'household_id': hh_id,
                'filing_status': 'single',
                'filer_ids': [int(other_row['person_id'])],
                'dependent_ids': [],
                'n_dependents': 0,
                'total_income': income,
                'tax_liability': liability,
            })
            tu_id += 1

        return tax_units

    def _get_qualifying_dependents(
        self,
        children: pd.DataFrame
    ) -> List[Dict]:
        """
        Determine which children qualify as dependents.

        Rules:
        - Children under 19
        - Full-time students under 24
        - Permanently disabled adults

        Args:
            children: DataFrame of child persons

        Returns:
            List of qualifying dependent dicts
        """
        dependents = []

        for _, child in children.iterrows():
            age = child['age']
            is_student = child.get('is_student', False)
            is_disabled = child.get('is_disabled', False)

            # Qualifying child test
            if age < 19:
                dependents.append(child.to_dict())
            elif age < 24 and is_student:
                dependents.append(child.to_dict())
            elif is_disabled:
                dependents.append(child.to_dict())

        return dependents

    def _standard_deduction(
        self,
        filing_status: str,
        n_dependents: int
    ) -> float:
        """
        Get standard deduction for filing status.

        Args:
            filing_status: Filing status string
            n_dependents: Number of dependents (not currently used)

        Returns:
            Standard deduction amount
        """
        return self.STANDARD_DEDUCTIONS.get(filing_status, 14600)

    def _get_tax_brackets(self, filing_status: str) -> List[Tuple[float, float]]:
        """Get tax brackets for filing status."""
        if filing_status == 'married_filing_jointly':
            return self.TAX_BRACKETS_MFJ
        elif filing_status == 'head_of_household':
            return self.TAX_BRACKETS_HOH
        else:
            # single and married_filing_separately use same brackets
            return self.TAX_BRACKETS_SINGLE

    def _calculate_bracket_tax(
        self,
        taxable_income: float,
        filing_status: str
    ) -> float:
        """
        Calculate tax from brackets.

        Args:
            taxable_income: Income after deductions
            filing_status: Filing status string

        Returns:
            Tax amount before credits
        """
        if taxable_income <= 0:
            return 0

        brackets = self._get_tax_brackets(filing_status)
        tax = 0
        prev_threshold = 0

        for threshold, rate in brackets:
            if taxable_income <= threshold:
                tax += (taxable_income - prev_threshold) * rate
                break
            else:
                tax += (threshold - prev_threshold) * rate
                prev_threshold = threshold

        return tax

    def _calculate_eitc(
        self,
        income: float,
        filing_status: str,
        n_children: int
    ) -> float:
        """
        Calculate Earned Income Tax Credit.

        Args:
            income: Earned income
            filing_status: Filing status string
            n_children: Number of qualifying children

        Returns:
            EITC amount
        """
        # Cap children at 3 for EITC purposes
        n_children = min(n_children, 3)

        params = self.EITC_PARAMS[n_children]
        max_credit, phase_in_rate, phase_in_end, phase_out_start_single, phase_out_start_mfj, phase_out_rate, phase_out_end_single, phase_out_end_mfj = params

        # Select phase-out thresholds based on filing status
        if filing_status == 'married_filing_jointly':
            phase_out_start = phase_out_start_mfj
            phase_out_end = phase_out_end_mfj
        else:
            phase_out_start = phase_out_start_single
            phase_out_end = phase_out_end_single

        # Phase-in: Credit increases as income rises up to phase_in_end
        if income <= phase_in_end:
            credit = income * phase_in_rate
        # Plateau: Maximum credit between phase_in_end and phase_out_start
        elif income <= phase_out_start:
            credit = max_credit
        # Phase-out: Credit decreases as income rises above phase_out_start
        elif income < phase_out_end:
            credit = max_credit - (income - phase_out_start) * phase_out_rate
        else:
            credit = 0

        # Ensure credit doesn't exceed maximum
        credit = min(credit, max_credit)

        return max(0, credit)

    def _calculate_ctc(
        self,
        income: float,
        filing_status: str,
        n_children: int
    ) -> float:
        """
        Calculate Child Tax Credit.

        Args:
            income: Adjusted gross income
            filing_status: Filing status string
            n_children: Number of qualifying children under 17

        Returns:
            CTC amount
        """
        if n_children == 0:
            return 0

        # Base credit
        credit = n_children * self.CTC_PER_CHILD

        # Phase-out threshold
        if filing_status == 'married_filing_jointly':
            threshold = self.CTC_PHASE_OUT_START_MFJ
        else:
            threshold = self.CTC_PHASE_OUT_START_SINGLE

        # Phase-out: $50 reduction per $1000 over threshold
        if income > threshold:
            excess = income - threshold
            # Round up to nearest $1000
            reduction_units = int((excess + 999) / 1000)
            reduction = reduction_units * 50
            credit = max(0, credit - reduction)

        return credit

    def _calculate_tax_liability(
        self,
        income: float,
        filing_status: str,
        n_dependents: int
    ) -> float:
        """
        Calculate overall tax liability after credits.

        Args:
            income: Total income
            filing_status: Filing status string
            n_dependents: Number of dependents

        Returns:
            Net tax liability (negative if refund)
        """
        # Standard deduction
        std_ded = self._standard_deduction(filing_status, n_dependents)
        taxable_income = max(0, income - std_ded)

        # Calculate bracket tax
        bracket_tax = self._calculate_bracket_tax(taxable_income, filing_status)

        # Count qualifying children for credits (under 17 for CTC)
        # For simplicity, assume all dependents are qualifying children
        n_children = n_dependents

        # Calculate credits
        eitc = self._calculate_eitc(income, filing_status, n_children)
        ctc = self._calculate_ctc(income, filing_status, n_children)

        # EITC is fully refundable, CTC partially refundable (up to $1700 per child in 2024)
        # For simplicity, treat CTC as refundable up to $1700 per child
        refundable_ctc = min(ctc, n_children * 1700)
        non_refundable_ctc = ctc - refundable_ctc

        # Apply non-refundable credits (can't go below zero)
        tax_after_non_refundable = max(0, bracket_tax - non_refundable_ctc)

        # Apply refundable credits
        net_tax = tax_after_non_refundable - eitc - refundable_ctc

        return net_tax


def prepare_cps_for_hierarchical(
    cps_person_data: pd.DataFrame,
    hh_id_col: str = 'household_id',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare CPS data for hierarchical synthesis.

    Takes person-level CPS data and creates:
    1. Household-level summary (one row per HH)
    2. Person-level data with position features

    Args:
        cps_person_data: CPS person-level data
        hh_id_col: Household ID column

    Returns:
        (hh_data, person_data) tuple
    """
    df = cps_person_data.copy()

    # Create household-level summary
    hh_agg = df.groupby(hh_id_col).agg({
        'age': ['count', lambda x: (x >= 18).sum(), lambda x: (x < 18).sum()],
    })
    hh_agg.columns = ['n_persons', 'n_adults', 'n_children']
    hh_agg = hh_agg.reset_index()

    # Add other HH-level vars (take first value per HH)
    hh_level_vars = ['state_fips', 'tenure', 'hh_weight']
    for var in hh_level_vars:
        if var in df.columns:
            first_vals = df.groupby(hh_id_col)[var].first()
            hh_agg[var] = hh_agg[hh_id_col].map(first_vals)

    return hh_agg, df
