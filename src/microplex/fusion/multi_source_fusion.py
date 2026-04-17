"""Multi-source fusion pipeline for synthetic microdata generation.

Combines multiple survey sources (e.g., SIPP, CPS, SCF) into a unified
synthetic dataset by:
1. Training survey-specific synthesizers (preserves marginal distributions)
2. Training cross-survey imputers (fills missing variables)
3. Training a unified model on stacked imputed data (captures joint patterns)
4. Stacking synthetics from all sources for maximum coverage

Example:
    >>> from microplex.fusion import MultiSourceFusion
    >>>
    >>> fusion = MultiSourceFusion(
    ...     shared_vars=['age', 'total_income'],
    ...     all_vars=['age', 'total_income', 'job1_income', 'job2_income'],
    ... )
    >>>
    >>> fusion.add_source('sipp', sipp_df, source_vars=['age', 'total_income', 'job1_income', 'job2_income'])
    >>> fusion.add_source('cps', cps_df, source_vars=['age', 'total_income'])
    >>>
    >>> fusion.fit(epochs=100)
    >>>
    >>> synthetic = fusion.generate(n_per_source=10000)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ..synthesizer import Synthesizer


@dataclass
class SourceConfig:
    """Configuration for a single data source."""
    name: str
    data: pd.DataFrame
    source_vars: list[str]
    n_periods: int = 1
    person_id_col: str = 'person_id'
    period_col: str = 'period'
    weight_col: str | None = None


@dataclass
class FusionConfig:
    """Configuration for the fusion pipeline."""
    shared_vars: list[str]
    all_vars: list[str]
    n_periods: int = 6
    imputer_epochs: int = 100
    synthesizer_epochs: int = 100
    unified_epochs: int = 100
    noise_scale: float = 0.1  # For bootstrap+noise generation
    random_state: int = 42


class MultiSourceFusion:
    """
    Multi-source fusion pipeline for synthetic panel data generation.

    Combines multiple survey sources into a unified synthetic dataset
    with maximum coverage across all source distributions.

    Attributes:
        config: FusionConfig with pipeline settings
        sources: Dict of SourceConfig for each added source
        imputers: Dict of trained imputers (source_name -> Synthesizer)
        synthesizers: Dict of trained synthesizers per source
        unified_synthesizer: Synthesizer trained on stacked imputed data
    """

    def __init__(
        self,
        shared_vars: list[str],
        all_vars: list[str],
        n_periods: int = 6,
        imputer_epochs: int = 100,
        synthesizer_epochs: int = 100,
        unified_epochs: int = 100,
        noise_scale: float = 0.1,
        random_state: int = 42,
    ):
        """
        Initialize fusion pipeline.

        Args:
            shared_vars: Variables present in all sources
            all_vars: Complete set of variables for output
            n_periods: Number of time periods for panel synthesis
            imputer_epochs: Training epochs for imputers
            synthesizer_epochs: Training epochs for source synthesizers
            unified_epochs: Training epochs for unified model
            noise_scale: Scale of noise added to bootstrap samples (0.1 = 10% of std)
            random_state: Random seed for reproducibility
        """
        self.config = FusionConfig(
            shared_vars=shared_vars,
            all_vars=all_vars,
            n_periods=n_periods,
            imputer_epochs=imputer_epochs,
            synthesizer_epochs=synthesizer_epochs,
            unified_epochs=unified_epochs,
            noise_scale=noise_scale,
            random_state=random_state,
        )

        self.sources: dict[str, SourceConfig] = {}
        self.imputers: dict[str, Synthesizer] = {}
        self.source_data: dict[str, pd.DataFrame] = {}
        self._is_fitted = False

        # These will be set during fit
        self._unified_synthesizer: Synthesizer | None = None
        self._reference_source: str | None = None  # Source with most vars

    def add_source(
        self,
        name: str,
        data: pd.DataFrame,
        source_vars: list[str],
        n_periods: int = 1,
        person_id_col: str = 'person_id',
        period_col: str = 'period',
        weight_col: str | None = None,
    ) -> Self:
        """
        Add a data source to the fusion pipeline.

        Args:
            name: Unique identifier for this source (e.g., 'sipp', 'cps')
            data: DataFrame with source data
            source_vars: Variables available in this source
            n_periods: Number of periods per person (1 for cross-sectional)
            person_id_col: Column name for person identifier
            period_col: Column name for time period
            weight_col: Column name for sample weights (optional)

        Returns:
            self for method chaining
        """
        # Validate
        missing = set(source_vars) - set(data.columns)
        if missing:
            raise ValueError(f"Source '{name}' missing columns: {missing}")

        shared_check = set(self.config.shared_vars) - set(source_vars)
        if shared_check:
            raise ValueError(f"Source '{name}' must have shared vars: {shared_check}")

        self.sources[name] = SourceConfig(
            name=name,
            data=data,
            source_vars=source_vars,
            n_periods=n_periods,
            person_id_col=person_id_col,
            period_col=period_col,
            weight_col=weight_col,
        )

        # Store processed data
        self.source_data[name] = data.copy()

        return self

    def fit(
        self,
        verbose: bool = True,
    ) -> Self:
        """
        Fit all components of the fusion pipeline.

        1. Identifies reference source (has most variables)
        2. Trains imputers from shared vars to source-specific vars
        3. Imputes missing vars onto each source
        4. Trains unified model on stacked imputed data

        Args:
            verbose: Whether to print progress

        Returns:
            self
        """
        if len(self.sources) < 2:
            raise ValueError("Need at least 2 sources for fusion")

        np.random.seed(self.config.random_state)
        torch.manual_seed(self.config.random_state)

        # Find reference source (most variables)
        self._reference_source = max(
            self.sources.keys(),
            key=lambda s: len(self.sources[s].source_vars)
        )
        ref_config = self.sources[self._reference_source]

        if verbose:
            print(f"Reference source: {self._reference_source} ({len(ref_config.source_vars)} vars)")

        # Get source-specific vars (vars in reference but not in shared)
        source_specific_vars = [
            v for v in ref_config.source_vars
            if v not in self.config.shared_vars
        ]

        if verbose:
            print(f"Shared vars: {self.config.shared_vars}")
            print(f"Source-specific vars: {source_specific_vars}")

        # Train imputers for each non-reference source
        if source_specific_vars:
            self._fit_imputers(source_specific_vars, verbose)

        # Create unified dataset
        unified_data = self._create_unified_dataset(source_specific_vars, verbose)

        # Train unified model
        if verbose:
            print(f"\nTraining unified model on {unified_data['person_id'].nunique()} persons...")

        self._unified_synthesizer = Synthesizer(
            target_vars=self.config.all_vars,
            condition_vars=[],
            zero_inflated=True,
            n_layers=6,
        )
        self._unified_synthesizer.fit(
            unified_data,
            epochs=self.config.unified_epochs,
            verbose=verbose,
        )

        self._is_fitted = True
        return self

    def _fit_imputers(
        self,
        source_specific_vars: list[str],
        verbose: bool,
    ) -> None:
        """Train imputers to predict source-specific vars from shared vars."""
        ref_config = self.sources[self._reference_source]
        ref_data = self.source_data[self._reference_source]

        # Get first period data for training imputer
        if ref_config.n_periods > 1:
            min_periods = ref_data.groupby(ref_config.person_id_col)[ref_config.period_col].min()
            min_periods = min_periods.reset_index()
            min_periods.columns = [ref_config.person_id_col, '_min_period']
            ref_merged = ref_data.merge(min_periods, on=ref_config.person_id_col)
            train_data = ref_merged[ref_merged[ref_config.period_col] == ref_merged['_min_period']]
        else:
            train_data = ref_data

        if verbose:
            print(f"\nTraining imputer: {self.config.shared_vars} → {source_specific_vars}")
            print(f"  Training on {len(train_data)} records from {self._reference_source}")

        imputer = Synthesizer(
            target_vars=source_specific_vars,
            condition_vars=self.config.shared_vars,
            zero_inflated=True,
            n_layers=4,
        )
        imputer.fit(
            train_data,
            epochs=self.config.imputer_epochs,
            verbose=verbose,
        )

        # Store imputer for all non-reference sources
        for source_name in self.sources:
            if source_name != self._reference_source:
                self.imputers[source_name] = imputer

    def _create_unified_dataset(
        self,
        source_specific_vars: list[str],
        verbose: bool,
    ) -> pd.DataFrame:
        """Create unified dataset by imputing missing vars and stacking."""
        unified_records = []
        person_id_offset = 0

        for source_name, config in self.sources.items():
            data = self.source_data[source_name].copy()

            if source_name == self._reference_source:
                # Reference source already has all vars
                if verbose:
                    print(f"\n{source_name}: {data[config.person_id_col].nunique()} persons (complete)")

                # Remap person_ids
                person_map = {p: i + person_id_offset for i, p in enumerate(data[config.person_id_col].unique())}
                data['_unified_pid'] = data[config.person_id_col].map(person_map)
                person_id_offset = max(person_map.values()) + 1

                for col in self.config.all_vars:
                    if col not in data.columns:
                        data[col] = 0

                unified_records.append(data)

            else:
                # Need to impute source-specific vars
                if verbose:
                    print(f"\n{source_name}: {data[config.person_id_col].nunique()} persons (imputing...)")

                imputer = self.imputers[source_name]

                # Impute for first period, then replicate
                if config.n_periods == 1:
                    # Cross-sectional: impute and expand to n_periods
                    imputed = imputer.generate(data[self.config.shared_vars], seed=self.config.random_state)

                    expanded_records = []
                    for idx, (_, row) in enumerate(data.iterrows()):
                        for t in range(self.config.n_periods):
                            rec = {
                                '_unified_pid': idx + person_id_offset,
                                config.period_col: t,
                            }
                            for var in self.config.shared_vars:
                                rec[var] = row[var]
                            for var in source_specific_vars:
                                rec[var] = float(np.clip(imputed.iloc[idx][var], 0, 1e10))
                            expanded_records.append(rec)

                    person_id_offset += len(data)
                    unified_records.append(pd.DataFrame(expanded_records))

                else:
                    # Panel: impute per period
                    imputed_data = data.copy()
                    imputed_vals = imputer.generate(data[self.config.shared_vars], seed=self.config.random_state)

                    for var in source_specific_vars:
                        imputed_data[var] = np.clip(imputed_vals[var].values, 0, 1e10)

                    person_map = {p: i + person_id_offset for i, p in enumerate(imputed_data[config.person_id_col].unique())}
                    imputed_data['_unified_pid'] = imputed_data[config.person_id_col].map(person_map)
                    person_id_offset = max(person_map.values()) + 1

                    unified_records.append(imputed_data)

        # Combine
        unified = pd.concat(unified_records, ignore_index=True)
        unified['person_id'] = unified['_unified_pid']

        # Ensure all vars present
        for var in self.config.all_vars:
            if var not in unified.columns:
                unified[var] = 0

        return unified[['person_id', 'period'] + self.config.all_vars]

    def generate(
        self,
        n_per_source: int = 10000,
        include_unified: bool = True,
        seed: int = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic data from all sources.

        Args:
            n_per_source: Number of synthetic persons per source
            include_unified: Whether to include synthetics from unified model
            seed: Random seed (uses config seed if None)

        Returns:
            DataFrame with synthetic panel data
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() before generate()")

        seed = seed or self.config.random_state
        np.random.seed(seed)

        all_synthetics = []
        person_id_offset = 0

        # Get source-specific vars
        ref_vars = self.sources[self._reference_source].source_vars
        source_specific_vars = [v for v in ref_vars if v not in self.config.shared_vars]

        # Generate from each source
        for source_name, config in self.sources.items():
            source_seed = seed + hash(source_name) % 10000

            if source_name == self._reference_source:
                # Bootstrap + noise from reference source
                synth_df = self._generate_bootstrap_noise(
                    source_name, n_per_source, source_seed
                )
            else:
                # Bootstrap + noise on shared vars, then impute
                synth_df = self._generate_with_imputation(
                    source_name, n_per_source, source_specific_vars, source_seed
                )

            synth_df['person_id'] = synth_df['person_id'] + person_id_offset
            synth_df['source'] = source_name
            person_id_offset = synth_df['person_id'].max() + 1

            all_synthetics.append(synth_df)

        # Generate from unified model
        if include_unified and self._unified_synthesizer is not None:
            unified_synth = self._unified_synthesizer.generate(
                pd.DataFrame({'_dummy': range(n_per_source)}),
                seed=seed + 999,
            )

            # Expand to panel if needed
            unified_records = []
            for pid in range(n_per_source):
                for t in range(self.config.n_periods):
                    rec = {
                        'person_id': pid + person_id_offset,
                        'period': t,
                        'source': 'unified',
                    }
                    for var in self.config.all_vars:
                        rec[var] = float(np.clip(unified_synth.iloc[pid][var], 0, 1e10))
                    unified_records.append(rec)

            all_synthetics.append(pd.DataFrame(unified_records))

        result = pd.concat(all_synthetics, ignore_index=True)
        return result[['person_id', 'period', 'source'] + self.config.all_vars]

    def _generate_bootstrap_noise(
        self,
        source_name: str,
        n_synth: int,
        seed: int,
    ) -> pd.DataFrame:
        """Generate via bootstrap + noise from source data."""
        config = self.sources[source_name]
        data = self.source_data[source_name]

        np.random.seed(seed)

        # Get unique persons
        persons = data[config.person_id_col].unique()

        # Bootstrap sample
        sampled_persons = np.random.choice(persons, n_synth, replace=True)

        records = []
        for pid, orig_pid in enumerate(sampled_persons):
            person_data = data[data[config.person_id_col] == orig_pid].sort_values(config.period_col)

            for t in range(min(self.config.n_periods, len(person_data))):
                row = person_data.iloc[t]
                rec = {'person_id': pid, 'period': t}

                for var in self.config.all_vars:
                    if var in config.source_vars:
                        # Add small noise
                        val = row[var]
                        if var != 'age':  # Don't add noise to age
                            noise = np.random.randn() * abs(val) * self.config.noise_scale
                            val = max(0, val + noise)
                        rec[var] = val
                    else:
                        rec[var] = 0

                records.append(rec)

        return pd.DataFrame(records)

    def _generate_with_imputation(
        self,
        source_name: str,
        n_synth: int,
        source_specific_vars: list[str],
        seed: int,
    ) -> pd.DataFrame:
        """Generate via bootstrap + noise on shared vars, then impute."""
        self.sources[source_name]
        data = self.source_data[source_name]
        imputer = self.imputers[source_name]

        np.random.seed(seed)

        # Get shared var values
        shared_vals = data[self.config.shared_vars].values
        std = shared_vals.std(axis=0)

        # Bootstrap + noise
        indices = np.random.choice(len(shared_vals), n_synth, replace=True)
        sampled = shared_vals[indices].copy()
        noise = np.random.randn(n_synth, len(self.config.shared_vars)) * std * self.config.noise_scale
        sampled = np.clip(sampled + noise, 0, None)

        # Clamp age if present
        if 'age' in self.config.shared_vars:
            age_idx = self.config.shared_vars.index('age')
            sampled[:, age_idx] = np.clip(sampled[:, age_idx], 0, 120)

        records = []
        for pid in range(n_synth):
            # Impute source-specific vars
            impute_df = pd.DataFrame([dict(zip(self.config.shared_vars, sampled[pid]))])
            imputed = imputer.generate(impute_df, seed=seed + pid)

            for t in range(self.config.n_periods):
                rec = {'person_id': pid, 'period': t}

                for i, var in enumerate(self.config.shared_vars):
                    rec[var] = sampled[pid, i]

                for var in source_specific_vars:
                    rec[var] = float(np.clip(imputed.iloc[0][var], 0, 1e10))

                records.append(rec)

        return pd.DataFrame(records)

    def evaluate_coverage(
        self,
        holdout_data: dict[str, pd.DataFrame],
        synthetic: pd.DataFrame = None,
        n_synth: int = 10000,
    ) -> pd.DataFrame:
        """
        Evaluate coverage on holdout data from each source.

        Args:
            holdout_data: Dict mapping source_name to holdout DataFrame
            synthetic: Pre-generated synthetic data (generates if None)
            n_synth: Number to generate per source if synthetic is None

        Returns:
            DataFrame with coverage metrics per source
        """
        if synthetic is None:
            synthetic = self.generate(n_per_source=n_synth)

        results = []

        for source_name, holdout in holdout_data.items():
            if source_name not in self.sources:
                warnings.warn(f"Unknown source '{source_name}', skipping")
                continue

            config = self.sources[source_name]

            # Determine which vars to evaluate on
            eval_vars = [v for v in config.source_vars if v in self.config.all_vars]

            # Compute coverage
            coverage = self._compute_coverage(
                holdout, synthetic, eval_vars,
                config.person_id_col, config.period_col
            )

            results.append({
                'source': source_name,
                'holdout_size': holdout[config.person_id_col].nunique(),
                'eval_vars': len(eval_vars),
                'coverage': coverage,
            })

        return pd.DataFrame(results)

    def _compute_coverage(
        self,
        holdout: pd.DataFrame,
        synthetic: pd.DataFrame,
        feature_cols: list[str],
        person_id_col: str,
        period_col: str,
    ) -> float:
        """Compute mean nearest-neighbor distance from holdout to synthetic."""

        def to_matrix(df, pid_col, period_col):
            rows = []
            for pid in sorted(df[pid_col].unique()):
                person_data = df[df[pid_col] == pid].sort_values(period_col)
                if len(person_data) >= self.config.n_periods:
                    vals = person_data[feature_cols].values[:self.config.n_periods]
                    rows.append(vals.flatten())
            return np.array(rows) if rows else np.zeros((0, len(feature_cols) * self.config.n_periods))

        holdout_mat = to_matrix(holdout, person_id_col, period_col)
        synth_mat = to_matrix(synthetic, 'person_id', 'period')

        if len(holdout_mat) == 0 or len(synth_mat) == 0:
            return float('inf')

        scaler = StandardScaler().fit(synth_mat)
        holdout_scaled = scaler.transform(holdout_mat)
        synth_scaled = scaler.transform(synth_mat)

        nn = NearestNeighbors(n_neighbors=1).fit(synth_scaled)
        distances, _ = nn.kneighbors(holdout_scaled)

        return float(np.mean(distances))

    def save(self, path: Path) -> None:
        """Save fitted pipeline to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        import json
        config_dict = {
            'shared_vars': self.config.shared_vars,
            'all_vars': self.config.all_vars,
            'n_periods': self.config.n_periods,
            'noise_scale': self.config.noise_scale,
            'random_state': self.config.random_state,
            'reference_source': self._reference_source,
            'sources': {
                name: {
                    'source_vars': cfg.source_vars,
                    'n_periods': cfg.n_periods,
                }
                for name, cfg in self.sources.items()
            }
        }
        with open(path / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)

        # Save imputers
        for name, imputer in self.imputers.items():
            imputer.save(path / f'imputer_{name}.pt')

        # Save unified synthesizer
        if self._unified_synthesizer is not None:
            self._unified_synthesizer.save(path / 'unified_synthesizer.pt')

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load fitted pipeline from directory."""
        path = Path(path)

        import json
        with open(path / 'config.json') as f:
            config_dict = json.load(f)

        fusion = cls(
            shared_vars=config_dict['shared_vars'],
            all_vars=config_dict['all_vars'],
            n_periods=config_dict['n_periods'],
            noise_scale=config_dict['noise_scale'],
            random_state=config_dict['random_state'],
        )

        fusion._reference_source = config_dict['reference_source']

        # Load imputers
        for name in config_dict['sources']:
            imputer_path = path / f'imputer_{name}.pt'
            if imputer_path.exists():
                fusion.imputers[name] = Synthesizer.load(imputer_path)

        # Load unified synthesizer
        unified_path = path / 'unified_synthesizer.pt'
        if unified_path.exists():
            fusion._unified_synthesizer = Synthesizer.load(unified_path)

        fusion._is_fitted = True
        return fusion
