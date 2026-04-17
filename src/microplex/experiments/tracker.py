"""Experiment tracking for synthesis experiments.

Tracks:
- Dataset splits (train/holdout shares per survey)
- Variables used with sources
- Panel structure (waves/snapshots if applicable)
- Technique details (model type, hyperparameters)
- Coverage metrics per survey
- Per-record coverage with pointers to nearest synthetic
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd


@dataclass
class DatasetSplit:
    """Information about a dataset's train/holdout split."""
    survey: str
    n_train: int
    n_holdout: int
    train_share: float
    variables_available: list[str]
    waves_used: list[int] | None = None  # For panel data
    year: int | None = None


@dataclass
class Variable:
    """A variable used in the experiment."""
    name: str
    sources: list[str]  # Which surveys have this variable
    role: str  # "predictor", "target", "weight"
    dtype: str  # "continuous", "categorical", "binary"
    missing_handling: str = "fill_zero"  # How missing values are handled


@dataclass
class ModelConfig:
    """Model configuration and hyperparameters."""
    model_type: str  # "zi-qdnn", "qrf", "maf", etc.
    architecture: dict = field(default_factory=dict)  # layers, units, etc.
    training: dict = field(default_factory=dict)  # epochs, batch_size, lr, etc.
    quantiles: list[float] | None = None  # For quantile models


@dataclass
class CoverageResult:
    """Coverage results for a survey's holdout set."""
    survey: str
    n_holdout: int
    coverage_median: float
    coverage_mean: float
    coverage_p95: float
    coverage_p99: float


@dataclass
class Experiment:
    """A complete synthesis experiment."""

    # Identification
    id: str
    name: str
    description: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Data configuration
    datasets: list[DatasetSplit] = field(default_factory=list)
    variables: list[Variable] = field(default_factory=list)
    target_variable: str = ""

    # Model configuration
    model: ModelConfig = field(default_factory=lambda: ModelConfig(model_type="unknown"))

    # Training info
    training_time_seconds: float = 0.0

    # Results
    coverage_results: list[CoverageResult] = field(default_factory=list)
    overall_coverage_median: float = 0.0
    overall_coverage_mean: float = 0.0

    # File paths to stored data
    synthetic_data_path: str | None = None
    holdout_coverage_path: str | None = None  # Per-record coverage
    model_path: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        """Create from dictionary."""
        # Reconstruct nested dataclasses
        d["datasets"] = [DatasetSplit(**ds) for ds in d.get("datasets", [])]
        d["variables"] = [Variable(**v) for v in d.get("variables", [])]
        d["model"] = ModelConfig(**d.get("model", {"model_type": "unknown"}))
        d["coverage_results"] = [CoverageResult(**cr) for cr in d.get("coverage_results", [])]
        return cls(**d)


class ExperimentTracker:
    """Track and save synthesis experiments."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir = self.base_dir / "experiments"
        self.experiments_dir.mkdir(exist_ok=True)
        self.data_dir = self.base_dir / "data"
        self.data_dir.mkdir(exist_ok=True)

    def create_experiment(
        self,
        name: str,
        description: str,
    ) -> Experiment:
        """Create a new experiment with auto-generated ID."""
        exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return Experiment(id=exp_id, name=name, description=description)

    def add_dataset(
        self,
        exp: Experiment,
        survey: str,
        n_train: int,
        n_holdout: int,
        variables_available: list[str],
        waves_used: list[int] | None = None,
        year: int | None = None,
    ) -> None:
        """Add dataset split info to experiment."""
        total = n_train + n_holdout
        exp.datasets.append(DatasetSplit(
            survey=survey,
            n_train=n_train,
            n_holdout=n_holdout,
            train_share=n_train / total if total > 0 else 0,
            variables_available=variables_available,
            waves_used=waves_used,
            year=year,
        ))

    def add_variable(
        self,
        exp: Experiment,
        name: str,
        sources: list[str],
        role: str = "predictor",
        dtype: str = "continuous",
        missing_handling: str = "fill_zero",
    ) -> None:
        """Add variable info to experiment."""
        exp.variables.append(Variable(
            name=name,
            sources=sources,
            role=role,
            dtype=dtype,
            missing_handling=missing_handling,
        ))

    def set_model_config(
        self,
        exp: Experiment,
        model_type: str,
        architecture: dict | None = None,
        training: dict | None = None,
        quantiles: list[float] | None = None,
    ) -> None:
        """Set model configuration."""
        exp.model = ModelConfig(
            model_type=model_type,
            architecture=architecture or {},
            training=training or {},
            quantiles=quantiles,
        )

    def add_coverage_result(
        self,
        exp: Experiment,
        survey: str,
        holdout_incomes: np.ndarray,
        coverage_distances: np.ndarray,
    ) -> None:
        """Add coverage results for a survey."""
        exp.coverage_results.append(CoverageResult(
            survey=survey,
            n_holdout=len(holdout_incomes),
            coverage_median=float(np.median(coverage_distances)),
            coverage_mean=float(np.mean(coverage_distances)),
            coverage_p95=float(np.percentile(coverage_distances, 95)),
            coverage_p99=float(np.percentile(coverage_distances, 99)),
        ))

    def save_synthetic_data(
        self,
        exp: Experiment,
        synthetic: np.ndarray,
        train_predictors: np.ndarray,
        predictor_names: list[str],
    ) -> str:
        """Save synthetic data to parquet."""
        path = self.data_dir / f"{exp.id}_synthetic.parquet"
        df = pd.DataFrame(train_predictors, columns=predictor_names)
        df["synthetic_income"] = synthetic
        df.to_parquet(path, index=False)
        exp.synthetic_data_path = str(path)
        return str(path)

    def save_holdout_coverage(
        self,
        exp: Experiment,
        holdout_df: pd.DataFrame,
        holdout_incomes: np.ndarray,
        coverage_distances: np.ndarray,
        nearest_synthetic_idx: np.ndarray,
        nearest_synthetic_values: np.ndarray,
    ) -> str:
        """Save per-record holdout coverage with pointers to nearest synthetic."""
        path = self.data_dir / f"{exp.id}_holdout_coverage.parquet"

        result = holdout_df.copy()
        result["holdout_income"] = holdout_incomes
        result["coverage_distance"] = coverage_distances
        result["nearest_synthetic_idx"] = nearest_synthetic_idx
        result["nearest_synthetic_value"] = nearest_synthetic_values
        result["coverage_error"] = holdout_incomes - nearest_synthetic_values

        result.to_parquet(path, index=False)
        exp.holdout_coverage_path = str(path)
        return str(path)

    def save_experiment(self, exp: Experiment) -> str:
        """Save experiment metadata to JSON."""
        path = self.experiments_dir / f"{exp.id}.json"
        with open(path, "w") as f:
            json.dump(exp.to_dict(), f, indent=2, default=str)
        return str(path)

    def load_experiment(self, exp_id: str) -> Experiment:
        """Load experiment from JSON."""
        path = self.experiments_dir / f"{exp_id}.json"
        with open(path) as f:
            return Experiment.from_dict(json.load(f))

    def list_experiments(self) -> list[dict]:
        """List all experiments with summary info."""
        experiments = []
        for path in self.experiments_dir.glob("*.json"):
            with open(path) as f:
                d = json.load(f)
                experiments.append({
                    "id": d["id"],
                    "name": d["name"],
                    "created_at": d["created_at"],
                    "model_type": d.get("model", {}).get("model_type", "unknown"),
                    "overall_coverage_median": d.get("overall_coverage_median", 0),
                    "n_datasets": len(d.get("datasets", [])),
                })
        return sorted(experiments, key=lambda x: x["created_at"], reverse=True)
