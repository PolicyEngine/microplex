"""Experiment registry for dashboard export."""

import json

# Import tracker - handle both package and direct import cases
try:
    from .tracker import ExperimentTracker
except ImportError:
    from tracker import ExperimentTracker


class ExperimentRegistry:
    """Registry that exports experiments for web dashboard."""

    def __init__(self, tracker: ExperimentTracker):
        self.tracker = tracker

    def export_for_dashboard(self, output_path: str | None = None) -> dict:
        """Export all experiments as JSON for web dashboard.

        Returns a structure suitable for visualization:
        {
            "experiments": [...],
            "comparisons": {
                "by_model": {...},
                "by_dataset": {...},
            }
        }
        """
        experiments = []
        for exp_summary in self.tracker.list_experiments():
            exp = self.tracker.load_experiment(exp_summary["id"])
            experiments.append(self._format_experiment(exp))

        result = {
            "experiments": experiments,
            "comparisons": self._compute_comparisons(experiments),
            "metadata": {
                "total_experiments": len(experiments),
                "model_types": list(set(e["model"]["type"] for e in experiments)),
                "datasets": list(set(
                    ds["survey"]
                    for e in experiments
                    for ds in e["datasets"]
                )),
            }
        }

        if output_path:
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

        return result

    def _format_experiment(self, exp) -> dict:
        """Format experiment for dashboard."""
        return {
            "id": exp.id,
            "name": exp.name,
            "description": exp.description,
            "created_at": exp.created_at,
            "datasets": [
                {
                    "survey": ds.survey,
                    "n_train": ds.n_train,
                    "n_holdout": ds.n_holdout,
                    "train_share": ds.train_share,
                    "variables": ds.variables_available,
                    "waves": ds.waves_used,
                    "year": ds.year,
                }
                for ds in exp.datasets
            ],
            "variables": [
                {
                    "name": v.name,
                    "sources": v.sources,
                    "role": v.role,
                    "dtype": v.dtype,
                }
                for v in exp.variables
            ],
            "target": exp.target_variable,
            "model": {
                "type": exp.model.model_type,
                "architecture": exp.model.architecture,
                "training": exp.model.training,
                "quantiles": exp.model.quantiles,
            },
            "training_time_seconds": exp.training_time_seconds,
            "coverage": {
                "overall_median": exp.overall_coverage_median,
                "overall_mean": exp.overall_coverage_mean,
                "by_survey": [
                    {
                        "survey": cr.survey,
                        "n": cr.n_holdout,
                        "median": cr.coverage_median,
                        "mean": cr.coverage_mean,
                        "p95": cr.coverage_p95,
                        "p99": cr.coverage_p99,
                    }
                    for cr in exp.coverage_results
                ],
            },
            "data_paths": {
                "synthetic": exp.synthetic_data_path,
                "holdout_coverage": exp.holdout_coverage_path,
                "model": exp.model_path,
            },
        }

    def _compute_comparisons(self, experiments: list[dict]) -> dict:
        """Compute comparison summaries across experiments."""
        by_model = {}
        by_dataset = {}

        for exp in experiments:
            model_type = exp["model"]["type"]
            if model_type not in by_model:
                by_model[model_type] = []
            by_model[model_type].append({
                "id": exp["id"],
                "name": exp["name"],
                "coverage_median": exp["coverage"]["overall_median"],
            })

            for ds in exp["datasets"]:
                survey = ds["survey"]
                if survey not in by_dataset:
                    by_dataset[survey] = []
                # Find coverage for this survey
                survey_coverage = next(
                    (c for c in exp["coverage"]["by_survey"] if c["survey"] == survey),
                    None
                )
                if survey_coverage:
                    by_dataset[survey].append({
                        "exp_id": exp["id"],
                        "exp_name": exp["name"],
                        "coverage_median": survey_coverage["median"],
                        "n_holdout": survey_coverage["n"],
                    })

        return {
            "by_model": by_model,
            "by_dataset": by_dataset,
        }
