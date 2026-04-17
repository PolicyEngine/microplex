"""Microplex Tracking Dashboard.

Compare microplex outputs against reference datasets:
- policyengine-us-data (Enhanced CPS, PUF)
- Yale Budget Lab
- PSL Tax-Data
- IRS SOI
- Census
- SSA

Usage:
    from dashboard import TrackingDashboard, run_dashboard

    # Run full dashboard
    dashboard = run_dashboard()

    # Or manually
    dashboard = TrackingDashboard()
    dashboard.load_irs_soi_targets()
    dashboard.load_ssa_targets()
    results = dashboard.compare_to_microplex(microplex_df)
    report = dashboard.generate_report()
"""

from .external_sources import (
    generate_targets_markdown,
    get_all_external_targets,
)
from .policyengine_comparison import (
    compare_all_variables,
    compare_distributions,
    run_policyengine_comparison,
)
from .tracking import (
    ComparisonResult,
    DataCoverage,
    TrackingDashboard,
    ValidationTarget,
    run_dashboard,
)

__all__ = [
    "TrackingDashboard",
    "ValidationTarget",
    "ComparisonResult",
    "DataCoverage",
    "run_dashboard",
    "compare_distributions",
    "compare_all_variables",
    "run_policyengine_comparison",
    "get_all_external_targets",
    "generate_targets_markdown",
]
