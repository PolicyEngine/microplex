"""Generic semantic transforms and checks for intermediate microdata frames."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum

import pandas as pd


class SemanticTransformStage(Enum):
    """Lifecycle stage where a semantic transform should be applied."""

    POST_SYNTHESIS = "post_synthesis"
    POST_IMPUTATION = "post_imputation"
    POST_DONOR_INTEGRATION = "post_donor_integration"
    POST_CALIBRATION = "post_calibration"
    POST_EXPORT = "post_export"


@dataclass(frozen=True)
class FrameSemanticTransform:
    """Declarative frame-level transform that enforces a semantic invariant."""

    name: str
    transform_frame: Callable[[pd.DataFrame], pd.DataFrame]
    required_columns: tuple[str, ...] = ()
    stage: SemanticTransformStage = SemanticTransformStage.POST_DONOR_INTEGRATION
    notes: str | None = None

    def applies_to(self, frame: pd.DataFrame) -> bool:
        """Return whether the transform can run on the provided frame."""
        return set(self.required_columns).issubset(frame.columns)

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the transform if the required columns are present."""
        if not self.applies_to(frame):
            return frame
        return self.transform_frame(frame)


@dataclass(frozen=True)
class FrameSemanticCheck:
    """Declarative frame-level check for semantic invariant violations."""

    name: str
    violation_mask: Callable[[pd.DataFrame], pd.Series]
    required_columns: tuple[str, ...] = ()
    stage: SemanticTransformStage = SemanticTransformStage.POST_DONOR_INTEGRATION
    notes: str | None = None

    def applies_to(self, frame: pd.DataFrame) -> bool:
        """Return whether the check can run on the provided frame."""
        return set(self.required_columns).issubset(frame.columns)


@dataclass(frozen=True)
class FrameSemanticCheckReport:
    """Summary of one semantic check on a specific frame."""

    name: str
    evaluated: bool
    violating_row_count: int
    required_columns_present: bool
    stage: SemanticTransformStage
    notes: str | None = None

    @property
    def passed(self) -> bool:
        """Whether the check passed after evaluation."""
        return self.evaluated and self.violating_row_count == 0


def apply_frame_semantic_transforms(
    frame: pd.DataFrame,
    transforms: Iterable[FrameSemanticTransform],
) -> pd.DataFrame:
    """Apply semantic transforms in sequence."""
    result = frame
    for transform in transforms:
        result = transform.apply(result)
    return result


def evaluate_frame_semantic_checks(
    frame: pd.DataFrame,
    checks: Iterable[FrameSemanticCheck],
) -> tuple[FrameSemanticCheckReport, ...]:
    """Evaluate semantic checks and return row-count reports."""
    reports: list[FrameSemanticCheckReport] = []
    for check in checks:
        applies = check.applies_to(frame)
        if not applies:
            reports.append(
                FrameSemanticCheckReport(
                    name=check.name,
                    evaluated=False,
                    violating_row_count=0,
                    required_columns_present=False,
                    stage=check.stage,
                    notes=check.notes,
                )
            )
            continue
        violation_mask = pd.Series(
            check.violation_mask(frame),
            index=frame.index,
        ).fillna(False)
        violating_row_count = int(violation_mask.astype(bool).sum())
        reports.append(
            FrameSemanticCheckReport(
                name=check.name,
                evaluated=True,
                violating_row_count=violating_row_count,
                required_columns_present=True,
                stage=check.stage,
                notes=check.notes,
            )
        )
    return tuple(reports)
