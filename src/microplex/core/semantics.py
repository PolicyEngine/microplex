"""Generic semantic transforms for intermediate microdata frames."""

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


def apply_frame_semantic_transforms(
    frame: pd.DataFrame,
    transforms: Iterable[FrameSemanticTransform],
) -> pd.DataFrame:
    """Apply semantic transforms in sequence."""
    result = frame
    for transform in transforms:
        result = transform.apply(result)
    return result
