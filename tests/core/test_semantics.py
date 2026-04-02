"""Tests for generic semantic transforms."""

from __future__ import annotations

import pandas as pd

from microplex.core.semantics import (
    FrameSemanticTransform,
    SemanticTransformStage,
    apply_frame_semantic_transforms,
)


def test_frame_semantic_transform_skips_when_required_columns_missing():
    frame = pd.DataFrame({"employment_income": [100.0]})
    transform = FrameSemanticTransform(
        name="zero_minor_wages",
        required_columns=("age", "employment_income"),
        transform_frame=lambda current: current.assign(employment_income=0.0),
        stage=SemanticTransformStage.POST_DONOR_INTEGRATION,
    )

    result = transform.apply(frame)

    assert result.equals(frame)


def test_apply_frame_semantic_transforms_runs_in_order():
    frame = pd.DataFrame({"value": [1.0]})
    first = FrameSemanticTransform(
        name="double",
        required_columns=("value",),
        transform_frame=lambda current: current.assign(value=current["value"] * 2.0),
    )
    second = FrameSemanticTransform(
        name="increment",
        required_columns=("value",),
        transform_frame=lambda current: current.assign(value=current["value"] + 1.0),
    )

    result = apply_frame_semantic_transforms(frame, (first, second))

    assert result["value"].tolist() == [3.0]
