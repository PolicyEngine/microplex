"""Tests for generic semantic transforms."""

from __future__ import annotations

import pandas as pd

from microplex.core.semantics import (
    FrameSemanticCheck,
    FrameSemanticCheckReport,
    FrameSemanticTransform,
    SemanticTransformStage,
    apply_frame_semantic_transforms,
    evaluate_frame_semantic_checks,
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


def test_evaluate_frame_semantic_checks_reports_violations():
    frame = pd.DataFrame(
        {
            "age": [12, 20],
            "employment_income": [25.0, 100.0],
        }
    )
    check = FrameSemanticCheck(
        name="minor_positive_wages",
        required_columns=("age", "employment_income"),
        violation_mask=lambda current: (current["age"] < 18)
        & (current["employment_income"] > 0),
        stage=SemanticTransformStage.POST_DONOR_INTEGRATION,
    )

    reports = evaluate_frame_semantic_checks(frame, (check,))

    assert reports == (
        FrameSemanticCheckReport(
            name="minor_positive_wages",
            evaluated=True,
            violating_row_count=1,
            required_columns_present=True,
            stage=SemanticTransformStage.POST_DONOR_INTEGRATION,
            notes=None,
        ),
    )
    assert reports[0].passed is False


def test_evaluate_frame_semantic_checks_marks_missing_columns_as_not_evaluated():
    frame = pd.DataFrame({"employment_income": [25.0]})
    check = FrameSemanticCheck(
        name="minor_positive_wages",
        required_columns=("age", "employment_income"),
        violation_mask=lambda current: current["employment_income"] > 0,
    )

    reports = evaluate_frame_semantic_checks(frame, (check,))

    assert reports[0].evaluated is False
    assert reports[0].required_columns_present is False
    assert reports[0].violating_row_count == 0
