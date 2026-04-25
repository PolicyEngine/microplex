"""Generic entity-table checkpoint round-trip tests.

``save_entity_table_checkpoint`` / ``load_entity_table_checkpoint``
persist a dict of named entity tables (households, persons, tax_units,
...) as parquet + a ``metadata.json`` index, keyed by a user-supplied
pipeline stage. Country-specific wrappers (microplex-us, microplex-uk,
...) build a bundle dataclass around this primitive so a downstream
rerun can resume from a saved state without repeating expensive
synthesis or materialization.

These tests drive:

1. Round-trip equivalence for a multi-table dict.
2. ``None``-valued entries round-trip as ``None``.
3. Metadata captures row counts, column names, stage, and any
   caller-provided ``extra_metadata``.
4. Loading a missing path raises a clear error.
5. ``expected_stage`` mismatch raises.
6. Re-saving overwrites the prior snapshot.
7. Invalid stage (empty string) is rejected at save time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microplex.core.checkpoints import (
    load_entity_table_checkpoint,
    save_entity_table_checkpoint,
)


def _make_tables() -> dict[str, pd.DataFrame | None]:
    rng = np.random.default_rng(0)
    n = 25
    return {
        "households": pd.DataFrame(
            {
                "household_id": np.arange(n),
                "weight": rng.uniform(0.1, 5.0, n),
            }
        ),
        "persons": pd.DataFrame(
            {
                "person_id": np.arange(n) * 10,
                "household_id": np.arange(n),
                "age": rng.integers(0, 90, n),
            }
        ),
        "tax_units": None,
    }


class TestEntityTableCheckpoint:
    def test_roundtrip_preserves_frames(self, tmp_path: Path) -> None:
        tables = _make_tables()
        save_entity_table_checkpoint(tables, tmp_path / "checkpoint", stage="stage_a")
        loaded, metadata = load_entity_table_checkpoint(tmp_path / "checkpoint")

        pd.testing.assert_frame_equal(loaded["households"], tables["households"])
        pd.testing.assert_frame_equal(loaded["persons"], tables["persons"])
        assert loaded["tax_units"] is None
        assert metadata["stage"] == "stage_a"

    def test_metadata_captures_schema(self, tmp_path: Path) -> None:
        tables = _make_tables()
        save_entity_table_checkpoint(tables, tmp_path / "checkpoint", stage="s")

        metadata_path = tmp_path / "checkpoint" / "metadata.json"
        assert metadata_path.exists()
        metadata = json.loads(metadata_path.read_text())

        assert metadata["stage"] == "s"
        assert metadata["tables"]["households"]["rows"] == 25
        assert "household_id" in metadata["tables"]["households"]["columns"]
        assert metadata["tables"]["persons"]["rows"] == 25
        assert metadata["tables"]["tax_units"] is None

    def test_extra_metadata_roundtrips(self, tmp_path: Path) -> None:
        """Callers can attach their own fingerprint data (config hashes, etc.)."""
        tables = _make_tables()
        extra = {"config_fingerprint": "abc123", "source_year": 2024}
        save_entity_table_checkpoint(
            tables,
            tmp_path / "checkpoint",
            stage="s",
            extra_metadata=extra,
        )
        _, metadata = load_entity_table_checkpoint(tmp_path / "checkpoint")
        assert metadata["extra"] == extra

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="checkpoint"):
            load_entity_table_checkpoint(tmp_path / "absent")

    def test_expected_stage_mismatch_raises(self, tmp_path: Path) -> None:
        tables = _make_tables()
        save_entity_table_checkpoint(tables, tmp_path / "c", stage="actual")
        with pytest.raises(ValueError, match="expected 'other'"):
            load_entity_table_checkpoint(tmp_path / "c", expected_stage="other")

    def test_save_overwrites_existing(self, tmp_path: Path) -> None:
        first = _make_tables()
        second = {"households": first["households"].head(5), "persons": None}
        save_entity_table_checkpoint(first, tmp_path / "c", stage="s")
        save_entity_table_checkpoint(second, tmp_path / "c", stage="s")
        loaded, _ = load_entity_table_checkpoint(tmp_path / "c")
        assert len(loaded["households"]) == 5
        assert loaded["persons"] is None
        assert "tax_units" not in loaded

    def test_empty_stage_rejected(self, tmp_path: Path) -> None:
        tables = _make_tables()
        with pytest.raises(ValueError, match="stage"):
            save_entity_table_checkpoint(tables, tmp_path / "c", stage="")
