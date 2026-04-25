"""Generic entity-table pipeline checkpoints.

``save_entity_table_checkpoint`` and ``load_entity_table_checkpoint``
persist a dict of named entity tables to disk as parquet files plus a
``metadata.json`` index keyed by a pipeline stage. Country-specific
microplex packages (e.g. ``microplex-us``) wrap these with typed entity
bundles so a downstream rerun can resume from a saved state without
redoing expensive upstream work (synthesis, donor imputation,
tax-benefit microsim).

Usage
-----

.. code-block:: python

    from microplex.core.checkpoints import save_entity_table_checkpoint

    save_entity_table_checkpoint(
        {"households": households_df, "persons": persons_df},
        Path("artifacts/run/checkpoint"),
        stage="post_imputation",
        extra_metadata={"config_fingerprint": config_hash},
    )

At load time, ``extra_metadata`` is available for cache-invalidation
decisions (stale checkpoint if the fingerprint differs from the
current config).
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd


def save_entity_table_checkpoint(
    tables: Mapping[str, pd.DataFrame | None],
    path: str | Path,
    *,
    stage: str,
    extra_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Persist a named dict of entity tables to ``path`` as parquet + metadata.

    Args:
        tables: Mapping from entity-table name (``"households"``,
            ``"persons"``, ...) to the DataFrame for that table, or
            ``None`` if the country / pipeline stage doesn't populate
            that entity.
        path: Target directory. Any existing directory at this path is
            removed and replaced.
        stage: Non-empty identifier describing the pipeline stage the
            checkpoint was taken at (``"post_imputation"``,
            ``"post_microsim"``, ...). Stored in ``metadata.json`` and
            validated by ``expected_stage`` on load.
        extra_metadata: Optional mapping attached to the checkpoint
            under the ``"extra"`` key — use for config fingerprints,
            source-data versions, etc. that a caller wants to check for
            cache invalidation.

    Returns:
        The directory the checkpoint was written to.
    """
    if not stage:
        raise ValueError("stage must be a non-empty string")

    checkpoint_dir = Path(path)
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True)

    table_metadata: dict[str, dict[str, Any] | None] = {}
    for table_name, frame in tables.items():
        if frame is None:
            table_metadata[table_name] = None
            continue
        frame.to_parquet(checkpoint_dir / f"{table_name}.parquet", index=False)
        table_metadata[table_name] = {
            "rows": int(len(frame)),
            "columns": list(frame.columns),
        }

    metadata: dict[str, Any] = {
        "format_version": 1,
        "stage": stage,
        "tables": table_metadata,
    }
    if extra_metadata is not None:
        metadata["extra"] = dict(extra_metadata)

    (checkpoint_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str)
    )
    return checkpoint_dir


def load_entity_table_checkpoint(
    path: str | Path,
    *,
    expected_stage: str | None = None,
) -> tuple[dict[str, pd.DataFrame | None], dict[str, Any]]:
    """Load a dict of entity tables previously saved by ``save_entity_table_checkpoint``.

    Returns ``(tables, metadata)``. If ``expected_stage`` is set and
    the saved stage doesn't match, a ``ValueError`` is raised —
    protects against resuming from the wrong pipeline stage.
    """
    checkpoint_dir = Path(path)
    metadata_path = checkpoint_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Entity-table checkpoint not found at {checkpoint_dir}"
        )
    metadata = json.loads(metadata_path.read_text())

    saved_stage = metadata.get("stage")
    if expected_stage is not None and saved_stage != expected_stage:
        raise ValueError(
            f"Checkpoint at {checkpoint_dir} has stage {saved_stage!r}, "
            f"expected {expected_stage!r}"
        )

    tables: dict[str, pd.DataFrame | None] = {}
    table_metadata = metadata.get("tables", {})
    for table_name, table_info in table_metadata.items():
        if table_info is None:
            tables[table_name] = None
            continue
        tables[table_name] = pd.read_parquet(
            checkpoint_dir / f"{table_name}.parquet"
        )
    return tables, metadata
