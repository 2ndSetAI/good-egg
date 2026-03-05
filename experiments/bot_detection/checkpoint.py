from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def write_json(path: Path, data: object) -> None:
    """Write data as JSON to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def read_json(path: Path) -> Any:
    """Read JSON from a file."""
    with open(path) as f:
        return json.load(f)


def write_stage_checkpoint(
    data_dir: Path,
    stage_name: str,
    row_counts: dict[str, int],
    details: dict[str, Any] | None = None,
) -> Path:
    """Write a stage completion checkpoint file.

    Returns the path to the written checkpoint.
    """
    checkpoint = {
        "stage": stage_name,
        "timestamp": datetime.now(UTC).isoformat(),
        "row_counts": row_counts,
        "details": details or {},
    }
    path = data_dir / f"{stage_name}_complete.json"
    write_json(path, checkpoint)
    logger.info("Checkpoint written: %s (%s)", path, row_counts)
    return path


def read_stage_checkpoint(
    data_dir: Path,
    stage_name: str,
) -> dict[str, Any] | None:
    """Read a stage checkpoint. Returns None if not found."""
    path = data_dir / f"{stage_name}_complete.json"
    if not path.exists():
        return None
    return read_json(path)
