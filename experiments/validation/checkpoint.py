from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def output_is_current(output_path: Path, *input_paths: Path) -> bool:
    """Return True if output_path exists and is newer than all input_paths.

    If no input_paths are given, simply checks existence.
    """
    if not output_path.exists():
        return False
    if not input_paths:
        return True
    output_mtime = output_path.stat().st_mtime
    return all(
        inp.exists() and output_mtime >= inp.stat().st_mtime
        for inp in input_paths
    )


def repo_already_collected(data_dir: Path, owner: str, repo: str) -> bool:
    """Check if a repo's PR data has already been collected."""
    output_file = data_dir / f"{owner}__{repo}.jsonl"
    return output_file.exists() and output_file.stat().st_size > 0


def author_already_fetched(authors_dir: Path, login: str) -> bool:
    """Check if an author's data has already been fetched."""
    output_file = authors_dir / f"{login}.json"
    return output_file.exists() and output_file.stat().st_size > 0


def load_scored_pr_keys(scored_dir: Path, filename: str) -> set[str]:
    """Load already-scored PR keys from a parquet file.

    Returns a set of 'repo::pr_number' strings.
    """
    parquet_path = scored_dir / filename
    if not parquet_path.exists():
        return set()
    try:
        import pandas as pd
        df = pd.read_parquet(parquet_path, columns=["repo", "pr_number"])
        return {
            f"{row['repo']}::{row['pr_number']}"
            for _, row in df.iterrows()
        }
    except Exception:
        logger.warning("Failed to load checkpoint from %s", parquet_path)
        return set()


def append_jsonl(path: Path, records: list[dict]) -> None:
    """Append records as JSON lines to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """Read all records from a JSONL file."""
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(path: Path, data: object) -> None:
    """Write data as JSON to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def read_json(path: Path) -> object:
    """Read JSON from a file."""
    with open(path) as f:
        return json.load(f)
