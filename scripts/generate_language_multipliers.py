#!/usr/bin/env python3
"""Generate language normalization multipliers from GitHub Innovation Graph data.

Downloads the languages.csv from github/innovationgraph, filters to the latest
quarter of programming languages, and computes ecosystem-size multipliers using
a power-law compression formula: (max_pushers / lang_pushers) ^ exponent.

Usage:
    python scripts/generate_language_multipliers.py
    python scripts/generate_language_multipliers.py --exponent 0.25 --min-pushers 10000
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.request

CSV_URL = (
    "https://raw.githubusercontent.com/github/innovationgraph/main/data/languages.csv"
)


def fetch_csv(url: str) -> str:
    """Download the CSV data from the given URL."""
    with urllib.request.urlopen(url) as resp:  # noqa: S310
        return resp.read().decode("utf-8")


def compute_multipliers(
    csv_text: str, exponent: float, min_pushers: int
) -> dict[str, float]:
    """Parse CSV and compute language multipliers.

    Filters to:
    - Latest quarter available in the data
    - language_type == "programming"

    Then sums num_pushers globally per language and computes:
        (max_pushers / lang_pushers) ^ exponent
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)

    if not rows:
        print("Error: CSV is empty", file=sys.stderr)
        sys.exit(1)

    # Find the latest quarter
    quarters = sorted({r["quarter"] for r in rows})
    latest_quarter = quarters[-1]
    print(f"Using quarter: {latest_quarter}", file=sys.stderr)

    # Filter to latest quarter and programming languages only
    filtered = [
        r
        for r in rows
        if r["quarter"] == latest_quarter and r["language_type"] == "programming"
    ]

    # Sum num_pushers globally per language
    pushers_by_lang: dict[str, int] = {}
    for row in filtered:
        lang = row["language"]
        count = int(row["num_pushers"])
        pushers_by_lang[lang] = pushers_by_lang.get(lang, 0) + count

    # Filter to languages with enough pushers
    pushers_by_lang = {
        lang: count
        for lang, count in pushers_by_lang.items()
        if count >= min_pushers
    }

    if not pushers_by_lang:
        print("Error: No languages met the min_pushers threshold", file=sys.stderr)
        sys.exit(1)

    max_pushers = max(pushers_by_lang.values())

    # Compute multipliers
    multipliers: dict[str, float] = {}
    for lang, count in pushers_by_lang.items():
        multipliers[lang] = round((max_pushers / count) ** exponent, 2)

    # Sort by multiplier value
    return dict(sorted(multipliers.items(), key=lambda x: x[1]))


def print_dict(multipliers: dict[str, float]) -> None:
    """Print the multipliers as a Python dict literal."""
    print("multipliers: dict[str, float] = Field(default_factory=lambda: {")
    for lang, mult in multipliers.items():
        print(f'    "{lang}": {mult},')
    print("})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate language multipliers from GitHub Innovation Graph"
    )
    parser.add_argument(
        "--exponent",
        type=float,
        default=0.3,
        help="Power-law compression exponent (default: 0.3)",
    )
    parser.add_argument(
        "--min-pushers",
        type=int,
        default=5000,
        help="Minimum num_pushers to include a language (default: 5000)",
    )
    args = parser.parse_args()

    print(f"Fetching {CSV_URL} ...", file=sys.stderr)
    csv_text = fetch_csv(CSV_URL)

    multipliers = compute_multipliers(csv_text, args.exponent, args.min_pushers)
    print(f"\n{len(multipliers)} languages with >= {args.min_pushers} pushers:\n")
    print_dict(multipliers)


if __name__ == "__main__":
    main()
