#!/usr/bin/env python3
"""Augment staff summary CSVs with names from the HMRI affiliate index.

Usage:
    python link_staff_affiliates.py staff_summary.csv \
        --index "HMRI affiliate index.csv" --output staff_summary_named.csv

The script expects the staff summary CSV exported from the dashboard. It joins
on the `staff_id` column (numberplate) to fetch staff names and faculty details
from the affiliate index.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def link_staff_summaries(
    staff_csv: Path,
    index_csv: Path,
    output_csv: Optional[Path] = None,
) -> Path:
    """Merge staff summary data with the affiliate index.

    Args:
        staff_csv: Path to the staff summary CSV exported from the dashboard.
        index_csv: Path to the HMRI affiliate index CSV.
        output_csv: Optional path for the merged CSV. If omitted, a new file is
            created alongside `staff_csv` with `_with_names` appended.

    Returns:
        Path to the written CSV file.
    """
    staff_df = pd.read_csv(staff_csv)
    if "staff_id" not in staff_df.columns:
        raise ValueError(
            "The staff summary CSV must include a 'staff_id' column. "
            "Ensure you exported it from the dashboard staff summary download."
        )

    index_df = pd.read_csv(index_csv)
    required_columns = {"NumberPlate", "Staff_First_Name", "Staff_Surname"}
    missing = required_columns - set(index_df.columns)
    if missing:
        raise ValueError(
            f"The affiliate index is missing required columns: {', '.join(sorted(missing))}"
        )

    staff_df = staff_df.copy()
    index_df = index_df.copy()

    staff_df["staff_id"] = staff_df["staff_id"].astype(str).str.strip()
    index_df["NumberPlate"] = index_df["NumberPlate"].astype(str).str.strip()

    index_lookup = (
        index_df[
            [
                "NumberPlate",
                "Staff_First_Name",
                "Staff_Surname",
                "Staff_Faculty",
                "Staff_School",
            ]
        ]
        .drop_duplicates(subset="NumberPlate")
        .rename(columns={"NumberPlate": "numberplate"})
    )

    merged = staff_df.merge(
        index_lookup,
        how="left",
        left_on="staff_id",
        right_on="numberplate",
    )

    merged["staff_full_name"] = (
        merged["Staff_First_Name"].fillna("").str.strip()
        + " "
        + merged["Staff_Surname"].fillna("").str.strip()
    )
    merged["staff_full_name"] = merged["staff_full_name"].str.strip().replace({"": pd.NA})

    merged = merged.drop(columns=["numberplate"])

    column_order = [
        "staff_id",
        "staff_full_name",
        "Staff_First_Name",
        "Staff_Surname",
        "Staff_Faculty",
        "Staff_School",
    ] + [
        col
        for col in merged.columns
        if col
        not in {
            "staff_id",
            "staff_full_name",
            "Staff_First_Name",
            "Staff_Surname",
            "Staff_Faculty",
            "Staff_School",
        }
    ]
    merged = merged[column_order]

    if output_csv is None:
        output_csv = staff_csv.with_name(f"{staff_csv.stem}_with_names{staff_csv.suffix}")

    merged.to_csv(output_csv, index=False)
    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join dashboard staff summary results with the HMRI affiliate index."
    )
    parser.add_argument(
        "staff_summary",
        type=Path,
        help="Path to the staff summary CSV exported from the dashboard.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("HMRI affiliate index.csv"),
        help="Path to the HMRI affiliate index CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the merged CSV. Defaults to '<staff_summary>_with_names.csv'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = link_staff_summaries(args.staff_summary, args.index, args.output)
    print(f"Merged staff summary written to: {output_path}")


if __name__ == "__main__":
    main()
