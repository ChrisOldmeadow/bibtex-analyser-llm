#!/usr/bin/env python3
"""Filter publications by staff identifier list.

Takes a CSV file with staff identifiers and filters the publication dataset
to only include publications authored by those staff members.

Usage:
    python scripts/filter_by_staff.py \\
        --staff-list staff_list.csv \\
        --publications data/hmri_enriched.csv \\
        --output filtered_publications.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def load_staff_list(staff_file: Path, column: str = None) -> set:
    """Load staff identifiers from CSV file.

    Args:
        staff_file: Path to CSV with staff identifiers
        column: Column name containing staff IDs (auto-detect if None)

    Returns:
        Set of staff identifiers (as strings)
    """
    df = pd.read_csv(staff_file)

    # Auto-detect staff ID column if not specified
    if column is None:
        # Look for common column names
        possible_cols = ['staff_id', 'numberplate', 'NumberPlate', 'Staff_ID', 'ID', 'id']
        for col in possible_cols:
            if col in df.columns:
                column = col
                break

        # If still not found, use first column
        if column is None:
            column = df.columns[0]
            print(f"Using first column '{column}' as staff identifier")

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in {staff_file}. Available: {', '.join(df.columns)}")

    # Convert to set of strings, removing any NaN values
    staff_ids = set(df[column].dropna().astype(str).str.strip())
    print(f"Loaded {len(staff_ids)} staff identifiers from {staff_file}")

    return staff_ids


def filter_publications(pub_file: Path, staff_ids: set, staff_column: str = None) -> pd.DataFrame:
    """Filter publications to only those by specified staff.

    Args:
        pub_file: Path to publication CSV
        staff_ids: Set of staff identifiers to filter by
        staff_column: Column name with staff IDs (auto-detect if None)

    Returns:
        Filtered DataFrame
    """
    print(f"\nLoading publications from {pub_file}...")
    df = pd.read_csv(pub_file)
    original_count = len(df)
    print(f"  Loaded {original_count} publications")

    # Auto-detect staff column if not specified
    if staff_column is None:
        # Prefer all_staff_ids if available (handles deduplicated multi-staff papers)
        if 'all_staff_ids' in df.columns:
            staff_column = 'all_staff_ids'
        elif 'staff_id' in df.columns:
            staff_column = 'staff_id'
        elif 'numberplate' in df.columns:
            staff_column = 'numberplate'
        elif 'NumberPlate' in df.columns:
            staff_column = 'NumberPlate'
        else:
            raise ValueError(f"Could not find staff ID column. Available: {', '.join(df.columns)}")

    if staff_column not in df.columns:
        raise ValueError(f"Column '{staff_column}' not found. Available: {', '.join(df.columns)}")

    print(f"  Filtering by column: {staff_column}")

    # Handle all_staff_ids (array column from deduplication)
    if staff_column == 'all_staff_ids':
        def matches_staff(row_value):
            if pd.isna(row_value):
                return False
            # Handle string representation of list
            if isinstance(row_value, str):
                # Remove brackets and split
                row_value = row_value.strip('[]').replace("'", "").replace('"', '')
                ids = [id.strip() for id in row_value.split(',') if id.strip()]
            elif isinstance(row_value, list):
                ids = [str(id).strip() for id in row_value]
            else:
                ids = [str(row_value).strip()]

            return any(id in staff_ids for id in ids)

        mask = df[staff_column].apply(matches_staff)

    # Handle single staff ID column
    else:
        df[staff_column] = df[staff_column].astype(str).str.strip()
        mask = df[staff_column].isin(staff_ids)

    filtered_df = df[mask].copy()
    filtered_count = len(filtered_df)

    print(f"\n  Filtered to {filtered_count} publications ({filtered_count/original_count*100:.1f}%)")
    print(f"  Excluded: {original_count - filtered_count} publications")

    return filtered_df


def main():
    parser = argparse.ArgumentParser(
        description="Filter publications by staff identifier list"
    )
    parser.add_argument(
        '--staff-list',
        type=Path,
        required=True,
        help='CSV file with staff identifiers'
    )
    parser.add_argument(
        '--publications',
        type=Path,
        required=True,
        help='Publication CSV file to filter'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output path for filtered publications CSV'
    )
    parser.add_argument(
        '--staff-column',
        type=str,
        help='Column name in staff list CSV (auto-detect if not specified)'
    )
    parser.add_argument(
        '--pub-staff-column',
        type=str,
        help='Column name in publications CSV with staff IDs (auto-detect if not specified)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.staff_list.exists():
        print(f"Error: Staff list file not found: {args.staff_list}")
        sys.exit(1)

    if not args.publications.exists():
        print(f"Error: Publications file not found: {args.publications}")
        sys.exit(1)

    try:
        # Load staff list
        print("="*60)
        print("FILTER PUBLICATIONS BY STAFF LIST")
        print("="*60)
        staff_ids = load_staff_list(args.staff_list, args.staff_column)

        # Filter publications
        filtered_df = filter_publications(args.publications, staff_ids, args.pub_staff_column)

        # Save output
        args.output.parent.mkdir(parents=True, exist_ok=True)
        filtered_df.to_csv(args.output, index=False)

        print("\n" + "="*60)
        print("SUCCESS")
        print("="*60)
        print(f"Filtered publications saved to: {args.output}")
        print(f"Total publications: {len(filtered_df)}")

        # Show unique staff count
        if 'all_staff_ids' in filtered_df.columns:
            # Count unique staff from array column
            all_staff = set()
            for ids in filtered_df['all_staff_ids'].dropna():
                if isinstance(ids, str):
                    ids = ids.strip('[]').replace("'", "").replace('"', '')
                    staff_list = [id.strip() for id in ids.split(',') if id.strip()]
                    all_staff.update(staff_list)
                elif isinstance(ids, list):
                    all_staff.update([str(id).strip() for id in ids])
            print(f"Unique staff members: {len(all_staff)}")

        print("="*60)

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
