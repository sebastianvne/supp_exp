#!/usr/bin/env python3
"""
Summarize boolean/status columns in composed pipeline CSV outputs.
"""
'''
Usage:
python scripts/composed_flag_counts.py \
  --directory results/pipelines/or-bench-hard-1k/composed_pipeline \
  --prefix Qwen-Qwen1.5-7B \
  --sample-rows 0
'''
import argparse
from pathlib import Path
from typing import List

import pandas as pd


REWRITE_COLUMN = "is_rewrite_filtered_by_implicit_pipeline"

DEFAULT_COLUMNS = [
    "is_filtered_by_composed_pipeline",
    REWRITE_COLUMN,
    "is_original_filtered_by_implicit_pipeline",
    "is_filtered_by_implicit_basemodel",
    "is_filtered_by_explicit_pipeline",
    "is_filtered_by_explicit_basemodel",
]


def _print_counts(series: pd.Series, column: str) -> None:
    counts = (
        series.value_counts(dropna=False)
        .rename_axis(column)
        .reset_index(name="count")
    )
    print(counts.to_string(index=False))


def summarize_dataframe(
    df: pd.DataFrame,
    label: str,
    columns: List[str],
    sample_rows: int,
    include_prompt: bool,
    rewrite_ratios: List[tuple],
) -> None:
    print(f"\n== {label} ==")
    print(f"Total rows: {len(df)}")
    for column in columns:
        if column not in df.columns:
            print(f"[Skip] Column '{column}' not found.")
            continue
        print(f"\n-- {column} value counts --")
        series = df[column]
        _print_counts(series, column)
        if column == REWRITE_COLUMN:
            total = len(series)
            positives = series.eq(1).sum()
            if total == 0:
                print("   ratio (value == 1): n/a (0/0)")
            else:
                ratio = positives / total
                print(f"   ratio (value == 1): {ratio:.2%} ({positives}/{total})")
            rewrite_ratios.append((label, positives, total))

        if sample_rows <= 0:
            continue

        sample_df = df[[column]].copy()
        if include_prompt and "prompt" in df.columns:
            sample_df["prompt"] = df["prompt"]
        if "final_response" in df.columns:
            sample_df["final_response"] = df["final_response"]

        print(f"\n   Sample rows (first {sample_rows}) for '{column}':")
        print(sample_df.head(sample_rows).to_string(index=False))


def summarize_file(
    path: Path,
    columns: List[str],
    sample_rows: int,
    include_prompt: bool,
    rewrite_ratios: List[tuple],
) -> None:
    df = pd.read_csv(path)
    summarize_dataframe(
        df, path.name, columns, sample_rows, include_prompt, rewrite_ratios
    )


def main():
    parser = argparse.ArgumentParser(
        description="Count status columns in composed pipeline CSV files."
    )
    parser.add_argument(
        "--directory",
        default="/disk1/users/wangjh/supp_exp/results/pipelines/or-bench-hard-1k/composed_pipeline",
        help="Directory containing composed pipeline CSV files.",
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="Filename prefix to match (e.g., 'claude-3-haiku-20240307_').",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Optional filename suffix to match (e.g., '_deception_True_3_results.csv').",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=DEFAULT_COLUMNS,
        help="Columns to summarize. Defaults to composed pipeline filter flags.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=3,
        help="Number of example rows per column to print (0 to skip).",
    )
    parser.add_argument(
        "--include-prompt",
        action="store_true",
        help="Include prompt text in the sample rows output.",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate all matched files into a single summary table.",
    )
    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.is_dir():
        raise SystemExit(f"Directory not found: {directory}")

    files = sorted(p for p in directory.glob(f"{args.prefix}*.csv") if p.is_file())
    if args.suffix:
        files = [p for p in files if p.name.endswith(args.suffix)]
    if not files:
        raise SystemExit(f"No files found with prefix '{args.prefix}' in {directory}")

    rewrite_ratios: List[tuple] = []
    if args.aggregate:
        frames = [pd.read_csv(path) for path in files]
        combined = pd.concat(frames, ignore_index=True)
        summarize_dataframe(
            combined,
            f"Aggregated summary ({len(files)} files)",
            args.columns,
            args.sample_rows,
            args.include_prompt,
            rewrite_ratios,
        )
    else:
        for path in files:
            summarize_file(
                path, args.columns, args.sample_rows, args.include_prompt, rewrite_ratios
            )

    if rewrite_ratios:
        print("\n== Rewrite filter ratio summary ==")
        for label, positives, total in rewrite_ratios:
            if total == 0:
                ratio_text = "n/a (0/0)"
            else:
                ratio = positives / total
                ratio_text = f"{ratio:.2%} ({positives}/{total})"
            print(f"{label}: {ratio_text}")


if __name__ == "__main__":
    main()

