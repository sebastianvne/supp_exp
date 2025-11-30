#!/usr/bin/env python3
"""
Compute refusal ratios where both rewrite and explicit pipelines filter the sample.

Example:
python scripts/both_flag_counts.py \
  --directory results/pipelines/or-bench-hard-1k/composed_pipeline \
  --output both_filter_results.md
"""
import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd


DEFAULT_MODELS = [
    "claude-3-haiku-20240307",
    "google-gemma-7b",
    "gpt-3.5-turbo-0125",
    "gpt-4o",
    "Qwen-Qwen1.5-7B",
    "meta-llama-Llama-2-7b-chat-hf",
    "meta-llama-Llama-3.1-8B",
]
CATEGORIES = [
    "deception",
    "harassment",
    "harmful",
    "hate",
    "illegal",
    "privacy",
    "self-harm",
    "sexual",
    "unethical",
    "violence",
]
REWRITE_COL = "is_rewrite_filtered_by_implicit_pipeline"
EXPLICIT_COL = "is_filtered_by_explicit_pipeline"


def compute_ratio(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=[REWRITE_COL, EXPLICIT_COL])
    total = len(df)
    if total == 0:
        return None
    both = (df[REWRITE_COL] == 1) & (df[EXPLICIT_COL] == 1)
    return both.sum() / total


def format_ratio(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value * 100:.1f}%"


def render_markdown(table: pd.DataFrame) -> str:
    header = ["model"] + CATEGORIES + ["overall"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|:" + "------|".join([""] * (len(header)))  # simple alignment
    ]
    for _, row in table.iterrows():
        cells = [row["model"]] + [format_ratio(row[cat]) for cat in CATEGORIES] + [format_ratio(row["overall"])]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_table(directory: Path, models: List[str]) -> pd.DataFrame:
    rows = []
    header = ["model"] + CATEGORIES + ["overall"]
    align = ["|:------------------------------"] + ["|----------"] * len(CATEGORIES) + ["|:--------"]
    lines = [
        "| model | " + " | ".join(CATEGORIES) + " | overall |",
        "|" + ":------------------------------|" + "|".join(["----------"] * len(CATEGORIES)) + "|:--------|",
    ]
    for model in models:
        row = {"model": model}
        overall_pos = 0.0
        overall_total = 0.0
        for category in CATEGORIES:
            csv_path = directory / f"{model}_{category}_True_3_results.csv"
            ratio = compute_ratio(csv_path)
            row[category] = ratio
            if ratio is not None:
                df = pd.read_csv(csv_path, usecols=[REWRITE_COL, EXPLICIT_COL])
                total = len(df)
                both = ((df[REWRITE_COL] == 1) & (df[EXPLICIT_COL] == 1)).sum()
                overall_pos += both
                overall_total += total
        row["overall"] = (overall_pos / overall_total) if overall_total else None
        rows.append(row)
        rows.append(row)

        line = ["| " + model]
        for category in CATEGORIES:
            value = format_ratio(row[category])
            line.append(value)
        line.append(format_ratio(row["overall"]))
        lines.append(" | ".join(line) + " |")

    return pd.DataFrame(rows), "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate refusal ratios requiring both implicit rewrite filter and explicit filter."
    )
    parser.add_argument(
        "--directory",
        default="/disk1/users/wangjh/supp_exp/results/pipelines/or-bench-hard-1k/composed_pipeline",
        help="Directory containing composed pipeline CSV files.",
    )
    parser.add_argument(
        "--output",
        default="both_filter_results.md",
        help="Markdown file to write results to.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Optional list of model prefixes to include.",
    )
    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.is_dir():
        raise SystemExit(f"Directory not found: {directory}")

    table, markdown_text = build_table(directory, args.models)
    output_path = Path(args.output)
    output_path.write_text(markdown_text + "\n")
    print(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()

