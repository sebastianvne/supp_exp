#!/usr/bin/env python3
"""
Plot heatmap and bar chart from implicit_rewrite_results.md.
"""

"""
Usage:
python scripts/plot_implicit_ratios.py \
  --markdown explicit_filter_results.md \
  --output-dir plots \
  --heatmap-output plots/explicit_heatmap.png \
  --heatmap-annot-output plots/explicit_heatmap_numbers.png \
  --bar-output plots/explicit_overall.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_markdown_table(md_path: Path) -> pd.DataFrame:
    rows = []
    with md_path.open() as f:
        for line in f:
            stripped = line.strip()
            if not stripped.startswith("|") or stripped.startswith("|:"):
                continue
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            rows.append(cells)

    if len(rows) < 2:
        raise ValueError("Markdown table must include header and at least one row.")

    header = rows[0]
    data = rows[1:]
    df = pd.DataFrame(data, columns=header).replace("", float("nan"))

    def to_float(value):
        if isinstance(value, str):
            stripped = value.strip()
            if stripped in {"-", ""}:
                return float("nan")
            if stripped.endswith("%"):
                stripped = stripped[:-1]
            try:
                return float(stripped)
            except ValueError:
                return float("nan")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    for column in df.columns[1:]:
        df[column] = df[column].map(to_float)
    return df


def plot_heatmap(df: pd.DataFrame, output: Path, annotate: bool = False) -> None:
    categories = df.columns[1:-1]
    data = df[categories].values

    plt.figure(figsize=(12, 4))
    heatmap = plt.imshow(data, cmap="viridis", aspect="auto")
    plt.xticks(range(len(categories)), categories, rotation=45, ha="right")
    plt.yticks(range(len(df["model"])), df["model"])
    cbar = plt.colorbar(heatmap)
    cbar.set_label("Rewrite filter ratio (%)")
    plt.title("Implicit rewrite filter ratios by model/category")

    if annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data[i, j]
                if pd.isna(value):
                    continue
                plt.text(
                    j,
                    i,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    color="white" if value > 50 else "black",
                    fontsize=8,
                )

    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def plot_overall_bar(df: pd.DataFrame, output: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.barh(df["model"], df["overall"], color="#1f77b4")
    plt.xlabel("Overall rewrite filter ratio (%)")
    plt.title("Implicit rewrite overall ratios")
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot implicit rewrite ratios.")
    parser.add_argument(
        "--markdown",
        default="implicit_rewrite_results.md",
        help="Path to the Markdown table file.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory to store generated figures.",
    )
    parser.add_argument(
        "--heatmap-output",
        default=None,
        help="Explicit path for the heatmap image (default: OUTPUT_DIR/implicit_rewrite_heatmap.png).",
    )
    parser.add_argument(
        "--heatmap-annot-output",
        default=None,
        help="Path for heatmap with numeric annotations (default: OUTPUT_DIR/implicit_rewrite_heatmap_values.png).",
    )
    parser.add_argument(
        "--bar-output",
        default=None,
        help="Explicit path for the overall bar chart (default: OUTPUT_DIR/implicit_rewrite_overall.png).",
    )
    args = parser.parse_args()

    md_path = Path(args.markdown)
    if not md_path.is_file():
        raise SystemExit(f"Markdown file not found: {md_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_markdown_table(md_path)
    heatmap_path = Path(args.heatmap_output) if args.heatmap_output else output_dir / "implicit_rewrite_heatmap.png"
    heatmap_annot_path = (
        Path(args.heatmap_annot_output)
        if args.heatmap_annot_output
        else output_dir / "implicit_rewrite_heatmap_values.png"
    )
    bar_path = Path(args.bar_output) if args.bar_output else output_dir / "implicit_rewrite_overall.png"

    plot_heatmap(df, heatmap_path, annotate=False)
    plot_heatmap(df, heatmap_annot_path, annotate=True)
    plot_overall_bar(df, bar_path)
    print(f"Saved heatmap to {heatmap_path}")
    print(f"Saved annotated heatmap to {heatmap_annot_path}")
    print(f"Saved overall bar chart to {bar_path}")


if __name__ == "__main__":
    main()

