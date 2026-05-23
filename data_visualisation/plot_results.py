#!/usr/bin/env python3
"""Create result tables and SVG visualisations from lm-eval JSONL files.

Usage:
    python data_visualisation/plot_results.py
    python data_visualisation/plot_results.py --metric acc_norm
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from visualisation.loader import TaskScore, add_zhoblimp_average, load_scores
from visualisation.svg_plots import (
    write_grouped_bar_svg,
    write_histogram_svg,
    write_kde_svg,
    write_paginated_grouped_bar_svgs,
)
from visualisation.tables import write_model_summary_csv, write_task_scores_csv


def split_scores(scores: list[TaskScore]) -> tuple[list[TaskScore], list[TaskScore]]:
    broad_scores = [score for score in scores if not score.task.startswith("zhoblimp_")]
    zhoblimp_scores = [score for score in scores if score.task.startswith("zhoblimp_")]
    return broad_scores, zhoblimp_scores


def values_by_model(scores: list[TaskScore]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for score in scores:
        grouped[score.model].append(score.mean_score)
    return dict(grouped)


def remove_stale_zhoblimp_pages(output_dir: Path) -> None:
    for path in output_dir.glob("zhoblimp_scores_by_model*.svg"):
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise JSONL sample results from results/main/.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/main"), help="Directory containing run folders.")
    parser.add_argument("--output-dir", type=Path, default=Path("data_visualisation/output"), help="Where outputs are written.")
    parser.add_argument("--metric", default="acc", help="Numeric JSONL metric to aggregate, for example acc or acc_norm.")
    parser.add_argument(
        "--zhoblimp-page-size",
        type=int,
        default=35,
        help="Number of zhoblimp subtasks per SVG bar-chart page.",
    )
    parser.add_argument("--histogram-bins", type=int, default=20, help="Number of bins for zhoblimp histograms.")
    args = parser.parse_args()

    scores = load_scores(args.results_dir, args.metric)
    if not scores:
        raise SystemExit(f"No '{args.metric}' values found in JSONL files under {args.results_dir}")

    scores_with_zhoblimp_average = add_zhoblimp_average(scores)
    broad_scores, zhoblimp_scores = split_scores(scores_with_zhoblimp_average)
    raw_zhoblimp_scores = [score for score in scores if score.task.startswith("zhoblimp_")]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_task_scores_csv(scores_with_zhoblimp_average, args.output_dir / "task_scores.csv")
    write_model_summary_csv(scores_with_zhoblimp_average, args.output_dir / "model_summary.csv")

    write_grouped_bar_svg(broad_scores, args.output_dir / "task_scores_by_model.svg", f"{args.metric} by task")
    remove_stale_zhoblimp_pages(args.output_dir)
    zhoblimp_pages = write_paginated_grouped_bar_svgs(
        zhoblimp_scores,
        args.output_dir,
        "zhoblimp_scores_by_model",
        f"{args.metric} by zhoblimp subtask",
        args.zhoblimp_page_size,
    )

    zhoblimp_values = values_by_model(raw_zhoblimp_scores)
    write_histogram_svg(
        zhoblimp_values,
        args.output_dir / "zhoblimp_accuracy_histogram.svg",
        f"{args.metric} distribution across zhoblimp subtasks",
        args.histogram_bins,
    )
    write_kde_svg(
        zhoblimp_values,
        args.output_dir / "zhoblimp_accuracy_kde.svg",
        f"{args.metric} KDE across zhoblimp subtasks",
    )

    print(f"Loaded {len(scores)} raw task scores from {args.results_dir}")
    print(f"Added zhoblimp averages for {len(scores_with_zhoblimp_average) - len(scores)} model(s)")
    print(f"Wrote {len(zhoblimp_pages)} zhoblimp subtask figure(s)")
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
