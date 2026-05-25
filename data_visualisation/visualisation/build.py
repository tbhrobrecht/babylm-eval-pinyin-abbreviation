"""Build all result tables and plots."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .loader import TaskScore, add_zhoblimp_average, load_scores, load_summary_scores
from .matplotlib_plots import (
    write_grouped_bar_svg,
    write_histogram_svg,
    write_kde_svg,
    write_paginated_heatmap_svgs,
    write_stacked_heatmap_svg,
)
from .tables import write_model_summary_csv, write_task_scores_csv


@dataclass(frozen=True)
class BuildSummary:
    raw_score_count: int
    model_count: int
    zhoblimp_average_count: int
    zhoblimp_page_count: int
    output_dir: Path


def build_visualisations(
    source_dir: Path,
    source_format: str,
    output_dir: Path,
    metric: str,
    run_policy: str,
    zhoblimp_page_size: int,
    histogram_bins: int,
) -> BuildSummary:
    if source_format == "summaries":
        scores = load_summary_scores(source_dir, metric)
    elif source_format == "jsonl":
        scores = load_scores(source_dir, metric, run_policy=run_policy)
    else:
        raise ValueError(f"Unknown source format: {source_format}")
    if not scores:
        raise SystemExit(f"No '{metric}' values found in {source_format} files under {source_dir}")

    scores_with_zhoblimp_average = add_zhoblimp_average(scores)
    broad_scores, zhoblimp_scores = split_scores(scores_with_zhoblimp_average)
    broad_scores = [score for score in broad_scores if score.task != "zeroshot_zho"]
    raw_zhoblimp_scores = [score for score in scores if score.task.startswith("zhoblimp_")]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_task_scores_csv(scores_with_zhoblimp_average, output_dir / "task_scores.csv")
    write_model_summary_csv(scores_with_zhoblimp_average, output_dir / "model_summary.csv")

    write_grouped_bar_svg(broad_scores, output_dir / "task_scores_by_model.svg", "Chinese Evaluation Task Performance")
    remove_stale_zhoblimp_pages(output_dir)
    zhoblimp_pages = write_paginated_heatmap_svgs(
        zhoblimp_scores,
        output_dir,
        "zhoblimp_scores_by_model",
        "ZhoBLiMP Subtask Accuracy",
        zhoblimp_page_size,
    )
    write_stacked_heatmap_svg(
        zhoblimp_scores,
        output_dir / "zhoblimp_scores_by_model_stacked.svg",
        "ZhoBLiMP Subtask Accuracy",
        zhoblimp_page_size,
    )

    zhoblimp_values = values_by_model(raw_zhoblimp_scores)
    write_histogram_svg(
        zhoblimp_values,
        output_dir / "zhoblimp_accuracy_histogram.svg",
        "Distribution of ZhoBLiMP Subtask Accuracy",
        histogram_bins,
    )
    write_kde_svg(
        zhoblimp_values,
        output_dir / "zhoblimp_accuracy_kde.svg",
        "Smoothed Distribution of ZhoBLiMP Subtask Accuracy",
    )

    return BuildSummary(
        raw_score_count=len(scores),
        model_count=len({score.model for score in scores}),
        zhoblimp_average_count=len(scores_with_zhoblimp_average) - len(scores),
        zhoblimp_page_count=len(zhoblimp_pages),
        output_dir=output_dir,
    )


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
    for path in output_dir.glob("zhoblimp_scores_by_model*.png"):
        path.unlink()

