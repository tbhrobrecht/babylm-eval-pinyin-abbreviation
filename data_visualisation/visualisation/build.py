"""Build all result tables and plots."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .loader import TaskScore, add_zhoblimp_average, load_scores
from .svg_plots import write_grouped_bar_svg, write_histogram_svg, write_kde_svg, write_paginated_grouped_bar_svgs
from .tables import write_model_summary_csv, write_task_scores_csv


@dataclass(frozen=True)
class BuildSummary:
    raw_score_count: int
    model_count: int
    zhoblimp_average_count: int
    zhoblimp_page_count: int
    output_dir: Path


def build_visualisations(
    results_dir: Path,
    output_dir: Path,
    metric: str,
    run_policy: str,
    zhoblimp_page_size: int,
    histogram_bins: int,
) -> BuildSummary:
    scores = load_scores(results_dir, metric, run_policy=run_policy)
    if not scores:
        raise SystemExit(f"No '{metric}' values found in JSONL files under {results_dir}")

    scores_with_zhoblimp_average = add_zhoblimp_average(scores)
    broad_scores, zhoblimp_scores = split_scores(scores_with_zhoblimp_average)
    raw_zhoblimp_scores = [score for score in scores if score.task.startswith("zhoblimp_")]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_task_scores_csv(scores_with_zhoblimp_average, output_dir / "task_scores.csv")
    write_model_summary_csv(scores_with_zhoblimp_average, output_dir / "model_summary.csv")

    write_grouped_bar_svg(broad_scores, output_dir / "task_scores_by_model.svg", f"{metric} by task")
    remove_stale_zhoblimp_pages(output_dir)
    zhoblimp_pages = write_paginated_grouped_bar_svgs(
        zhoblimp_scores,
        output_dir,
        "zhoblimp_scores_by_model",
        f"{metric} by zhoblimp subtask",
        zhoblimp_page_size,
    )

    zhoblimp_values = values_by_model(raw_zhoblimp_scores)
    write_histogram_svg(
        zhoblimp_values,
        output_dir / "zhoblimp_accuracy_histogram.svg",
        f"{metric} distribution across zhoblimp subtasks",
        histogram_bins,
    )
    write_kde_svg(
        zhoblimp_values,
        output_dir / "zhoblimp_accuracy_kde.svg",
        f"{metric} KDE across zhoblimp subtasks",
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

