"""CSV writers for aggregated result data."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .loader import TaskScore


def write_task_scores_csv(scores: list[TaskScore], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "task", "metric", "mean_score", "stderr", "sample_count", "source_file"],
        )
        writer.writeheader()
        for score in sorted(scores, key=lambda item: (item.model, item.task, item.metric)):
            writer.writerow(
                {
                    "model": score.model,
                    "task": score.task,
                    "metric": score.metric,
                    "mean_score": f"{score.mean_score:.6f}",
                    "stderr": f"{score.stderr:.6f}",
                    "sample_count": score.sample_count,
                    "source_file": ";".join(path.as_posix() for path in score.source_files),
                }
            )


def write_model_summary_csv(scores: list[TaskScore], output_path: Path) -> None:
    by_model: dict[str, list[TaskScore]] = defaultdict(list)
    for score in scores:
        by_model[score.model].append(score)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "metric", "mean_score", "task_count", "sample_count"])
        writer.writeheader()
        for model, model_scores in sorted(by_model.items()):
            writer.writerow(
                {
                    "model": model,
                    "metric": model_scores[0].metric,
                    "mean_score": f"{mean(score.mean_score for score in model_scores):.6f}",
                    "task_count": len(model_scores),
                    "sample_count": sum(score.sample_count for score in model_scores),
                }
            )

