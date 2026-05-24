"""Load and aggregate lm-eval JSONL sample files."""

from __future__ import annotations

import math
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


TIMESTAMP_RE = re.compile(r"_(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)$")


@dataclass(frozen=True)
class TaskScore:
    model: str
    task: str
    metric: str
    mean_score: float
    stderr: float
    sample_count: int
    source_files: tuple[Path, ...]


def task_name_from_sample_file(path: Path) -> str:
    stem = path.stem
    if stem.startswith("samples_"):
        stem = stem[len("samples_") :]
    return TIMESTAMP_RE.sub("", stem)


def timestamp_from_sample_file(path: Path) -> str:
    match = TIMESTAMP_RE.search(path.stem)
    if match:
        return match.group("timestamp")
    return f"{path.stat().st_mtime:.6f}"


def model_name_from_folder(folder: Path) -> str:
    return folder.name.split("__")[-1]


def standard_error(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance) / math.sqrt(len(values))


def discover_sample_files(results_dir: Path, run_policy: str) -> list[Path]:
    sample_files = sorted(results_dir.glob("*/*.jsonl"))
    if run_policy == "all":
        return sample_files
    if run_policy != "latest":
        raise ValueError(f"Unknown run policy: {run_policy}")

    latest_by_task: dict[tuple[str, str], Path] = {}
    latest_timestamp: dict[tuple[str, str], str] = {}
    for sample_file in sample_files:
        model = model_name_from_folder(sample_file.parent)
        task = task_name_from_sample_file(sample_file)
        key = (model, task)
        timestamp = timestamp_from_sample_file(sample_file)
        if key not in latest_by_task or timestamp > latest_timestamp[key]:
            latest_by_task[key] = sample_file
            latest_timestamp[key] = timestamp
    return sorted(latest_by_task.values())


def load_scores(results_dir: Path, metric: str, run_policy: str = "latest") -> list[TaskScore]:
    grouped_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    grouped_files: dict[tuple[str, str], list[Path]] = defaultdict(list)

    for sample_file in discover_sample_files(results_dir, run_policy):
        values: list[float] = []
        with sample_file.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {sample_file}:{line_number}: {exc}") from exc
                if metric in row and isinstance(row[metric], (int, float)):
                    values.append(float(row[metric]))

        if not values:
            continue

        model = model_name_from_folder(sample_file.parent)
        task = task_name_from_sample_file(sample_file)
        key = (model, task)
        grouped_values[key].extend(values)
        grouped_files[key].append(sample_file)

    scores: list[TaskScore] = []
    for (model, task), values in grouped_values.items():
        scores.append(
            TaskScore(
                model=model,
                task=task,
                metric=metric,
                mean_score=mean(values),
                stderr=standard_error(values),
                sample_count=len(values),
                source_files=tuple(grouped_files[(model, task)]),
            )
        )
    return scores


def add_zhoblimp_average(scores: list[TaskScore]) -> list[TaskScore]:
    by_model: dict[str, list[TaskScore]] = defaultdict(list)
    for score in scores:
        if score.task.startswith("zhoblimp_"):
            by_model[score.model].append(score)

    averages: list[TaskScore] = []
    for model, model_scores in sorted(by_model.items()):
        values = [score.mean_score for score in model_scores]
        averages.append(
            TaskScore(
                model=model,
                task="zhoblimp",
                metric=model_scores[0].metric,
                mean_score=mean(values),
                stderr=standard_error(values),
                sample_count=sum(score.sample_count for score in model_scores),
                source_files=(),
            )
        )
    return scores + averages
