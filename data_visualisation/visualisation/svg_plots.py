"""SVG plotting helpers for result summaries."""

from __future__ import annotations

import html
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .loader import TaskScore


COLORS = ["#2f6f9f", "#c44e52", "#55a868", "#8172b3", "#ccb974", "#64b5cd"]


def shorten_label(label: str, max_length: int = 34) -> str:
    if len(label) <= max_length:
        return label
    return label[: max_length - 1] + "..."


def paginate(items: list[str], page_size: int) -> list[list[str]]:
    return [items[index : index + page_size] for index in range(0, len(items), page_size)]


def write_grouped_bar_svg(
    scores: list[TaskScore],
    output_path: Path,
    title: str,
    task_order: list[str] | None = None,
) -> None:
    if not scores:
        return

    models = sorted({score.model for score in scores})
    if task_order is None:
        task_order = sorted(
            {score.task for score in scores},
            key=lambda task: mean(score.mean_score for score in scores if score.task == task),
            reverse=True,
        )

    score_lookup = {(score.model, score.task): score.mean_score for score in scores}
    width = max(980, 120 + len(task_order) * max(42, len(models) * 22))
    height = 620
    margin_left = 76
    margin_right = 28
    margin_top = 72
    margin_bottom = 170
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    group_width = chart_width / max(1, len(task_order))
    bar_width = min(22, group_width / max(1, len(models)) * 0.75)

    lines = svg_header(width, height, title, margin_left)
    add_y_axis(lines, margin_left, margin_right, margin_top, chart_height, width)

    for task_index, task in enumerate(task_order):
        center_x = margin_left + task_index * group_width + group_width / 2
        for model_index, model in enumerate(models):
            value = score_lookup.get((model, task))
            if value is None:
                continue
            x = center_x - (len(models) * bar_width) / 2 + model_index * bar_width
            bar_height = max(0, min(1, value)) * chart_height
            y = margin_top + chart_height - bar_height
            color = COLORS[model_index % len(COLORS)]
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 2:.1f}" height="{bar_height:.1f}" fill="{color}">'
                f"<title>{html.escape(model)} | {html.escape(task)}: {value:.3f}</title></rect>"
            )
        label = html.escape(shorten_label(task))
        lines.append(
            f'<text x="{center_x:.1f}" y="{margin_top + chart_height + 16}" transform="rotate(62 {center_x:.1f} {margin_top + chart_height + 16})" class="label">{label}</text>'
        )

    add_legend(lines, models, margin_left, height)
    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paginated_grouped_bar_svgs(
    scores: list[TaskScore],
    output_dir: Path,
    filename_prefix: str,
    title: str,
    page_size: int,
) -> list[Path]:
    task_order = sorted(
        {score.task for score in scores},
        key=lambda task: mean(score.mean_score for score in scores if score.task == task),
        reverse=True,
    )
    outputs: list[Path] = []
    pages = paginate(task_order, page_size)
    for page_number, page_tasks in enumerate(pages, start=1):
        suffix = f"_{page_number:02d}" if len(pages) > 1 else ""
        output_path = output_dir / f"{filename_prefix}{suffix}.svg"
        page_scores = [score for score in scores if score.task in page_tasks]
        write_grouped_bar_svg(page_scores, output_path, f"{title} ({page_number}/{len(pages)})", page_tasks)
        outputs.append(output_path)
    return outputs


def write_histogram_svg(values_by_model: dict[str, list[float]], output_path: Path, title: str, bins: int) -> None:
    width = 980
    height = 560
    margin_left = 76
    margin_right = 28
    margin_top = 72
    margin_bottom = 72
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    bin_width = 1 / bins
    histograms = {model: histogram(values, bins) for model, values in values_by_model.items()}
    max_count = max((max(counts) for counts in histograms.values() if counts), default=1)

    lines = svg_header(width, height, title, margin_left)
    add_count_axis(lines, margin_left, margin_right, margin_top, chart_height, width, max_count)
    models = sorted(values_by_model)
    cluster_width = chart_width / bins
    bar_width = cluster_width / max(1, len(models)) * 0.78

    for bin_index in range(bins):
        for model_index, model in enumerate(models):
            count = histograms[model][bin_index]
            x = margin_left + bin_index * cluster_width + model_index * bar_width
            bar_height = count / max_count * chart_height if max_count else 0
            y = margin_top + chart_height - bar_height
            color = COLORS[model_index % len(COLORS)]
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 2:.1f}" height="{bar_height:.1f}" fill="{color}" opacity="0.82">'
                f"<title>{html.escape(model)} | {bin_index * bin_width:.2f}-{(bin_index + 1) * bin_width:.2f}: {count}</title></rect>"
            )

    add_x_axis_ticks(lines, margin_left, margin_top + chart_height, chart_width)
    add_legend(lines, models, margin_left, height)
    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_kde_svg(values_by_model: dict[str, list[float]], output_path: Path, title: str) -> None:
    width = 980
    height = 560
    margin_left = 76
    margin_right = 28
    margin_top = 72
    margin_bottom = 72
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    models = sorted(values_by_model)
    curves = {model: kde_curve(values) for model, values in values_by_model.items()}
    max_density = max((density for curve in curves.values() for _, density in curve), default=1.0)

    lines = svg_header(width, height, title, margin_left)
    add_density_axis(lines, margin_left, margin_right, margin_top, chart_height, width, max_density)

    for model_index, model in enumerate(models):
        color = COLORS[model_index % len(COLORS)]
        points = []
        for x_value, density in curves[model]:
            x = margin_left + x_value * chart_width
            y = margin_top + chart_height - (density / max_density * chart_height if max_density else 0)
            points.append(f"{x:.1f},{y:.1f}")
        lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5">'
            f"<title>{html.escape(model)}</title></polyline>"
        )

    add_x_axis_ticks(lines, margin_left, margin_top + chart_height, chart_width)
    add_legend(lines, models, margin_left, height)
    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def histogram(values: list[float], bins: int) -> list[int]:
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, max(0, int(value * bins)))
        counts[index] += 1
    return counts


def kde_curve(values: list[float], points: int = 160) -> list[tuple[float, float]]:
    if not values:
        return []
    if len(values) == 1:
        bandwidth = 0.08
    else:
        avg = mean(values)
        std = math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))
        bandwidth = max(0.04, 1.06 * std * (len(values) ** -0.2))

    curve: list[tuple[float, float]] = []
    scale = 1 / (len(values) * bandwidth * math.sqrt(2 * math.pi))
    for index in range(points):
        x = index / (points - 1)
        density = scale * sum(math.exp(-0.5 * ((x - value) / bandwidth) ** 2) for value in values)
        curve.append((x, density))
    return curve


def svg_header(width: int, height: int, title: str, margin_left: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;fill:#1f2933}.axis{stroke:#9aa5b1;stroke-width:1}.grid{stroke:#e4e7eb;stroke-width:1}.label{font-size:12px}.title{font-size:22px;font-weight:700}.legend{font-size:13px}</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="34" class="title">{html.escape(title)}</text>',
    ]


def add_y_axis(lines: list[str], margin_left: int, margin_right: int, margin_top: int, chart_height: int, width: int) -> None:
    for tick in range(0, 6):
        value = tick / 5
        y = margin_top + chart_height - value * chart_height
        lines.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" class="label">{value:.1f}</text>')
    add_axes(lines, margin_left, margin_right, margin_top, chart_height, width)


def add_count_axis(
    lines: list[str], margin_left: int, margin_right: int, margin_top: int, chart_height: int, width: int, max_count: int
) -> None:
    for tick in range(0, 6):
        value = round(max_count * tick / 5)
        y = margin_top + chart_height - tick / 5 * chart_height
        lines.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" class="label">{value}</text>')
    add_axes(lines, margin_left, margin_right, margin_top, chart_height, width)


def add_density_axis(
    lines: list[str], margin_left: int, margin_right: int, margin_top: int, chart_height: int, width: int, max_density: float
) -> None:
    for tick in range(0, 6):
        value = max_density * tick / 5
        y = margin_top + chart_height - tick / 5 * chart_height
        lines.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" class="label">{value:.1f}</text>')
    add_axes(lines, margin_left, margin_right, margin_top, chart_height, width)


def add_axes(lines: list[str], margin_left: int, margin_right: int, margin_top: int, chart_height: int, width: int) -> None:
    lines.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_height}" class="axis"/>')
    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top + chart_height}" x2="{width - margin_right}" y2="{margin_top + chart_height}" class="axis"/>'
    )


def add_x_axis_ticks(lines: list[str], margin_left: int, y: int, chart_width: int) -> None:
    for tick in range(0, 6):
        value = tick / 5
        x = margin_left + value * chart_width
        lines.append(f'<line x1="{x:.1f}" y1="{y}" x2="{x:.1f}" y2="{y + 5}" class="axis"/>')
        lines.append(f'<text x="{x:.1f}" y="{y + 22}" text-anchor="middle" class="label">{value:.1f}</text>')


def add_legend(lines: list[str], models: list[str], margin_left: int, height: int) -> None:
    legend_y = height - 28
    for model_index, model in enumerate(models):
        x = margin_left + model_index * 210
        color = COLORS[model_index % len(COLORS)]
        lines.append(f'<rect x="{x}" y="{legend_y - 12}" width="14" height="14" fill="{color}"/>')
        lines.append(f'<text x="{x + 20}" y="{legend_y}" class="legend">{html.escape(model)}</text>')
