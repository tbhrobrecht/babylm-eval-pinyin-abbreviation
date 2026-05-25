"""Publication-style Matplotlib figures for result summaries."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
from scipy.stats import gaussian_kde

from .loader import TaskScore
from .model_labels import display_model_name, model_sort_key


PALETTE = [
    "#1b4965",
    "#2a9d8f",
    "#e76f51",
    "#6d597a",
    "#d4a373",
    "#457b9d",
    "#8ab17d",
    "#bc4749",
]
HEATMAP_CMAP = LinearSegmentedColormap.from_list("academic_blue", ["#f7fbff", "#9ecae1", "#2171b5", "#08306b"])
PREFERRED_TASK_ORDER = [
    "hellaswag_zh_mubench",
    "winogrande_zh_mubench",
    "xcomps_zh",
    "xstorycloze_zh_mubench",
    "zhoblimp",
    "zeroshot_zho",
]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "axes.edgecolor": "#4b5563",
            "axes.linewidth": 0.8,
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "svg.hashsalt": "babylm-visualisations",
        }
    )


def model_order(models: set[str]) -> list[str]:
    return sorted(models, key=model_sort_key)


def pretty_model(model: str) -> str:
    return display_model_name(model)


def pretty_task(task: str) -> str:
    labels = {
        "hellaswag_zh_mubench": "HellaSwag",
        "winogrande_zh_mubench": "WinoGrande",
        "xcomps_zh": "XCOMPS",
        "xstorycloze_zh_mubench": "XStoryCloze",
        "zhoblimp": "ZhoBLiMP",
        "zeroshot_zho": "Macro average",
    }
    return labels.get(task, task.replace("zhoblimp_", "").replace("_", " "))


def save_figure(fig: plt.Figure, output_path: Path, title: str) -> None:
    fig.savefig(
        output_path,
        format="svg",
        bbox_inches="tight",
        metadata={"Title": title, "Creator": "Matplotlib", "Date": None},
    )
    fig.savefig(
        output_path.with_suffix(".png"),
        format="png",
        dpi=300,
        bbox_inches="tight",
        metadata={"Title": title, "Software": "Matplotlib"},
    )
    plt.close(fig)


def style_percentage_axis(ax: plt.Axes) -> None:
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.grid(axis="y", color="#d1d5db", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def task_order_for(scores: list[TaskScore]) -> list[str]:
    tasks = {score.task for score in scores}
    preferred = [task for task in PREFERRED_TASK_ORDER if task in tasks]
    return preferred + sorted(tasks - set(preferred))


def write_grouped_bar_svg(
    scores: list[TaskScore],
    output_path: Path,
    title: str,
    task_order: list[str] | None = None,
) -> None:
    if not scores:
        return

    apply_style()
    models = model_order({score.model for score in scores})
    tasks = task_order or task_order_for(scores)
    score_lookup = {(score.model, score.task): score for score in scores}
    x = np.arange(len(tasks))
    width = 0.84 / len(models)
    fig, ax = plt.subplots(figsize=(12.8, 6.4), constrained_layout=True)

    for index, model in enumerate(models):
        values = []
        errors = []
        for task in tasks:
            score = score_lookup.get((model, task))
            values.append(np.nan if score is None else score.mean_score * 100)
            errors.append(0.0 if score is None else score.stderr * 100)
        positions = x - 0.42 + width / 2 + index * width
        ax.bar(
            positions,
            values,
            width=width * 0.92,
            yerr=errors,
            capsize=2,
            error_kw={"elinewidth": 0.65, "capthick": 0.65, "ecolor": "#374151"},
            color=PALETTE[index % len(PALETTE)],
            edgecolor="white",
            linewidth=0.45,
            label=pretty_model(model),
        )

    style_percentage_axis(ax)
    ax.set_title(title, pad=16)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x, [pretty_task(task) for task in tasks])
    ax.tick_params(axis="x", labelrotation=0, pad=8)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        fontsize=9,
        columnspacing=1.4,
        handlelength=1.4,
    )
    ax.text(
        0,
        -0.26,
        "Error bars indicate one standard error where available; the published benchmark reports point estimates only.",
        transform=ax.transAxes,
        color="#4b5563",
        fontsize=8.5,
    )
    save_figure(fig, output_path, title)


def paginate(items: list[str], page_size: int) -> list[list[str]]:
    return [items[index : index + page_size] for index in range(0, len(items), page_size)]


def write_paginated_heatmap_svgs(
    scores: list[TaskScore],
    output_dir: Path,
    filename_prefix: str,
    title: str,
    page_size: int,
) -> list[Path]:
    if not scores:
        return []

    apply_style()
    models = model_order({score.model for score in scores})
    tasks = sorted(
        {score.task for score in scores},
        key=lambda task: (-mean(score.mean_score for score in scores if score.task == task), task),
    )
    score_lookup = {(score.model, score.task): score.mean_score for score in scores}
    pages = paginate(tasks, page_size)
    outputs: list[Path] = []

    for page_number, page_tasks in enumerate(pages, start=1):
        matrix = np.array([[score_lookup.get((model, task), np.nan) for task in page_tasks] for model in models])
        width = max(11.5, len(page_tasks) * 0.33)
        fig, ax = plt.subplots(figsize=(width, 4.6), constrained_layout=True)
        image = ax.imshow(matrix, aspect="auto", cmap=HEATMAP_CMAP, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"{title} ({page_number}/{len(pages)})", pad=15)
        ax.set_xlabel("Subtask (ranked by mean accuracy across models)")
        ax.set_yticks(range(len(models)), [pretty_model(model) for model in models])
        labels = [textwrap.shorten(pretty_task(task), width=25, placeholder="...") for task in page_tasks]
        ax.set_xticks(range(len(page_tasks)), labels, rotation=58, ha="right", rotation_mode="anchor", fontsize=8)
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
        colorbar.set_label("Accuracy")
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        suffix = f"_{page_number:02d}" if len(pages) > 1 else ""
        output_path = output_dir / f"{filename_prefix}{suffix}.svg"
        save_figure(fig, output_path, title)
        outputs.append(output_path)
    return outputs


def write_stacked_heatmap_svg(
    scores: list[TaskScore],
    output_path: Path,
    title: str,
    page_size: int,
) -> None:
    if not scores:
        return

    apply_style()
    models = model_order({score.model for score in scores})
    tasks = sorted(
        {score.task for score in scores},
        key=lambda task: (-mean(score.mean_score for score in scores if score.task == task), task),
    )
    score_lookup = {(score.model, score.task): score.mean_score for score in scores}
    pages = paginate(tasks, page_size)
    width = max(11.5, min(page_size, len(tasks)) * 0.33)
    height = 3.65 * len(pages) + 1.0
    fig, axes = plt.subplots(len(pages), 1, figsize=(width, height), constrained_layout=True, squeeze=False)
    images = []

    for ax, page_tasks in zip(axes[:, 0], pages):
        matrix = np.array([[score_lookup.get((model, task), np.nan) for task in page_tasks] for model in models])
        images.append(ax.imshow(matrix, aspect="auto", cmap=HEATMAP_CMAP, vmin=0, vmax=1, interpolation="nearest"))
        ax.set_yticks(range(len(models)), [pretty_model(model) for model in models])
        labels = [textwrap.shorten(pretty_task(task), width=25, placeholder="...") for task in page_tasks]
        ax.set_xticks(range(len(page_tasks)), labels, rotation=58, ha="right", rotation_mode="anchor", fontsize=8)
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="semibold")
    axes[-1, 0].set_xlabel("Subtask (ranked by mean accuracy across models)", labelpad=10)
    colorbar = fig.colorbar(images[0], ax=list(axes[:, 0]), fraction=0.02, pad=0.02)
    colorbar.set_label("Accuracy")
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    save_figure(fig, output_path, title)


def write_histogram_svg(values_by_model: dict[str, list[float]], output_path: Path, title: str, bins: int) -> None:
    if not values_by_model:
        return

    apply_style()
    models = model_order(set(values_by_model))
    columns = 2
    rows = math.ceil(len(models) / columns)
    fig = plt.figure(figsize=(11.5, 2.15 * rows + 1.0), constrained_layout=True)
    grid = fig.add_gridspec(rows, columns)
    axes: list[plt.Axes] = []
    for index in range(len(models)):
        if len(models) % columns and index == len(models) - 1:
            position = grid[rows - 1, :]
        else:
            position = grid[index // columns, index % columns]
        shared_axes = {"sharex": axes[0], "sharey": axes[0]} if axes else {}
        axes.append(fig.add_subplot(position, **shared_axes))
    bin_edges = np.linspace(0, 100, bins + 1)

    for index, (ax, model) in enumerate(zip(axes, models)):
        values = np.asarray(values_by_model[model]) * 100
        mean_value = np.mean(values)
        ax.hist(values, bins=bin_edges, color=PALETTE[index % len(PALETTE)], alpha=0.84, edgecolor="white", linewidth=0.45)
        ax.axvline(mean_value, color="#111827", linestyle="--", linewidth=1)
        ax.set_title(pretty_model(model), fontsize=10, loc="left", pad=5)
        ax.annotate(
            f"Mean: {mean_value:.1f}%",
            xy=(mean_value, 0.88),
            xycoords=ax.get_xaxis_transform(),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
        )
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.65)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if index >= (rows - 1) * columns:
            ax.set_xlabel("Accuracy (%)")
        if index % columns == 0:
            ax.set_ylabel("Subtasks")
    fig.suptitle(title, fontsize=14, fontweight="semibold")
    save_figure(fig, output_path, title)


def write_kde_svg(values_by_model: dict[str, list[float]], output_path: Path, title: str) -> None:
    if not values_by_model:
        return

    apply_style()
    models = model_order(set(values_by_model))
    x = np.linspace(0, 100, 401)
    fig, ax = plt.subplots(figsize=(11.5, 6.2), constrained_layout=True)
    for index, model in enumerate(models):
        values = np.asarray(values_by_model[model]) * 100
        density = gaussian_kde(values)(x)
        color = PALETTE[index % len(PALETTE)]
        ax.plot(x, density, color=color, linewidth=2, label=pretty_model(model))

    ax.set_title(title, pad=16)
    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.grid(axis="y", color="#d1d5db", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=9)
    save_figure(fig, output_path, title)
