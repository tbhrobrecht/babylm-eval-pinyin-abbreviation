#!/usr/bin/env python3
"""Create result tables and SVG visualisations from lm-eval summary files.

By default, this uses every JSON summary and the optional benchmark table in
summaries/. JSONL samples remain available with --results-dir.

Usage:
    python data_visualisation/plot_results.py
    python data_visualisation/plot_results.py --watch
    python data_visualisation/plot_results.py --results-dir results/main --run-policy all
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

from visualisation.build import BuildSummary, build_visualisations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise aggregated result summaries from summaries/.")
    parser.add_argument("--summaries-dir", type=Path, default=Path("summaries"), help="Directory containing summaries and benchmark data.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Use JSONL sample run folders instead of summary JSON files.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data_visualisation/output"), help="Where outputs are written.")
    parser.add_argument("--metric", default="acc", help="Numeric metric to aggregate, for example acc or acc_norm.")
    parser.add_argument(
        "--run-policy",
        choices=("latest", "all"),
        default="latest",
        help="For --results-dir only: use the newest run per model/task or aggregate all matching JSONL runs.",
    )
    parser.add_argument(
        "--zhoblimp-page-size",
        type=int,
        default=35,
        help="Number of zhoblimp subtasks per SVG heatmap page.",
    )
    parser.add_argument("--histogram-bins", type=int, default=20, help="Number of bins for zhoblimp histograms.")
    parser.add_argument("--watch", action="store_true", help="Refresh the graphs repeatedly as result files change.")
    parser.add_argument("--watch-interval", type=float, default=15.0, help="Seconds between refreshes in watch mode.")
    return parser.parse_args()


def run_once(args: argparse.Namespace) -> BuildSummary:
    source_dir = args.results_dir if args.results_dir is not None else args.summaries_dir
    source_format = "jsonl" if args.results_dir is not None else "summaries"
    summary = build_visualisations(
        source_dir=source_dir,
        source_format=source_format,
        output_dir=args.output_dir,
        metric=args.metric,
        run_policy=args.run_policy,
        zhoblimp_page_size=args.zhoblimp_page_size,
        histogram_bins=args.histogram_bins,
    )
    print_summary(summary, source_dir, source_format, args.run_policy)
    return summary


def print_summary(summary: BuildSummary, source_dir: Path, source_format: str, run_policy: str) -> None:
    policy_text = f", {run_policy} runs" if source_format == "jsonl" else ""
    print(f"Loaded {summary.raw_score_count} task scores from {source_dir} ({source_format}{policy_text})")
    print(f"Found {summary.model_count} model(s)")
    print(f"Added zhoblimp averages for {summary.zhoblimp_average_count} model(s)")
    print(f"Wrote {summary.zhoblimp_page_count} zhoblimp subtask figure(s)")
    print(f"Wrote outputs to {summary.output_dir}")


def input_signature(source_dir: Path, source_format: str) -> tuple[tuple[str, int, int], ...]:
    signature = []
    if source_format == "jsonl":
        paths = sorted(source_dir.glob("*/*.jsonl"))
    else:
        paths = sorted(source_dir.glob("*.json"))
        benchmark_path = source_dir / "benchmark"
        if benchmark_path.exists():
            paths.append(benchmark_path)
    for path in paths:
        stat = path.stat()
        signature.append((path.as_posix(), stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def watch(args: argparse.Namespace) -> None:
    source_dir = args.results_dir if args.results_dir is not None else args.summaries_dir
    source_format = "jsonl" if args.results_dir is not None else "summaries"
    print(f"Watching {source_dir} every {args.watch_interval:g}s. Press Ctrl+C to stop.")
    last_signature: tuple[tuple[str, int, int], ...] | None = None
    while True:
        signature = input_signature(source_dir, source_format)
        if signature != last_signature:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Results changed; rebuilding graphs.")
            run_once(args)
            last_signature = signature
        time.sleep(args.watch_interval)


def main() -> None:
    args = parse_args()
    if args.watch:
        watch(args)
    else:
        run_once(args)


if __name__ == "__main__":
    main()
