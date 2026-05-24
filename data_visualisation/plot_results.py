#!/usr/bin/env python3
"""Create result tables and SVG visualisations from lm-eval JSONL files.

By default, this uses the newest JSONL file for each model/task pair. That means
dropping a new model folder into results/main/ and rerunning this script updates
all CSVs and graphs automatically.

Usage:
    python data_visualisation/plot_results.py
    python data_visualisation/plot_results.py --watch
    python data_visualisation/plot_results.py --run-policy all
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

from visualisation.build import BuildSummary, build_visualisations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise JSONL sample results from results/main/.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/main"), help="Directory containing run folders.")
    parser.add_argument("--output-dir", type=Path, default=Path("data_visualisation/output"), help="Where outputs are written.")
    parser.add_argument("--metric", default="acc", help="Numeric JSONL metric to aggregate, for example acc or acc_norm.")
    parser.add_argument(
        "--run-policy",
        choices=("latest", "all"),
        default="latest",
        help="Use only the newest run per model/task, or aggregate all matching JSONL runs.",
    )
    parser.add_argument(
        "--zhoblimp-page-size",
        type=int,
        default=35,
        help="Number of zhoblimp subtasks per SVG bar-chart page.",
    )
    parser.add_argument("--histogram-bins", type=int, default=20, help="Number of bins for zhoblimp histograms.")
    parser.add_argument("--watch", action="store_true", help="Refresh the graphs repeatedly as result files change.")
    parser.add_argument("--watch-interval", type=float, default=15.0, help="Seconds between refreshes in watch mode.")
    return parser.parse_args()


def run_once(args: argparse.Namespace) -> BuildSummary:
    summary = build_visualisations(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        metric=args.metric,
        run_policy=args.run_policy,
        zhoblimp_page_size=args.zhoblimp_page_size,
        histogram_bins=args.histogram_bins,
    )
    print_summary(summary, args)
    return summary


def print_summary(summary: BuildSummary, args: argparse.Namespace) -> None:
    print(f"Loaded {summary.raw_score_count} task scores from {args.results_dir} ({args.run_policy} runs)")
    print(f"Found {summary.model_count} model(s)")
    print(f"Added zhoblimp averages for {summary.zhoblimp_average_count} model(s)")
    print(f"Wrote {summary.zhoblimp_page_count} zhoblimp subtask figure(s)")
    print(f"Wrote outputs to {summary.output_dir}")


def latest_results_signature(results_dir: Path) -> tuple[tuple[str, int, int], ...]:
    signature = []
    for path in sorted(results_dir.glob("*/*.jsonl")):
        stat = path.stat()
        signature.append((path.as_posix(), stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def watch(args: argparse.Namespace) -> None:
    print(f"Watching {args.results_dir} every {args.watch_interval:g}s. Press Ctrl+C to stop.")
    last_signature: tuple[tuple[str, int, int], ...] | None = None
    while True:
        signature = latest_results_signature(args.results_dir)
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
