#!/usr/bin/env python3
"""Transliterate Mandarin Hanzi into pinyin abbreviation tokens."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tasks.pinyin_abbreviation import PINYIN_FORMATS, transliterate_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Hanzi to the BabyLM pinyin abbreviation format, e.g. "
            "'中国' -> 'Z4g2' or, with --pinyin_format initials, 'zg'."
        )
    )
    parser.add_argument("inputs", nargs="*", help="Text snippets or file paths to transliterate.")
    parser.add_argument(
        "--pinyin_format",
        default="tone_length",
        choices=PINYIN_FORMATS,
        help="Output format: original initial+digit tokens or lowercase initials only.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite input files in place. Every input must be an existing file.",
    )
    return parser.parse_args()


def transliterate_input(value: str, in_place: bool, pinyin_format: str) -> str | None:
    path = Path(value)
    if path.is_file():
        converted = transliterate_text(path.read_text(encoding="utf-8"), pinyin_format)
        if in_place:
            path.write_text(converted, encoding="utf-8")
            return None
        return converted

    if in_place:
        raise FileNotFoundError(f"--in-place input is not a file: {value}")
    return transliterate_text(value, pinyin_format)


def main() -> None:
    args = parse_args()

    inputs = args.inputs or [sys.stdin.read()]
    outputs = [
        output
        for value in inputs
        if (output := transliterate_input(value, args.in_place, args.pinyin_format)) is not None
    ]

    if outputs:
        sys.stdout.write("\n".join(outputs))
        if not outputs[-1].endswith("\n"):
            sys.stdout.write("\n")


if __name__ == "__main__":
    main()
