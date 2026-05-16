"""Mandarin Hanzi transliteration helpers for Chinese evaluation tasks.

The target representation is one token per jieba word. Each Hanzi syllable in
that word becomes its pinyin initial plus a single digit encoding tone and
pinyin length. For example, "我们几点吃饭" becomes "W6M7 J6D8 C2f7".
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module=r"jieba\._compat",
)

try:
    import jieba
    from pypinyin import Style, pinyin
except ImportError as exc:  # pragma: no cover - exercised at runtime config load
    raise ImportError(
        "Chinese pinyin abbreviation tasks require jieba and pypinyin. "
        "Install project requirements with `pip install -r requirements.txt`."
    ) from exc

jieba.setLogLevel(logging.WARNING)


def _is_hanzi(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2A6DF
        or 0x2A700 <= codepoint <= 0x2B73F
        or 0x2B740 <= codepoint <= 0x2B81F
        or 0x2B820 <= codepoint <= 0x2CEAF
        or 0x2CEB0 <= codepoint <= 0x2EBEF
        or 0x30000 <= codepoint <= 0x3134F
    )


def _abbreviate_syllable(syllable: str) -> str:
    if not syllable:
        return syllable

    if syllable[-1] in "12345":
        body, tone = syllable[:-1], int(syllable[-1])
    else:
        body, tone = syllable, 5
    if not body:
        return syllable

    initial = body[0].lower()
    if tone in (1, 3, 5):
        initial = initial.upper()

    if tone in (1, 2):
        digit = 0
    else:
        digit = 5

    length = len(body)
    if length <= 1:
        digit += 0
    elif length == 2:
        digit += 1
    elif length == 3:
        digit += 2
    elif length == 4:
        digit += 3
    else:
        digit += 4

    return f"{initial}{digit}"


def _append_word_token(parts: list[str], token: str) -> None:
    if parts and parts[-1] and not parts[-1].endswith((" ", "\n", "\t")):
        parts.append(" ")
    parts.append(token)


def transliterate_text(text: Any) -> Any:
    """Convert Hanzi to word-level pinyin-initial tokens with encoded digits."""
    if not isinstance(text, str) or not any(_is_hanzi(char) for char in text):
        return text

    parts: list[str] = []
    hanzi_run: list[str] = []
    previous_was_hanzi = False

    def flush_hanzi_run() -> None:
        nonlocal previous_was_hanzi
        if not hanzi_run:
            return
        for word in jieba.lcut("".join(hanzi_run)):
            pronunciations = pinyin(
                word,
                style=Style.TONE3,
                neutral_tone_with_five=True,
                strict=False,
            )
            token = "".join(
                _abbreviate_syllable(pronunciation[0] if pronunciation else "")
                for pronunciation in pronunciations
            )
            _append_word_token(parts, token)
        hanzi_run.clear()
        previous_was_hanzi = True

    for char in text:
        if _is_hanzi(char):
            hanzi_run.append(char)
            continue

        flush_hanzi_run()
        if previous_was_hanzi and not char.isspace() and char.isalnum():
            parts.append(" ")
        parts.append(char)
        previous_was_hanzi = char.isspace()

    flush_hanzi_run()
    return "".join(parts)


def transliterate_doc(doc: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    """Return a shallow copy with selected string/list string fields converted."""
    converted = dict(doc)
    for field in fields:
        value = converted.get(field)
        if isinstance(value, list):
            converted[field] = [transliterate_text(item) for item in value]
        else:
            converted[field] = transliterate_text(value)
    return converted


def mubench_doc_to_text(doc: dict[str, Any]) -> str:
    return transliterate_text(doc["prompt"])


def mubench_doc_to_choice(doc: dict[str, Any]) -> list[str]:
    return [transliterate_text(choice) for choice in doc["choices"]]


def paired_sentence_doc_to_choice(doc: dict[str, Any]) -> list[str]:
    return [
        transliterate_text(doc["acceptable_sent"]),
        transliterate_text(doc["unacceptable_sent"]),
    ]


def zhoblimp_doc_to_choice(doc: dict[str, Any]) -> list[str]:
    return [
        transliterate_text(doc["sentence_good"]),
        transliterate_text(doc["sentence_bad"]),
    ]


def mmlu_doc_to_text(doc: dict[str, Any]) -> str:
    question = transliterate_text(doc["question"].strip())
    option_a = transliterate_text(doc["option_a"])
    option_b = transliterate_text(doc["option_b"])
    option_c = transliterate_text(doc["option_c"])
    option_d = transliterate_text(doc["option_d"])
    return (
        f"{question}\n"
        f"A. {option_a}\n"
        f"B. {option_b}\n"
        f"C. {option_c}\n"
        f"D. {option_d}\n"
        "Answer:"
    )


def sib200_doc_to_text(doc: dict[str, Any]) -> str:
    text = transliterate_text(doc["text"])
    return (
        "Given the categories science/technology, travel, politics, sports, "
        f"health, entertainment, or geography; what category does the text: '{text}' "
        "belong to:"
    )
