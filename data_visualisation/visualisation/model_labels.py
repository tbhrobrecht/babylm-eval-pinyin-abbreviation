"""Paper-facing display labels for evaluated models."""

from __future__ import annotations


MODEL_LABELS = {
    "10k_gpu1": "BABYLM",
    "10k_whitespace_false_gpu": "BABYLM-NoWS",
    "10k_whitespace_false_gpu1": "BABYLM-NoWS",
    "10k_chinese_gpu1": "CHINESE",
    "10k_chinese_whitespace_false_gpu1": "CHINESE-NoWS",
    "20k_chinese_whitespace_false_gpu1": "2CHINESE-NoWS",
    "10k_hanzi_gpu1": "HANZI",
    "10k_initials_gpu1": "INITIALS",
    "benchmark": "BASELINE",
}

MODEL_DISPLAY_ORDER = [
    "BABYLM",
    "BABYLM-NoWS",
    "CHINESE",
    "CHINESE-NoWS",
    "2CHINESE-NoWS",
    "INITIALS",
    "HANZI",
    "BASELINE",
]
_MODEL_DISPLAY_POSITION = {label: index for index, label in enumerate(MODEL_DISPLAY_ORDER)}


def display_model_name(model: str) -> str:
    """Return the publication label for a source model identifier."""
    return MODEL_LABELS.get(model, model.replace("_", " "))


def model_sort_key(model: str) -> tuple[int, str]:
    """Order models according to the presentation order, then any unknown labels."""
    label = display_model_name(model)
    return (_MODEL_DISPLAY_POSITION.get(label, len(MODEL_DISPLAY_ORDER)), label)
