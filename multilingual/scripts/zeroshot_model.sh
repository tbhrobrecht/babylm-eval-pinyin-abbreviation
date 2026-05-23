#!/bin/bash
set -e

model_name=""
langs="eng nld zho"
revision="main"
pinyin_format="tone_length"
device="cpu"
PYTHON_BIN="${PYTHON_BIN:-python}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name) model_name="$2"; shift 2 ;;
        --langs) langs="$2"; shift 2 ;;
        --revision) revision="$2"; shift 2 ;;
        --pinyin_format) pinyin_format="$2"; shift 2 ;;
        --device) device="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$model_name" ]]; then
    echo "Error: --model_name is required"; exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if [[ -x "../.venv/Scripts/python.exe" ]]; then
        PYTHON_BIN="../.venv/Scripts/python.exe"
    else
        echo "Error: Python executable not found. Set PYTHON_BIN to your Python path."; exit 1
    fi
fi

for lang in $langs; do
    task_name="zeroshot_${lang}"
    echo "Evaluating ${lang} (revision: ${revision}, pinyin_format: ${pinyin_format}, device: ${device})"
    if [[ "$lang" == "zho" ]]; then
        export PINYIN_ABBREVIATION_FORMAT="$pinyin_format"
    else
        unset PINYIN_ABBREVIATION_FORMAT
    fi
    "$PYTHON_BIN" -m lm_eval \
        --model hf \
        --model_args pretrained=${model_name},revision=${revision},trust_remote_code=True \
        --tasks ${task_name} \
        --device ${device} \
        --output_path ../results/${revision} \
        --batch_size auto:10 \
        --num_fewshot 0 \
        --log_samples \
        --include_path tasks/

    echo "Completed evaluation for ${lang}"
done
