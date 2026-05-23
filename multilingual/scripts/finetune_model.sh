#!/bin/bash
set -e

MODEL_NAME=""
langs="eng nld zho"
LR=5e-5
PATIENCE=3
BSZ=64
MAX_EPOCHS=10
SEED=12
PINYIN_FORMAT="tone_length"
PYTHON_BIN="${PYTHON_BIN:-python}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name)  MODEL_NAME="$2";  shift 2 ;;
        --langs)       langs="$2";       shift 2 ;;
        --lr)          LR="$2";          shift 2 ;;
        --patience)    PATIENCE="$2";    shift 2 ;;
        --bsz)         BSZ="$2";         shift 2 ;;
        --max_epochs)  MAX_EPOCHS="$2";  shift 2 ;;
        --seed)        SEED="$2";        shift 2 ;;
        --pinyin_format) PINYIN_FORMAT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL_NAME" ]]; then
    echo "Error: --model_name is required"; exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if [[ -x "../.venv/Scripts/python.exe" ]]; then
        PYTHON_BIN="../.venv/Scripts/python.exe"
    else
        echo "Error: Python executable not found. Set PYTHON_BIN to your Python path."; exit 1
    fi
fi

model_basename=$(basename $MODEL_NAME)
for LANGUAGE in $langs; do
    LANGUAGE=${LANGUAGE:0:2}  # eng/nld/zho -> en/nl/zh
    for task in arc belebele bmlama include mnli sib200 truthfulqa xnli; do
        TRAIN_NAME=$task
        VALID_NAME=$task
        DO_TRAIN=True
        MODEL_NAME_FULL=$MODEL_NAME

        mkdir -p finetune/results/$model_basename/${LANGUAGE}/$task/

        "$PYTHON_BIN" finetune/finetune_classification.py \
              --model_name_or_path "$MODEL_NAME" \
              --language "$LANGUAGE" \
              --pinyin_format "$PINYIN_FORMAT" \
              --output_dir "finetune/results/${model_basename}/${LANGUAGE}/${task}" \
              --train_file "finetune/data/multilingual/${LANGUAGE}/${task}/${task}_${LANGUAGE}.train.jsonl" \
              --validation_file "finetune/data/multilingual/${LANGUAGE}/${task}/${task}_${LANGUAGE}.valid.jsonl" \
              --do_train $DO_TRAIN \
              --do_eval \
              --do_predict \
              --max_seq_length 128 \
              --per_device_train_batch_size "$BSZ" \
              --learning_rate "$LR" \
              --num_train_epochs "$MAX_EPOCHS" \
              --patience "$PATIENCE" \
              --eval_strategy epoch \
              --save_strategy epoch \
              --overwrite_output_dir \
              --seed "$SEED"
    done
done

# Add `--trust_remote_code` if you need to load custom config/model files.
# If you run into memory issues, try reducing $BSZ or reducing `--max_seq_length` first.
