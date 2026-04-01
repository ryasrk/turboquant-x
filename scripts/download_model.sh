#!/usr/bin/env bash
# Download Qwen2.5-7B-Instruct GGUF model from HuggingFace
# Usage: ./scripts/download_model.sh [model_dir]

set -euo pipefail

MODEL_DIR="${1:-models}"
REPO_ID="Smoffyy/Qwen2.5-7B-Instruct-Pure-GGUF"
FILENAME="Qwen2.5-7B-q4_k_m.gguf"

mkdir -p "$MODEL_DIR"

echo "Downloading $FILENAME from $REPO_ID..."
echo "Target directory: $MODEL_DIR"

if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download "$REPO_ID" "$FILENAME" --local-dir "$MODEL_DIR"
elif command -v wget &>/dev/null; then
    URL="https://huggingface.co/${REPO_ID}/resolve/main/${FILENAME}"
    wget -c -O "${MODEL_DIR}/${FILENAME}" "$URL"
else
    echo "ERROR: Neither huggingface-cli nor wget found."
    echo "Install with: pip install huggingface-hub"
    exit 1
fi

echo "Downloaded: ${MODEL_DIR}/${FILENAME}"
ls -lh "${MODEL_DIR}/${FILENAME}"
