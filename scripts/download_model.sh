#!/usr/bin/env bash
# Download GGUF models from HuggingFace
# Usage: ./scripts/download_model.sh [model] [model_dir]
#   model: qwen2.5-7b | qwen3.5-35b (default) | llama-2-70b | llama-2-70b-q2
#   model_dir: directory to save model (default: models/)

set -euo pipefail

MODEL="${1:-qwen3.5-35b}"
MODEL_DIR="${2:-models}"

case "$MODEL" in
  qwen2.5-7b)
    REPO_ID="Smoffyy/Qwen2.5-7B-Instruct-Pure-GGUF"
    FILENAME="Qwen2.5-7B-q4_k_m.gguf"
    ;;
  qwen3.5-35b)
    REPO_ID="Smoffyy/Qwen3.5-35B-A3B-Instruct-Pure-GGUF"
    FILENAME="Qwen3.5-35B-A3B-q4_k_m.gguf"
    ;;
  llama-2-70b)
    REPO_ID="TheBloke/Llama-2-70B-Chat-GGUF"
    FILENAME="llama-2-70b-chat.Q4_K_S.gguf"
    echo "Note: 37 GB model. Needs 32+ GB RAM. Expect ~1 tok/s on RTX 4060 Ti."
    ;;
  llama-2-70b-q2)
    # Smaller quantization — fits in 32 GB RAM with headroom → ~2-3x faster than Q4_K_S
    REPO_ID="TheBloke/Llama-2-70B-Chat-GGUF"
    FILENAME="llama-2-70b-chat.Q2_K.gguf"
    echo "Note: ~19 GB Q2_K model. Fits fully in 32 GB RAM. Expect ~2-3 tok/s."
    ;;
  *)
    echo "ERROR: Unknown model '$MODEL'"
    echo "Usage: $0 [qwen2.5-7b|qwen3.5-35b|llama-2-70b|llama-2-70b-q2] [model_dir]"
    exit 1
    ;;
esac

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
