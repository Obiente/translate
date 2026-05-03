#!/usr/bin/env sh
set -eu

MODEL_NAME="${WHISPER_MODEL_NAME:-base}"
MODEL_BASE_URL="${WHISPER_MODEL_BASE_URL:-https://huggingface.co/ggerganov/whisper.cpp/resolve/main}"
DOWNLOAD_ENABLED="${WHISPER_MODEL_DOWNLOAD:-auto}"

if [ -z "${WHISPER_MODEL_PATH:-}" ]; then
  export WHISPER_MODEL_PATH="/models/ggml-${MODEL_NAME}.bin"
fi

if [ ! -f "$WHISPER_MODEL_PATH" ]; then
  case "$DOWNLOAD_ENABLED" in
    1|true|TRUE|yes|YES|on|ON|auto|AUTO)
      mkdir -p "$(dirname "$WHISPER_MODEL_PATH")"
      echo "Downloading Whisper model to $WHISPER_MODEL_PATH"
      curl -fsSL -o "$WHISPER_MODEL_PATH" "${MODEL_BASE_URL}/$(basename "$WHISPER_MODEL_PATH")"
      ;;
    *)
      echo "Whisper model not found at $WHISPER_MODEL_PATH" >&2
      echo "Set WHISPER_MODEL_PATH to an existing file or enable WHISPER_MODEL_DOWNLOAD." >&2
      exit 1
      ;;
  esac
fi

exec /usr/local/bin/whisper-server
