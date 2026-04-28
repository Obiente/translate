#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${WHISPER_CPP_PREFIX:-"$ROOT_DIR/.local/whisper"}"
SRC_DIR="${WHISPER_CPP_SRC:-"$ROOT_DIR/.local/src/whisper.cpp"}"
BUILD_DIR="${WHISPER_CPP_BUILD:-"$ROOT_DIR/.local/build/whisper.cpp"}"
REPO="${WHISPER_CPP_REPO:-https://github.com/ggerganov/whisper.cpp}"
REF="${WHISPER_CPP_REF:-master}"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

need_cmd go
need_cmd git
need_cmd cmake
need_cmd make
need_cmd gcc

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Warning: ffmpeg is not installed. Streaming PCM can work, but chunked browser uploads may fail." >&2
fi

if [ ! -f "$PREFIX/include/whisper.h" ]; then
  mkdir -p "$(dirname "$SRC_DIR")" "$(dirname "$BUILD_DIR")" "$PREFIX"

  if [ ! -d "$SRC_DIR/.git" ]; then
    echo "Cloning whisper.cpp into $SRC_DIR"
    git clone --depth 1 --branch "$REF" "$REPO" "$SRC_DIR"
  else
    echo "Using existing whisper.cpp checkout at $SRC_DIR"
  fi

  echo "Building whisper.cpp into $PREFIX"
  cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DWHISPER_BUILD_TESTS=OFF \
    -DWHISPER_BUILD_EXAMPLES=OFF
  cmake --build "$BUILD_DIR" -j"$(nproc 2>/dev/null || echo 2)"
  cmake --install "$BUILD_DIR"
fi

export CGO_ENABLED=1
export CGO_CFLAGS="-I$PREFIX/include ${CGO_CFLAGS:-}"
export CGO_CPPFLAGS="-I$PREFIX/include ${CGO_CPPFLAGS:-}"
export CGO_LDFLAGS="-L$PREFIX/lib -lwhisper -lggml -lggml-base -lggml-cpu -lstdc++ -lm -pthread -Wl,-rpath,$PREFIX/lib ${CGO_LDFLAGS:-}"
export LD_LIBRARY_PATH="$PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd "$ROOT_DIR"

if [ -z "${WHISPER_MODEL_PATH:-}" ]; then
  export WHISPER_MODEL_PATH="./models/ggml-base.en.bin"
fi

if [ ! -f "$WHISPER_MODEL_PATH" ]; then
  echo "Warning: WHISPER_MODEL_PATH does not exist yet: $WHISPER_MODEL_PATH" >&2
  echo "The server can start, but transcription will fail until a model is present." >&2
fi

exec go run -tags whisper_cpp ./cmd/server
