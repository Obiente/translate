# syntax=docker/dockerfile:1.7

# --- Stage 1: Build and install whisper.cpp (headers + shared libs) ---
FROM debian:bookworm AS whisper

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

ARG WHISPER_CPP_REPO=https://github.com/ggerganov/whisper.cpp
# You can pin to a specific commit or tag via --build-arg WHISPER_CPP_REF=<ref>
ARG WHISPER_CPP_REF=master

WORKDIR /src
RUN git clone --depth 1 --branch "${WHISPER_CPP_REF}" "${WHISPER_CPP_REPO}" whisper.cpp

# Configure, build and install to /opt/whisper
RUN cmake -S whisper.cpp -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/whisper \
 && cmake --build build -j \
 && cmake --install build


# --- Stage 2: Build Go server with whisper_cpp tag ---
FROM golang:1.23-bookworm AS builder

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache go modules first
COPY server-go/go.mod server-go/go.sum ./server-go/
RUN cd server-go && go mod download

# Bring in full source
COPY . .

# Bring in whisper installation from previous stage
COPY --from=whisper /opt/whisper /opt/whisper

# Enable CGO and point to whisper headers / libs; embed rpath for runtime
ENV CGO_ENABLED=1
ENV CGO_CFLAGS="-I/opt/whisper/include"
ENV CGO_CPPFLAGS="-I/opt/whisper/include"
ENV CGO_LDFLAGS="-L/opt/whisper/lib -lwhisper -lggml -lggml-base -lggml-cpu -lstdc++ -lm -pthread -Wl,-rpath,/usr/local/lib"

# Build the server with whisper.cpp support
RUN cd server-go && go build -tags whisper_cpp -o /out/whisper-server ./cmd/server


# --- Stage 3: (optional) Model download stage for better caching ---
FROM debian:bookworm-slim AS model
ARG DEBIAN_FRONTEND=noninteractive
ARG DOWNLOAD_MODEL=0
ARG MODEL_NAME=large-v3-turbo
RUN set -eux; \
    mkdir -p /out; \
    if [ "$DOWNLOAD_MODEL" = "1" ]; then \
      apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
      curl -fsSL -o "/out/ggml-${MODEL_NAME}.bin" "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-${MODEL_NAME}.bin" && \
      apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*; \
    fi


# --- Stage 4: Runtime image ---
FROM debian:bookworm-slim AS runtime

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libstdc++6 \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# Copy whisper shared libs into standard location and refresh cache
COPY --from=whisper /opt/whisper/lib /usr/local/lib
RUN ldconfig

# Copy server binary
COPY --from=builder /out/whisper-server /usr/local/bin/whisper-server

# Default configuration; override at runtime as needed
# MODEL_NAME controls the expected model filename (ggml-${MODEL_NAME}.bin)
ARG MODEL_NAME=large-v3-turbo
ENV WHISPER_GO_ADDR=:8080 \
  WHISPER_MODEL_PATH=/models/ggml-${MODEL_NAME}.bin \
  WHISPER_THREADS=8 \
  LOG_LEVEL=info

# Create a mount point for models
RUN mkdir -p /models
# Copy pre-downloaded model (if any) from the model stage; this layer remains cached independently of app changes
COPY --from=model /out/ /models/

EXPOSE 8080
CMD ["/usr/local/bin/whisper-server"]
