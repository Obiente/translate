## Obiente Translate

Obiente Translate is a lightweight app for live, multi-source transcription and instant translation. It turns spoken conversations into readable text in real time and presents them in a clean, distraction-free "Focus" view — ideal for events, kiosks, and live captioning.

Why people use it
- Make spoken content readable and shareable instantly.
- Help multilingual audiences follow live conversations in real time.
- Perfect for online calls, language exchange, and multilingual friendships.
- Provide a clear, centered display for public installations and events.

Core features
- Live transcription from microphones and system audio.
- Optional real-time translation into target languages.
- Focus Mode: line-by-line, centered Lyrics style display.
- Participant chips for quick toggling and speaker clarity.
- **Voice Activity Detection (VAD)**: Automatically filters out silence and background noise, reducing network traffic and preventing hallucinations from silent audio chunks.
- **Noise Gate**: Suppresses low-level audio to improve transcription quality.

Quick start
```bash
pnpm install
pnpm dev    # start the dev server (usually http://localhost:3000)
```

Local Go speech server
```bash
# Default dev path. Docker builds whisper.cpp, includes ffmpeg, and serves on
# http://127.0.0.1:8080. Put models in gowhisper/models.
pnpm models:base
pnpm dev:go
```

Useful variants
```bash
# Use another mounted model from gowhisper/models.
pnpm models:turbo
MODEL_NAME=large-v3-turbo pnpm dev:go

# Bare-metal fallback if you want to debug CGO locally.
pnpm dev:go:local
```

Use multilingual model names such as `tiny`, `base`, or `large-v3-turbo`.
Avoid `.en` models for this app; Whisper `.en` models only transcribe English.

The Vite dev server now proxies `/ws/transcribe`, `/transcribe`, `/translate`,
`/detect`, `/languages`, and `/healthz` to `http://127.0.0.1:8080`, so local
development uses the Go path by default. Set `VITE_WHISPER_ENDPOINT` or
`VITE_WHISPER_STREAMING_ENDPOINT` to point the frontend elsewhere.

Translation notes
- The Go server exposes LibreTranslate-compatible `/translate`, `/detect`, and
  `/languages` endpoints and fans out multi-target translations concurrently.
- Dev and production use `https://libretranslate.obiente.cloud` by default.
  Keep translation traffic routed through the Go server so caching and
  multi-target fan-out stay consistent.
- Translation results are cached in-process and source=target requests return
  immediately.
- The browser often records WebM/Ogg/MP4 for chunked mode. The Docker dev
  server includes `ffmpeg`; streaming mode sends PCM and does not depend on it.

Build for production
```bash
pnpm build
pnpm preview
```

## Audio Filtering

To optimize network usage and transcription quality, the app includes intelligent audio filtering:

### Voice Activity Detection (VAD)
- Automatically detects whether audio contains speech or just silence/noise
- Prevents sending empty chunks that waste bandwidth and cause hallucinations
- Uses RMS energy analysis and Zero Crossing Rate (ZCR) detection
- Only sends audio chunks when actual speech is detected

### Noise Gate
- Suppresses low-level background noise and hiss
- Applies envelope-following gain reduction with configurable attack/release
- Cleans up audio before sending to the transcription server
- Default threshold: -40dB (ratio: 10:1)

### Configuration
The filters automatically scale to the audio sample rate. Default settings:
- Energy threshold: 0.01 (RMS)
- Max ZCR: 0.3 (higher = more noise-like)
- Min speech duration: ~0.1 seconds
- Noise gate threshold: 0.01 with 10:1 ratio

TL;DR Obiente Translate makes international speech readable and accessible in real time with a minimal, usable UI.
