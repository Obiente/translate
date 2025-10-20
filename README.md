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
