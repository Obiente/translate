"""High-performance speech-to-text server built on WhisperLive.

This module exposes the same FastAPI surface area as the legacy ``whisper.py``
module but replaces the transcription core with the optimised WhisperLive
implementation. It focuses on two goals:

* Keep latency low for incremental streaming transcripts.
* Maximise throughput for offline (chunked) uploads by relying on the
  WhisperLive batching utilities.

The implementation intentionally keeps dependencies local to this file so it
can act as a drop-in replacement for the existing server entrypoint.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, Iterable, List, Optional, Tuple, Set

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
import httpx
from whisper_live.transcriber.transcriber_faster_whisper import Segment, TranscriptionInfo, WhisperModel

_log_level_name = os.getenv("WHISPERLIVE_LOG_LEVEL", "WARNING").strip().upper()
_log_level = getattr(logging, _log_level_name, logging.WARNING)
logging.basicConfig(level=_log_level)
logger = logging.getLogger("whisperlive")
logger.setLevel(_log_level)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _env_flag(name: str, default: str = "true") -> bool:
    value = os.getenv(name, default)
    return str(value).strip().lower() not in {"", "0", "false", "no"}


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Translation plumbing (mirrors the legacy whisper server behaviour)
# ---------------------------------------------------------------------------

TRANSLATION_BASE_URL = os.getenv("TRANSLATION_BASE_URL", "http://127.0.0.1:5000")
TRANSLATION_TIMEOUT = _env_float("TRANSLATION_TIMEOUT", 8.0)
TRANSLATION_ENABLED = _env_flag("WHISPER_SERVER_TRANSLATIONS", "true")
TRANSLATION_ALTERNATIVES_MAX = min(max(_env_int("TRANSLATION_ALTERNATIVES", 5), 0), 5)

_translation_client: httpx.AsyncClient | None = None
_translation_client_lock = asyncio.Lock()


async def _get_translation_client() -> httpx.AsyncClient:
    global _translation_client
    if _translation_client is not None:
        return _translation_client

    async with _translation_client_lock:
        if _translation_client is None:
            _translation_client = httpx.AsyncClient(
                base_url=TRANSLATION_BASE_URL,
                timeout=TRANSLATION_TIMEOUT,
                headers={"Content-Type": "application/json"},
            )
    return _translation_client


def _parse_target_languages(value: object) -> List[str]:
    if value is None:
        return []

    if isinstance(value, list):
        candidates = [str(item) for item in value]
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                candidates = [str(item) for item in parsed]
            else:
                candidates = [stripped]
        except json.JSONDecodeError:
            candidates = [part.strip() for part in stripped.split(",")]
    else:
        candidates = [str(value)]

    unique: List[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        entry = candidate.strip()
        if entry and entry not in seen:
            unique.append(entry)
            seen.add(entry)
    return unique


def _coerce_alternative_limit(raw_value: object) -> int:
    if raw_value is None:
        return 0

    numeric = 0
    if isinstance(raw_value, (int, float)):
        numeric = int(raw_value)
    elif isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped:
            return 0
        try:
            numeric = int(float(stripped))
        except ValueError:
            return 0
    else:
        return 0

    if numeric <= 0:
        return 0
    return min(numeric, TRANSLATION_ALTERNATIVES_MAX)


async def _translate_transcript(
    text: str,
    source_language: str | None,
    targets: Iterable[str] | None,
    alternative_limit: int,
) -> Tuple[Dict[str, Dict[str, object]], str | None]:
    if not TRANSLATION_ENABLED:
        return {}, None

    trimmed = text.strip()
    if not trimmed:
        return {}, None

    unique_targets: List[str] = []
    seen = set()
    for raw in targets or []:
        candidate = str(raw).strip()
        if candidate and candidate not in seen:
            unique_targets.append(candidate)
            seen.add(candidate)

    if not unique_targets:
        return {}, None

    requested_alternatives = _coerce_alternative_limit(alternative_limit)
    client = await _get_translation_client()
    payloads: List[Tuple[str, Dict[str, object]]] = []
    for target in unique_targets:
        payload: Dict[str, object] = {
            "q": trimmed,
            "source": source_language or "auto",
            "target": target,
            "format": "text",
        }
        if requested_alternatives > 0:
            payload["alternatives"] = requested_alternatives
        payloads.append((target, payload))

    async def request(pair: Tuple[str, Dict[str, object]]):
        target, payload = pair
        try:
            response = await client.post("/translate", json=payload)
            response.raise_for_status()
            data = response.json()
            translated = str(data.get("translatedText", "")).strip()
            raw_alternatives = data.get("alternatives")
            alternatives: List[str] = []
            if isinstance(raw_alternatives, list):
                for entry in raw_alternatives:
                    value = None
                    if isinstance(entry, str):
                        value = entry.strip()
                    elif isinstance(entry, dict):
                        for key in ("translatedText", "translation", "text", "value"):
                            maybe = entry.get(key)
                            if isinstance(maybe, str):
                                value = maybe.strip()
                                if value:
                                    break
                    if value and value not in alternatives:
                        alternatives.append(value)
            detected = data.get("detectedLanguage")
            detected_language = detected if isinstance(detected, str) else None
            return target, translated, alternatives, detected_language
        except Exception as exc:  # pragma: no cover - network safeguard
            logger.warning("Translation request failed for %s: %s", target, exc)
            return target, "", [], None

    results = await asyncio.gather(*(request(item) for item in payloads))

    translations: Dict[str, Dict[str, object]] = {}
    detected_language: str | None = None
    for target, primary, alternatives, detected in results:
        translations[target] = {"primary": primary, "alternatives": alternatives}
        if detected and not detected_language:
            detected_language = detected

    return translations, detected_language


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_STREAM_WINDOW = 1 * 60
STREAM_MAX_WINDOW_SECONDS = _env_float("WHISPERLIVE_STREAM_WINDOW", DEFAULT_STREAM_WINDOW)
STREAM_MIN_STEP_SECONDS = _env_float("WHISPERLIVE_STREAM_MIN_STEP", 0.25)
STREAM_EMIT_INTERVAL = _env_float("WHISPERLIVE_STREAM_MIN_INTERVAL", 0.5)
STREAM_CONTEXT_SECONDS = _env_float("WHISPERLIVE_STREAM_CONTEXT", STREAM_MAX_WINDOW_SECONDS * 0.65)
STREAM_TRANSLATION_INTERVAL = _env_float("WHISPERLIVE_STREAM_TRANSLATION_INTERVAL", 0.5)
AUTO_FINAL_REPEAT_THRESHOLD = max(_env_int("WHISPERLIVE_AUTO_FINAL_REPEAT_THRESHOLD", 2), 1)
SENTENCE_FINAL_REPEAT_THRESHOLD = max(_env_int("WHISPERLIVE_SENTENCE_FINAL_REPEAT_THRESHOLD", 2), 1)


def _detect_device() -> Tuple[str, str]:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "float16"
        if torch.backends.mps.is_available():  # pragma: no cover - macOS metal
            return "mps", "float16"
    except Exception:  # pragma: no cover - torch optional
        pass
    return "cpu", "int8"


def _extract_complete_sentences(text: str) -> Tuple[str, str]:
    """Extract complete sentences from text.
    
    Returns:
        Tuple of (complete_sentences, remaining_text)
    """
    if not text.strip():
        return "", ""
    
    # Define sentence-ending punctuation
    sentence_endings = {'.', '!', '?', '。', '！', '？'}  # Include some common non-English punctuation
    
    # Find the last sentence ending
    last_ending_pos = -1
    for i, char in enumerate(text):
        if char in sentence_endings:
            # Make sure it's not just a decimal number or abbreviation
            # Simple heuristic: if followed by space and capital letter, it's likely a sentence end
            if i + 2 < len(text) and text[i + 1] == ' ' and text[i + 2].isupper():
                last_ending_pos = i
            elif i == len(text) - 1:  # End of text
                last_ending_pos = i
    
    if last_ending_pos >= 0:
        complete = text[:last_ending_pos + 1].strip()
        remaining = text[last_ending_pos + 1:].strip()
        return complete, remaining
    
    return "", text


def _resample_audio(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    duration = audio.shape[0] / src_rate
    target_samples = int(duration * dst_rate)
    if target_samples <= 0:
        return np.empty(0, dtype=np.float32)
    # Linear interpolation keeps dependencies minimal.
    indices = np.linspace(0, audio.shape[0] - 1, num=target_samples)
    resampled = np.interp(indices, np.arange(audio.shape[0]), audio)
    return resampled.astype(np.float32)


def _decode_audio_chunk(
    chunk: bytes,
    sample_rate: Optional[int],
    encoding: Optional[str],
) -> np.ndarray:
    if not chunk:
        return np.empty(0, dtype=np.float32)

    try:
        if encoding == "pcm16":
            audio = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768.0
        elif encoding == "float32":
            audio = np.frombuffer(chunk, dtype="<f4").astype(np.float32)
        elif chunk[:4] == b"RIFF" and chunk[8:12] == b"WAVE":
            import soundfile as sf  # type: ignore

            data, rate = sf.read(BytesIO(chunk), dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            audio = data.astype(np.float32)
            if sample_rate and rate != sample_rate:
                audio = _resample_audio(audio, rate, sample_rate)
        else:
            # Fallback: assume float32 little endian
            audio = np.frombuffer(chunk, dtype="<f4").astype(np.float32)
    except Exception:
        # As a safe guard, treat as float32 if decoding fails.
        audio = np.frombuffer(chunk, dtype="<f4").astype(np.float32)

    if sample_rate and audio.size:
        audio = _ensure_sample_rate(audio, sample_rate)
    return audio


def _ensure_sample_rate(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    # The streaming pipeline expects DEFAULT_SAMPLE_RATE internally.
    if sample_rate == DEFAULT_SAMPLE_RATE:
        return audio
    return _resample_audio(audio, sample_rate, DEFAULT_SAMPLE_RATE)


def _segments_to_text(segments: Iterable[Segment]) -> str:
    parts = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def _compute_delta(previous: str, current: str) -> str:
    if not previous:
        return current
    if current.startswith(previous):
        return current[len(previous):].lstrip()
    max_length = min(len(previous), len(current))
    index = 0
    while index < max_length and previous[index] == current[index]:
        index += 1
    delta = current[index:].lstrip()
    return delta or current


# ---------------------------------------------------------------------------
# WhisperLive runtime manager
# ---------------------------------------------------------------------------

@dataclass
class TranscriptionResult:
    text: str
    language: Optional[str]
    translations: Dict[str, Dict[str, object]]
    info: Optional[TranscriptionInfo] = None
    segments: Optional[List[Segment]] = None


class WhisperLiveManager:
    """Owning wrapper around the WhisperLive ``WhisperModel`` instance."""

    def __init__(self) -> None:
        model_ref = os.getenv("WHISPERLIVE_MODEL", "turbo")
        cpu_threads = _env_int("WHISPERLIVE_CPU_THREADS", 0)
        num_workers = max(_env_int("WHISPERLIVE_NUM_WORKERS", 4), 1)
        device, compute_type = _detect_device()

        logger.info(
            "Initialising WhisperLive model", extra={
                "model": model_ref,
                "device": device,
                "compute_type": compute_type,
                "num_workers": num_workers,
            },
        )

        self.model = WhisperModel(
            model_ref,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )
        self.stream_chunk_length = _env_int("WHISPERLIVE_CHUNK_LENGTH", 15)

    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        language_hint: Optional[str],
        target_languages: Iterable[str],
        translation_alternative_limit: int,
        *,
        translate: bool = True,
    ) -> TranscriptionResult:
        loop = asyncio.get_running_loop()
        language = None if not language_hint or language_hint == "auto" else language_hint

        def _run() -> Tuple[str, Optional[str], Dict[str, Dict[str, object]], TranscriptionInfo]:
            segments_iter, info = self.model.transcribe(
                BytesIO(audio_bytes),
                language=language,
                vad_filter=True,
                without_timestamps=False,
                chunk_length=self.stream_chunk_length,
                word_timestamps=False,
            )
            segments = list(segments_iter or [])
            text = _segments_to_text(segments)
            detected_language = info.language if info else None
            return text, detected_language, segments, info

        text, detected_language, segments, info = await loop.run_in_executor(None, _run)

        if translate and target_languages:
            translations, translation_detected = await _translate_transcript(
                text,
                detected_language or language,
                target_languages,
                translation_alternative_limit,
            )
        else:
            translations = {}
            translation_detected = None
        final_language = translation_detected or detected_language or language
        return TranscriptionResult(text=text, language=final_language, translations=translations, info=info, segments=segments)

    async def transcribe_array(
        self,
        audio: np.ndarray,
        language_hint: Optional[str],
        target_languages: Iterable[str],
        translation_alternative_limit: int,
        vad_filter: bool,
        *,
        translate: bool = True,
    ) -> TranscriptionResult:
        loop = asyncio.get_running_loop()
        language = None if not language_hint or language_hint == "auto" else language_hint
        audio = audio.astype(np.float32)

        def _run() -> Tuple[str, Optional[str], Dict[str, Dict[str, object]], TranscriptionInfo]:
            segments_iter, info = self.model.transcribe(
                audio,
                language=language,
                vad_filter=vad_filter,
                without_timestamps=False,
                chunk_length=self.stream_chunk_length,
                word_timestamps=False,
            )
            segments = list(segments_iter or [])
            text = _segments_to_text(segments)
            detected_language = info.language if info else None
            return text, detected_language, segments, info

        text, detected_language, segments, info = await loop.run_in_executor(None, _run)

        if translate and target_languages:
            translations, translation_detected = await _translate_transcript(
                text,
                detected_language or language,
                target_languages,
                translation_alternative_limit,
            )
        else:
            translations = {}
            translation_detected = None
        final_language = translation_detected or detected_language or language
        return TranscriptionResult(text=text, language=final_language, translations=translations, info=info, segments=segments)


# ---------------------------------------------------------------------------
# Streaming session management
# ---------------------------------------------------------------------------

class WhisperLiveStreamSession:
    """Maintains state for a single streaming WebSocket connection."""

    def __init__(
        self,
        manager: WhisperLiveManager,
        websocket: WebSocket,
        channel_id: str,
        language_hint: Optional[str],
        target_languages: Iterable[str],
        translation_alternative_limit: int,
        sample_rate: int,
    ) -> None:
        self.manager = manager
        self.websocket = websocket
        self.channel_id = channel_id
        self.language_hint = language_hint
        self.target_languages = list(target_languages)
        self.translation_alternative_limit = translation_alternative_limit
        self.sample_rate = sample_rate or DEFAULT_SAMPLE_RATE
        self.buffer = np.empty(0, dtype=np.float32)
        self.last_text = ""
        self.current_text = ""
        self.last_language = language_hint if language_hint and language_hint != "auto" else None
        self.last_translation_source = ""
        self.last_translations: Dict[str, Dict[str, object]] = {}
        self.last_emit = 0.0
        self.last_translation_time = 0.0
        self.repeated_transcript_count = 0
        self.last_repeated_transcript = ""
        # Sentence-level final tracking
        self.sentence_repeat_count = 0
        self.last_sentence_beginning = ""
        self.confirmed_text_prefix = ""  # Text that has been finalized via sentence completion
        self.lock = asyncio.Lock()
        self.new_audio_event = asyncio.Event()
        self.closed = False
        self.final_requested = False
        self.final_future: Optional[asyncio.Future[None]] = None
        self.sequence = 0
        self.min_samples = int(STREAM_MIN_STEP_SECONDS * DEFAULT_SAMPLE_RATE)
        self.context_samples = int(STREAM_CONTEXT_SECONDS * DEFAULT_SAMPLE_RATE)
        self.processor_task = asyncio.create_task(self._processing_loop())

    def _trim_buffer(self) -> None:
        max_samples = int(STREAM_MAX_WINDOW_SECONDS * DEFAULT_SAMPLE_RATE)
        if self.buffer.size > max_samples:
            self.buffer = self.buffer[-max_samples:]

    async def add_chunk(
        self,
        chunk_bytes: bytes,
        encoding: Optional[str],
        is_final: bool,
        sample_rate: Optional[int] = None,
    ) -> None:
        audio = _decode_audio_chunk(chunk_bytes, sample_rate or self.sample_rate, encoding)
        loop = asyncio.get_running_loop()
        future: Optional[asyncio.Future[None]] = None

        async with self.lock:
            if audio.size:
                self.buffer = np.concatenate((self.buffer, audio)) if self.buffer.size else audio
                self._trim_buffer()
                self.last_audio_time = time.monotonic()
                self.vad_hits = 0
            if sample_rate and sample_rate != self.sample_rate:
                self.sample_rate = sample_rate
            if is_final:
                self.final_requested = True
                if self.final_future is None or self.final_future.done():
                    self.final_future = loop.create_future()
                future = self.final_future
            self.new_audio_event.set()

        if is_final and future is not None:
            await future
            await self.websocket.send_json({"type": "utterance_complete", "channel_id": self.channel_id})

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.new_audio_event.set()
        if self.processor_task:
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass

    async def _processing_loop(self) -> None:
        try:
            while True:
                await self.new_audio_event.wait()
                self.new_audio_event.clear()
                if self.closed:
                    break
                try:
                    await self._maybe_run_inference()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception("Streaming inference failure: %s", exc)
                    await self._emit_error(str(exc))
        except asyncio.CancelledError:  # pragma: no cover - cancellation path
            pass
        finally:
            if self.final_future and not self.final_future.done():
                self.final_future.set_result(None)

    async def _emit_error(self, detail: str) -> None:
        try:
            await self.websocket.send_json({"type": "error", "detail": detail})
        except Exception:  # pragma: no cover - connection dropped
            pass

    async def _maybe_run_inference(self) -> None:
        now = time.monotonic()

        async with self.lock:
            snapshot = self.buffer.copy()
            final_requested = self.final_requested
            enough_samples = snapshot.size >= self.min_samples if self.min_samples > 0 else snapshot.size > 0
            should_run = final_requested or (enough_samples and (now - self.last_emit) >= STREAM_EMIT_INTERVAL)

        if not should_run:
            return

        if snapshot.size == 0:
            return

        result = await self.manager.transcribe_array(
            snapshot,
            self.language_hint,
            self.target_languages,
            self.translation_alternative_limit,
            vad_filter=True,
            translate=False,
        )

        await self._handle_result(result, final_requested)

    async def _handle_result(self, result: TranscriptionResult, is_final: bool) -> None:
        text = result.text.strip()
        if not text and not is_final:
            return

        language = result.language or self.last_language or self.language_hint
        
        # Check for sentence-level finals first (for long continuous speech)
        sentence_final_text = None
        remaining_text = text
        
        if text and not is_final:
            # Extract complete sentences from the current transcript
            complete_sentences, remaining = _extract_complete_sentences(text)
            
            if complete_sentences:
                # Check if the beginning of the complete sentences has stabilized
                sentence_beginning = complete_sentences[:min(50, len(complete_sentences))]  # First 50 chars
                
                if sentence_beginning == self.last_sentence_beginning:
                    self.sentence_repeat_count += 1
                    logger.debug(
                        "Sentence beginning repeated",
                        extra={
                            "channel_id": self.channel_id,
                            "sentence_count": self.sentence_repeat_count,
                            "sentence_preview": sentence_beginning,
                        },
                    )
                else:
                    self.sentence_repeat_count = 1
                    self.last_sentence_beginning = sentence_beginning
                
                # If sentence beginning is stable, emit it as final
                now_ts = time.monotonic()
                should_emit_sentence_final = (
                    self.sentence_repeat_count >= SENTENCE_FINAL_REPEAT_THRESHOLD
                    and (now_ts - self.last_emit) >= STREAM_EMIT_INTERVAL
                    and complete_sentences not in self.confirmed_text_prefix  # Avoid re-emitting
                )
                
                if should_emit_sentence_final:
                    logger.debug(
                        "Emitting sentence-level final",
                        extra={
                            "channel_id": self.channel_id,
                            "sentence_count": self.sentence_repeat_count,
                            "sentences_len": len(complete_sentences),
                            "remaining_len": len(remaining),
                        },
                    )
                    sentence_final_text = complete_sentences
                    remaining_text = remaining
                    self.confirmed_text_prefix = complete_sentences
                    self.sentence_repeat_count = 0
                    self.last_sentence_beginning = ""
        
        # Standard repeated transcript detection for complete final emission
        delta = _compute_delta(self.last_text, text)
        
        if text and text == self.last_repeated_transcript:
            self.repeated_transcript_count += 1
            logger.debug(
                "Repeated transcript detected",
                extra={
                    "channel_id": self.channel_id,
                    "count": self.repeated_transcript_count,
                    "text_preview": text[:50],
                },
            )
        else:
            self.repeated_transcript_count = 1
            self.last_repeated_transcript = text

        # Emit final if we've seen the same transcript enough times
        now_ts = time.monotonic()
        should_emit_full_final = (
            text
            and self.repeated_transcript_count >= AUTO_FINAL_REPEAT_THRESHOLD
            and not is_final
            and (now_ts - self.last_emit) >= STREAM_EMIT_INTERVAL
        )

        if should_emit_full_final:
            logger.debug(
                "Auto-final triggered by repeated transcript",
                extra={
                    "channel_id": self.channel_id,
                    "repeat_count": self.repeated_transcript_count,
                    "final_text_len": len(text),
                },
            )
            is_final = True
            self.repeated_transcript_count = 0  # Reset after emission
            self.last_repeated_transcript = ""
            # Clear confirmed prefix since we're finalizing everything
            self.confirmed_text_prefix = ""

        # Handle sentence-level final emission
        if sentence_final_text and not is_final:
            await self._emit_sentence_final(sentence_final_text, language)
            # Trim buffer and reset context after sentence final
            await self._reset_context_after_sentence_final()
            # Continue processing with remaining text for partial updates
            text = remaining_text
            delta = _compute_delta("", text)  # Reset delta since we finalized previous content

        if not is_final and not delta and not sentence_final_text:
            logger.debug(
                "Skipping emit: no delta and not final",
                extra={"channel_id": self.channel_id, "sequence": self.sequence},
            )
            return

        full_text = text
        self.current_text = text

        translations = self.last_translations
        should_translate = False
        now = time.monotonic()
        if self.target_languages:
            if is_final:
                should_translate = True
            elif full_text != self.last_translation_source and (now - self.last_translation_time) >= STREAM_TRANSLATION_INTERVAL:
                should_translate = True

        if should_translate:
            translations, detected_language = await _translate_transcript(
                full_text,
                language or self.language_hint,
                self.target_languages,
                self.translation_alternative_limit,
            )
            if detected_language:
                language = detected_language
            self.last_translation_source = full_text
            self.last_translations = translations
            self.last_translation_time = now
        elif not self.target_languages:
            self.last_translations = {}
            translations = {}

        translations_payload = self.last_translations if self.last_translation_source == full_text else {}

        # Base payload to sender (include room metadata if present)
        # Look up room context from connection
        meta = CONNECTION_INFO.get(self.websocket) if 'CONNECTION_INFO' in globals() else None
        room_id = meta.get("room_id") if meta else None
        peer_id = meta.get("peer_id") if meta else None
        peer_label = meta.get("peer_label") if meta else None

        payload = {
            "type": "transcript",
            "text": delta,
            "fullText": full_text,
            "language": language,
            "isFinal": is_final,
            "translations": translations_payload,
            "channel_id": self.channel_id,
            "sequence": self.sequence,
        }
        if room_id:
            payload.update({"room_id": room_id, "peer_id": peer_id, "peer_label": peer_label})
        await self.websocket.send_json(payload)
        # Broadcast to room peers (non-sender) so remote clients receive updates
        if room_id:
            try:
                await _broadcast_room_transcript(self.websocket, {
                    "text": delta,
                    "fullText": full_text,
                    "language": language,
                    "isFinal": is_final,
                    "translations": translations_payload,
                }, self.channel_id, self.sequence)
            except Exception:
                pass
        self.sequence += 1
        self.last_emit = now_ts

        self.last_language = language
        self.last_text = text

        if is_final:
            await self._finalize_and_reset_context(final_requested=True)
        else:
            await self._shrink_buffer()

    async def _emit_sentence_final(self, sentence_text: str, language: Optional[str]) -> None:
        """Emit a final event for completed sentences during long continuous speech."""
        translations = {}
        should_translate = False
        now = time.monotonic()
        
        if self.target_languages:
            should_translate = True
        
        if should_translate:
            translations, detected_language = await _translate_transcript(
                sentence_text,
                language or self.language_hint,
                self.target_languages,
                self.translation_alternative_limit,
            )
            if detected_language:
                language = detected_language
        
        # Include room metadata for sentence finals too
        meta = CONNECTION_INFO.get(self.websocket) if 'CONNECTION_INFO' in globals() else None
        room_id = meta.get("room_id") if meta else None
        peer_id = meta.get("peer_id") if meta else None
        peer_label = meta.get("peer_label") if meta else None

        payload = {
            "type": "transcript",
            "text": sentence_text,  # For sentence finals, text is the complete sentence
            "fullText": sentence_text,
            "language": language,
            "isFinal": True,
            "translations": translations,
            "channel_id": self.channel_id,
            "sequence": self.sequence,
        }
        if room_id:
            payload.update({"room_id": room_id, "peer_id": peer_id, "peer_label": peer_label})
        await self.websocket.send_json(payload)
        if room_id:
            try:
                await _broadcast_room_transcript(self.websocket, {
                    "text": sentence_text,
                    "fullText": sentence_text,
                    "language": language,
                    "isFinal": True,
                    "translations": translations,
                }, self.channel_id, self.sequence)
            except Exception:
                pass
        self.sequence += 1
        self.last_emit = now
        
        logger.debug(
            "Sentence final emitted",
            extra={
                "channel_id": self.channel_id,
                "sentence_len": len(sentence_text),
                "sequence": self.sequence - 1,
            },
        )

    async def _shrink_buffer(self) -> None:
        if self.context_samples <= 0:
            return
        async with self.lock:
            if self.buffer.size > self.context_samples:
                self.buffer = self.buffer[-self.context_samples:]

    async def _reset_context_after_sentence_final(self) -> None:
        """Reset context after emitting a sentence-level final to prevent buffer growth."""
        async with self.lock:
            # More aggressive buffer trimming - keep only recent context for next sentence
            # This prevents unlimited growth during long continuous speech
            keep_context = max(self.min_samples * 4, int(3.0 * DEFAULT_SAMPLE_RATE))  # Keep ~3 seconds
            if self.buffer.size > keep_context:
                self.buffer = self.buffer[-keep_context:]
                logger.debug(
                    "Context reset after sentence final",
                    extra={
                        "channel_id": self.channel_id,
                        "new_buffer_size": self.buffer.size,
                        "keep_context_size": keep_context,
                    },
                )
            
            # Reset text tracking for the finalized content
            self.last_text = ""  # Reset since we finalized previous content
            self.repeated_transcript_count = 0
            self.last_repeated_transcript = ""

    async def _finalize_and_reset_context(self, final_requested: bool) -> None:
        """Finalize and reset context after a complete final emission."""
        async with self.lock:
            now_ts = time.monotonic()
            
            # Keep a small amount of recent audio context for next utterance
            # This helps with continuity while preventing unlimited growth
            keep_recent = max(self.min_samples * 2, int(1.0 * DEFAULT_SAMPLE_RATE))  # Keep ~1 second
            if self.buffer.size > keep_recent:
                self.buffer = self.buffer[-keep_recent:]
                logger.debug(
                    "Context trimmed after full final",
                    extra={
                        "channel_id": self.channel_id,
                        "new_buffer_size": self.buffer.size,
                        "keep_recent_size": keep_recent,
                    },
                )
            else:
                # If buffer is small, clear it completely
                self.buffer = np.empty(0, dtype=np.float32)
            
            self.final_requested = False
            future = self.final_future
            self.final_future = None
            self.last_translation_source = ""
            self.last_translations = {}
            self.last_text = ""
            self.current_text = ""
            self.last_translation_time = 0.0
            # Reset sentence tracking state
            self.sentence_repeat_count = 0
            self.last_sentence_beginning = ""
            self.confirmed_text_prefix = ""
            self.repeated_transcript_count = 0
            self.last_repeated_transcript = ""

            self.last_audio_time = now_ts

        if final_requested and future and not future.done():
            future.set_result(None)





# ---------------------------------------------------------------------------
# FastAPI surface
# ---------------------------------------------------------------------------

manager = WhisperLiveManager()
app = FastAPI(title="WhisperLive Server", version="1.0.0")


# ---------------------------------------------------------------------------
# Room management (server-side broadcast, no P2P)
# ---------------------------------------------------------------------------

# Track connections and room membership
ROOMS: Dict[str, Set[WebSocket]] = {}
CONNECTION_INFO: Dict[WebSocket, Dict[str, Optional[str]]] = {}
ROOM_LOCK = asyncio.Lock()


def _normalize_room_fields(message: Dict[str, object]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    room_id = message.get("room_id") or message.get("roomId")
    peer_id = message.get("peer_id") or message.get("peerId")
    peer_label = message.get("peer_label") or message.get("peerLabel")
    return (
        str(room_id).strip() if isinstance(room_id, str) and room_id.strip() else None,
        str(peer_id).strip() if isinstance(peer_id, str) and peer_id.strip() else None,
        str(peer_label).strip() if isinstance(peer_label, str) and peer_label.strip() else None,
    )


async def _add_to_room(ws: WebSocket, room_id: str, peer_id: Optional[str], peer_label: Optional[str]) -> None:
    async with ROOM_LOCK:
        # Remove from previous room if any
        prev = CONNECTION_INFO.get(ws)
        prev_room = prev.get("room_id") if prev else None
        if prev_room and prev_room in ROOMS:
            ROOMS[prev_room].discard(ws)
            if not ROOMS[prev_room]:
                del ROOMS[prev_room]

        # Update connection info
        CONNECTION_INFO[ws] = {
            "room_id": room_id,
            "peer_id": peer_id,
            "peer_label": peer_label,
        }
        ROOMS.setdefault(room_id, set()).add(ws)

    # Ack and roster broadcast outside lock
    try:
        await ws.send_json({
            "type": "room_joined",
            "room_id": room_id,
            "peer_id": peer_id,
            "peer_label": peer_label,
        })
    except Exception:
        pass
    await _broadcast_room_roster(room_id)


async def _leave_room(ws: WebSocket) -> None:
    async with ROOM_LOCK:
        info = CONNECTION_INFO.get(ws)
        if not info:
            return
        room_id = info.get("room_id")
        CONNECTION_INFO[ws] = {"room_id": None, "peer_id": info.get("peer_id"), "peer_label": info.get("peer_label")}
        if room_id and room_id in ROOMS:
            ROOMS[room_id].discard(ws)
            empty = not ROOMS[room_id]
        else:
            room_id = None
            empty = False
        if room_id and empty:
            del ROOMS[room_id]

    if room_id:
        try:
            await ws.send_json({"type": "room_left", "room_id": room_id})
        except Exception:
            pass
        await _broadcast_room_roster(room_id)


async def _remove_connection(ws: WebSocket) -> None:
    async with ROOM_LOCK:
        info = CONNECTION_INFO.pop(ws, None)
        room_id = info.get("room_id") if info else None
        if room_id and room_id in ROOMS:
            ROOMS[room_id].discard(ws)
            empty = not ROOMS[room_id]
            if empty:
                del ROOMS[room_id]
        else:
            room_id = None
    if room_id:
        await _broadcast_room_roster(room_id)


def _build_room_roster(room_id: str) -> List[Dict[str, Optional[str]]]:
    members: List[Dict[str, Optional[str]]] = []
    seen_peer_ids: Set[str] = set()
    sockets = ROOMS.get(room_id) or set()
    for s in sockets:
        meta = CONNECTION_INFO.get(s) or {}
        pid = meta.get("peer_id")
        plabel = meta.get("peer_label")
        # Deduplicate by peer_id (prefer first seen)
        if pid and pid in seen_peer_ids:
            continue
        if pid:
            seen_peer_ids.add(pid)
        members.append({
            "peer_id": pid,
            "peer_label": plabel,
            "channel_id": None,
        })
    return members


async def _broadcast_room_roster(room_id: str) -> None:
    sockets = list(ROOMS.get(room_id) or [])
    if not sockets:
        return
    roster = _build_room_roster(room_id)
    message = {"type": "room_roster", "room_id": room_id, "members": roster}
    for s in sockets:
        try:
            await s.send_json(message)
        except Exception:
            continue


async def _broadcast_room_transcript(ws: WebSocket, payload: Dict[str, object], channel_id: str, sequence: int) -> None:
    meta = CONNECTION_INFO.get(ws)
    if not meta:
        return
    room_id = meta.get("room_id")
    if not room_id:
        return
    sockets = list(ROOMS.get(room_id) or [])
    if not sockets:
        return
    # Prepare broadcast payload
    out = dict(payload)
    out.update({
        "type": "room_transcript",
        "room_id": room_id,
        "peer_id": meta.get("peer_id"),
        "peer_label": meta.get("peer_label"),
        "channel_id": channel_id,
        "sequence": sequence,
    })
    for s in sockets:
        if s is ws:
            continue
        try:
            await s.send_json(out)
        except Exception:
            continue


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global _translation_client
    if _translation_client is not None:
        try:
            await _translation_client.aclose()
        finally:
            _translation_client = None


@app.post("/transcribe")
async def transcribe_audio(
    channel_id: str | None = Form(None),
    language: str | None = Form(None),
    targets: str | None = Form(None),
    target_languages: str | None = Form(None),
    translation_alternatives: str | None = Form(None),
    file: UploadFile = File(...),
):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio buffer")

    language_hint = (language or "").strip() or None
    combined_targets: List[str] = []
    for raw in (targets, target_languages):
        if raw:
            combined_targets.extend(_parse_target_languages(raw))
    if combined_targets:
        unique_targets: List[str] = []
        seen: set[str] = set()
        for entry in combined_targets:
            if entry not in seen:
                unique_targets.append(entry)
                seen.add(entry)
        combined_targets = unique_targets

    alternative_limit = _coerce_alternative_limit(translation_alternatives)

    result = await manager.transcribe_bytes(
        audio_bytes,
        language_hint,
        combined_targets,
        alternative_limit,
    )

    return {
        "text": result.text,
        "language": result.language,
        "translations": result.translations,
        "info": {
            "duration": getattr(result.info, "duration", None),
            "languageProbability": getattr(result.info, "language_probability", None),
        },
    }


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()

    session: Optional[WhisperLiveStreamSession] = None

    try:
        while True:
            payload = await websocket.receive_text()
            try:
                message = json.loads(payload)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "detail": "Invalid JSON payload"})
                continue

            message_type = (message.get("type") or "").lower()

            if message_type == "ping":
                await websocket.send_json({"type": "pong", "ts": message.get("ts")})
                continue

            if message_type == "start":
                if session is not None:
                    try:
                        await session.close()
                    except Exception:
                        pass
                channel_id = str(message.get("channel_id") or f"session-{int(time.time()*1000)}")
                language_hint = message.get("language")
                sample_rate = message.get("sample_rate") or message.get("sampleRate") or DEFAULT_SAMPLE_RATE
                try:
                    sample_rate_int = int(sample_rate)
                except (TypeError, ValueError):
                    sample_rate_int = DEFAULT_SAMPLE_RATE
                targets_value = message.get("target_languages")
                if targets_value is None and "targets" in message:
                    targets_value = message.get("targets")
                target_languages = _parse_target_languages(targets_value)
                alternative_limit = _coerce_alternative_limit(
                    message.get("translation_alternatives") or message.get("alternatives")
                )
                # Implicit room join if room metadata is present
                room_id, peer_id, peer_label = _normalize_room_fields(message)
                if room_id:
                    await _add_to_room(websocket, room_id, peer_id, peer_label)
                session = WhisperLiveStreamSession(
                    manager=manager,
                    websocket=websocket,
                    channel_id=channel_id,
                    language_hint=language_hint,
                    target_languages=target_languages,
                    translation_alternative_limit=alternative_limit,
                    sample_rate=sample_rate_int,
                )
                await websocket.send_json({"type": "session_started", "channel_id": channel_id})
                continue

            if message_type == "chunk":
                if session is None:
                    await websocket.send_json({"type": "error", "detail": "Session not started"})
                    continue
                data = message.get("data")
                if not isinstance(data, str):
                    await websocket.send_json({"type": "error", "detail": "Missing audio chunk data"})
                    continue
                try:
                    chunk_bytes = base64.b64decode(data, validate=True)
                except Exception:
                    await websocket.send_json({"type": "error", "detail": "Invalid base64 payload"})
                    continue

                encoding = message.get("encoding") or message.get("mime_type") or message.get("mimeType")
                is_final = bool(message.get("is_final") or message.get("isFinal"))
                chunk_sample_rate = message.get("sample_rate") or message.get("sampleRate")
                try:
                    chunk_sample_rate_int = int(chunk_sample_rate) if chunk_sample_rate is not None else None
                except (TypeError, ValueError):
                    chunk_sample_rate_int = None
                
                # Handle dynamic parameter updates during active session
                chunk_targets = message.get("target_languages")
                if chunk_targets is None and "targets" in message:
                    chunk_targets = message.get("targets")
                if chunk_targets is not None:
                    session.target_languages = _parse_target_languages(chunk_targets)
                
                chunk_alternative_limit = message.get("translation_alternatives")
                if chunk_alternative_limit is None and "alternatives" in message:
                    chunk_alternative_limit = message.get("alternatives")
                if chunk_alternative_limit is not None:
                    session.translation_alternative_limit = _coerce_alternative_limit(chunk_alternative_limit)
                
                chunk_language_hint = message.get("language")
                if chunk_language_hint is not None:
                    session.language_hint = chunk_language_hint if chunk_language_hint != "auto" else None
                # Implicit room join/update if room metadata present in chunks
                room_id, peer_id, peer_label = _normalize_room_fields(message)
                if room_id:
                    await _add_to_room(websocket, room_id, peer_id, peer_label)
                
                await session.add_chunk(
                    chunk_bytes,
                    encoding=encoding,
                    is_final=is_final,
                    sample_rate=chunk_sample_rate_int,
                )
                continue

            if message_type == "stop":
                break

            if message_type == "join_room":
                room_id, peer_id, peer_label = _normalize_room_fields(message)
                if not room_id:
                    await websocket.send_json({"type": "error", "detail": "room_id required"})
                    continue
                await _add_to_room(websocket, room_id, peer_id, peer_label)
                continue

            if message_type == "leave_room":
                await _leave_room(websocket)
                continue

            await websocket.send_json({"type": "error", "detail": "Unsupported message type"})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Unexpected websocket error: %s", exc)
        try:
            await websocket.send_json({"type": "error", "detail": str(exc)})
        except Exception:
            pass
    finally:
        if session is not None:
            try:
                await session.close()
            except Exception:
                pass
        await _remove_connection(websocket)
        try:
            await websocket.close()
        except Exception:  # pragma: no cover - already closed
            pass
