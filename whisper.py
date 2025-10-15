from collections import deque
from io import BytesIO
import base64
import binascii
import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import Dict
from uuid import uuid4

import ffmpeg
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from faster_whisper import BatchedInferencePipeline, WhisperModel
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger("whisper")
logging.basicConfig(level=logging.INFO)

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "turbo")  # Optimized for CPU streaming
try:
    WHISPER_BATCH_SIZE = max(int(os.getenv("WHISPER_BATCH_SIZE", "6")), 1)
except ValueError:
    WHISPER_BATCH_SIZE = 1

app = FastAPI()
model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device="cpu",
    compute_type="int8",  # Best balance for CPU (int8 > float16 > float32 for CPU)
    num_workers=WHISPER_BATCH_SIZE,
    cpu_threads=0,  # Use all available CPU cores
)
batched_pipeline = (
    BatchedInferencePipeline(model=model) if WHISPER_BATCH_SIZE > 1 else None
)

MP4_INITIALIZATION_SEGMENTS: Dict[str, bytes] = {}
STREAMING_INITIAL_SEGMENTS: Dict[str, bytes] = {}
# Streaming configuration
MAX_STREAM_BUFFER_SIZE = 32 * 1024 * 1024  # 32 MiB buffer limit (reduced for efficiency)
STREAMING_WINDOW_SECONDS = 6.0  # Legacy value; buffer trimming now size-based only
MIN_CHUNK_SIZE = 4096  # ~0.125s at 16kHz (very small for frequent intermediate updates)
AUTO_FINAL_SILENCE_THRESHOLD = 0.05  # Shorter silence threshold due to turbo's accuracy
AUTO_FINAL_REPEAT_THRESHOLD = 2  # Reasonable finalization threshold
    
# Transcription throttling - minimum interval between transcriptions
MIN_TRANSCRIPTION_INTERVAL = 0.1  # 100ms balance between responsiveness and stability

# Audio parameters for buffer management
DEFAULT_SAMPLE_RATE = 16000  # Default 16kHz sample rate (for estimation)
BITS_PER_SAMPLE = 16
BYTES_PER_SAMPLE = BITS_PER_SAMPLE // 8  # 16-bit audio
DEFAULT_BYTES_PER_SECOND = DEFAULT_SAMPLE_RATE * BYTES_PER_SAMPLE  # ~32KB per second


def _build_wav_from_pcm(pcm_bytes: bytes, sample_rate: int) -> bytes:
    """Construct a WAV file from raw PCM samples."""
    data_size = len(pcm_bytes)
    riff_chunk_size = data_size + 36
    byte_rate = sample_rate * BYTES_PER_SAMPLE  # mono, 16-bit
    block_align = BYTES_PER_SAMPLE

    header = bytearray(44)
    header[0:4] = b"RIFF"
    header[4:8] = riff_chunk_size.to_bytes(4, "little", signed=False)
    header[8:12] = b"WAVE"
    header[12:16] = b"fmt "
    header[16:20] = (16).to_bytes(4, "little", signed=False)  # PCM header size
    header[20:22] = (1).to_bytes(2, "little", signed=False)   # PCM format
    header[22:24] = (1).to_bytes(2, "little", signed=False)   # Mono channel
    header[24:28] = sample_rate.to_bytes(4, "little", signed=False)
    header[28:32] = byte_rate.to_bytes(4, "little", signed=False)
    header[32:34] = block_align.to_bytes(2, "little", signed=False)
    header[34:36] = BITS_PER_SAMPLE.to_bytes(2, "little", signed=False)  # Bits per sample
    header[36:40] = b"data"
    header[40:44] = data_size.to_bytes(4, "little", signed=False)

    return bytes(header) + pcm_bytes

def calculate_bytes_per_second(sample_rate: int | None) -> int:
    """Calculate bytes per second for given sample rate"""
    effective_rate = sample_rate if sample_rate and sample_rate > 0 else DEFAULT_SAMPLE_RATE
    return effective_rate * BYTES_PER_SAMPLE

def calculate_window_bytes(sample_rate: int | None) -> int:
    """Calculate max window bytes for given sample rate"""
    return int(STREAMING_WINDOW_SECONDS * calculate_bytes_per_second(sample_rate))


def _env_flag(name: str, default: str = "true") -> bool:
    value = os.getenv(name, default)
    return str(value).strip().lower() not in {"", "0", "false", "no"}


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return max(int(raw_value), 0)
    except ValueError:
        return default


ENABLE_VAD_FILTER = _env_flag("WHISPER_VAD_FILTER", "true")
VAD_MIN_SILENCE_DURATION_MS = _env_int("WHISPER_VAD_MIN_SILENCE_MS", 500)

VAD_PARAMETERS = (
    {"min_silence_duration_ms": VAD_MIN_SILENCE_DURATION_MS}
    if ENABLE_VAD_FILTER
    else None
)


class TranscriptionError(Exception):
    """Represents a recoverable transcription failure."""

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


def _normalize_language_hint(language: str | None) -> str:
    hint = (language or "").strip()
    return "" if hint.lower() == "auto" else hint


def _compute_delta(previous: str, current: str) -> str:
    if not previous:
        return current

    if current.startswith(previous):
        return current[len(previous) :].lstrip()

    max_length = min(len(previous), len(current))
    index = 0
    while index < max_length and previous[index] == current[index]:
        index += 1

    delta = current[index:].lstrip()
    return delta or current


def _is_recoverable_streaming_failure(detail: str | None) -> bool:
    if not detail:
        return False

    lowered = detail.lower()
    recoverable_signatures = (
        "could not seek",
        "invalid frame size",
        "invalid data",
        "error while decoding",
        "ebml header parsing failed",  # WebM concatenation issues
        "invalid data found when processing input"  # General format issues
    )

    return any(signature in lowered for signature in recoverable_signatures)


def _resolve_input_format(content_type: str | None, is_streaming: bool = False) -> dict[str, str]:
    if not content_type:
        return {}

    lowered = content_type.split(";", 1)[0].strip().lower()
    
    # For streaming, we expect the client to send WAV or other simple formats
    # No special WebM handling needed - client should convert before sending
    mime_map = {
        "audio/wav": "wav",
        "audio/x-wav": "wav", 
        "audio/mp4": "mp4",
        "audio/aac": "aac",
        "audio/mpeg": "mp3",
        "audio/mp3": "mp3",
        "audio/ogg": "ogg",
        "audio/opus": "ogg",
        "audio/flac": "flac",
        "audio/x-flac": "flac",
    }

    if lowered in mime_map:
        return {"format": mime_map[lowered]}

    for pattern, ffmpeg_format in mime_map.items():
        if pattern in lowered:
            return {"format": ffmpeg_format}

    if "mp4" in lowered:
        return {"format": "mp4"}
    if "wav" in lowered:
        return {"format": "wav"}
    if "ogg" in lowered:
        return {"format": "ogg"}
    if "mpeg" in lowered or "mp3" in lowered:
        return {"format": "mp3"}

    return {}


async def _transcribe_audio_blob(
    audio_bytes: bytes,
    content_type: str | None,
    channel_id: str | None,
    language_hint: str,
    enable_vad: bool = True,
    is_streaming: bool = False,
    sample_rate: int | None = None,
) -> tuple[str, str | None]:
    if not audio_bytes:
        raise TranscriptionError("Empty audio buffer.")

    content_type = content_type or ""
    lower_content_type = content_type.lower()
    key = channel_id or None

    if key and "mp4" in lower_content_type:
        box_type = audio_bytes[4:8] if len(audio_bytes) >= 8 else b""
        if box_type == b"ftyp":
            init_segment = _extract_mp4_initialization_segment(audio_bytes)
            cached = init_segment if init_segment else audio_bytes
            MP4_INITIALIZATION_SEGMENTS[key] = cached
        else:
            header = MP4_INITIALIZATION_SEGMENTS.get(key)
            if header:
                audio_bytes = header + audio_bytes

    debug_dump_path = _maybe_dump_raw(audio_bytes, content_type)
    input_format_kwargs = _resolve_input_format(content_type, is_streaming)

    def decode_to_wav() -> bytes:
        # For streaming, expect client to send WAV or other simple formats
        # If already WAV, return as-is
        if audio_bytes.startswith(b'RIFF') and b'WAVE' in audio_bytes[:20]:
            print(f"DEBUG: Audio is already WAV format, returning as-is")
            return audio_bytes
        
        try:
            wav_bytes, _ = (
                ffmpeg.input("pipe:0", **input_format_kwargs)
                .output(
                    "pipe:1",
                    format="wav",
                    acodec="pcm_s16le",
                    ac=1,
                    ar="16000",  # Always resample to 16kHz for Whisper
                )
                .run(
                    input=audio_bytes,
                    capture_stdout=True,
                    capture_stderr=True,
                    cmd=["ffmpeg", "-nostdin", "-loglevel", "error"],
                )
            )
            return wav_bytes
        except ffmpeg.Error as exc:  # pragma: no cover - runtime safeguard
            detail = exc.stderr.decode().strip() or "Audio decoding failed."
            if debug_dump_path:
                detail = f"{detail} | raw saved: {debug_dump_path}"
            if key:
                MP4_INITIALIZATION_SEGMENTS.pop(key, None)
            raise TranscriptionError(detail) from exc

    wav_bytes = await run_in_threadpool(decode_to_wav)

    wav_stream = BytesIO(wav_bytes)

    def do_transcribe():
        wav_stream.seek(0)
        # Optimized parameters for turbo model performance
        kwargs: dict[str, object] = {
            "beam_size": 3 if is_streaming else 4,  # Turbo handles higher beam sizes efficiently
            "temperature": 0.0,  # Lower temperature for more deterministic output
            "patience": 1.8,  # Higher patience since turbo is fast enough to handle it
            "repetition_penalty": 1.02,  # Very light penalty - turbo is less repetitive
        }
        if ENABLE_VAD_FILTER and enable_vad:
            kwargs["vad_filter"] = True
            if VAD_PARAMETERS:
                kwargs["vad_parameters"] = dict(VAD_PARAMETERS)
        if language_hint:
            kwargs["language"] = language_hint
        if batched_pipeline is not None:
            kwargs["batch_size"] = WHISPER_BATCH_SIZE
            segments, info = batched_pipeline.transcribe(wav_stream, **kwargs)
        else:
            segments, info = model.transcribe(wav_stream, **kwargs)
        return segments, info

    segments, info = await run_in_threadpool(do_transcribe)
    transcript = " ".join(segment.text for segment in segments).strip()

    return transcript, getattr(info, "language", None)


def _maybe_dump_raw(audio_bytes: bytes, content_type: str | None) -> Path | None:
    debug_dir = os.getenv("WHISPER_DEBUG_DIR")
    if not debug_dir:
        return None

    try:
        target_dir = Path(debug_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        suffix = mimetypes.guess_extension(content_type or "") or ".bin"
        filename = f"{uuid4()}{suffix}"
        target_path = target_dir / filename
        with target_path.open("wb") as handle:
            handle.write(audio_bytes)
        return target_path
    except Exception:
        return None


def _extract_mp4_initialization_segment(data: bytes) -> bytes | None:
    """
    Extract MP4 initialization segment (ftyp + moov boxes only).
    Stops before any media data (moof/mdat) that contains actual audio.
    """
    offset = 0
    total = len(data)
    has_ftyp = False
    has_moov = False

    while offset + 8 <= total:
        size = int.from_bytes(data[offset : offset + 4], "big")
        box_type = data[offset + 4 : offset + 8]

        # Track essential boxes for init segment
        if box_type == b"ftyp":
            has_ftyp = True
        elif box_type == b"moov":
            has_moov = True

        # Stop at media data - these contain actual audio frames
        if box_type in {b"moof", b"mdat"}:
            init_segment = data[:offset] if offset > 0 else None
            # Only return if we have both essential boxes and no audio data
            if init_segment and has_ftyp and has_moov:
                return init_segment
            return None

        if size == 0:
            break

        if size == 1:
            if offset + 16 > total:
                break
            size = int.from_bytes(data[offset + 8 : offset + 16], "big")

        if size <= 0:
            break

        offset += size

    # If we processed the whole chunk without finding media data,
    # only return it if it's a proper init segment (has ftyp+moov)
    if has_ftyp and has_moov and total < 50000:  # Reasonable size limit
        return data
    
    return None


def _is_webm_initialization_chunk(data: bytes) -> bool:
    """
    Detect if a WebM chunk is likely an initialization segment.
    WebM init segments contain EBML header and metadata but NO audio clusters.
    """
    if len(data) < 8:
        return False
    
    # Look for EBML header (0x1A45DFA3)
    ebml_header = b'\x1a\x45\xdf\xa3'
    
    # Must start with EBML header for proper init segment
    if not data.startswith(ebml_header):
        return False
    
    # Check for audio cluster markers that indicate this contains audio data
    # WebM clusters contain actual audio/video data and should not be cached
    cluster_marker = b'\x1f\x43\xb6\x75'  # Cluster element ID
    simple_block = b'\xa3'  # SimpleBlock element (contains audio frames)
    block_group = b'\xa0'   # BlockGroup element (contains audio frames)
    
    # If we find cluster/audio markers, this is NOT an init segment
    if (cluster_marker in data or 
        simple_block in data or 
        block_group in data):
        return False
    
    # True init segments should be reasonably sized (not too large)
    # and contain metadata/codec info but no actual media frames
    if len(data) > 10000:  # If larger than 10KB, likely contains audio
        return False
        
    # Additional check: init segments typically contain track info
    track_entry = b'\xae'  # TrackEntry element ID
    if track_entry not in data:
        return False
    
    return True

@app.post("/transcribe")
async def transcribe_audio(
    channel_id: str | None = Form(None),
    language: str | None = Form(None),
    file: UploadFile = File(...),
):
    audio_bytes = await file.read()
    language_hint = _normalize_language_hint(language)

    try:
        transcript, detected_language = await _transcribe_audio_blob(
            audio_bytes,
            file.content_type,
            channel_id,
            language_hint,
        )
    except TranscriptionError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc

    return {"text": transcript, "language": detected_language}


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()

    # Session state
    channel_id: str | None = None
    language_hint = ""
    mime_type: str | None = None
    session_sample_rate: int | None = None  # Track sample rate for dynamic window calculations
    
    # Audio buffer management - sliding window
    audio_chunks = deque()  # Store (timestamp, raw_bytes) tuples
    total_audio_bytes = 0
    chunk_counter = 0
    is_wav_stream = False
    
    # Transcription state
    last_full_transcript = ""
    current_transcript = ""
    detected_language: str | None = None
    last_sequence_processed = -1
    last_transcription_time = 0.0
    
    # Auto-finalization tracking
    repeated_transcript_count = 0
    last_repeated_transcript = ""
    
    # Error recovery tracking
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 3  # Faster error recovery for responsiveness
    
    def add_audio_chunk(chunk_bytes: bytes) -> None:
        nonlocal total_audio_bytes, chunk_counter, is_wav_stream
        
        if not chunk_bytes:
            return
            
        import time
        timestamp = time.time()
        
        # Simple storage - client should send WAV or other simple formats
        if chunk_bytes.startswith(b"RIFF") and b"WAVE" in chunk_bytes[:16] and len(chunk_bytes) >= 44:
            # Strip header so we can rebuild a single coherent WAV later
            pcm_payload = chunk_bytes[44:]
            audio_chunks.append((timestamp, pcm_payload))
            total_audio_bytes += len(pcm_payload)
            is_wav_stream = True
        else:
            audio_chunks.append((timestamp, chunk_bytes))
            total_audio_bytes += len(chunk_bytes)
        chunk_counter += 1
        
        # Enforce maximum buffer size to avoid unbounded growth
        dropped_bytes = 0
        while total_audio_bytes > MAX_STREAM_BUFFER_SIZE and audio_chunks:
            _, old_chunk = audio_chunks.popleft()
            total_audio_bytes -= len(old_chunk)
            dropped_bytes += len(old_chunk)

        if dropped_bytes:
            print(f"DEBUG: Dropped {dropped_bytes} bytes to enforce MAX_STREAM_BUFFER_SIZE ({MAX_STREAM_BUFFER_SIZE})")

        # Calculate approximate duration for logging (using session sample rate if available)
        bytes_per_second = calculate_bytes_per_second(session_sample_rate)
        approx_duration = total_audio_bytes / bytes_per_second if bytes_per_second > 0 else 0
        print(f"DEBUG: Audio buffer now has {len(audio_chunks)} chunks, {total_audio_bytes} bytes (~{approx_duration:.1f}s @ {session_sample_rate or DEFAULT_SAMPLE_RATE}Hz)")
    
    def get_audio_buffer() -> bytes:
        """Get the current audio buffer for transcription"""
        if not audio_chunks:
            return b""
        
        if is_wav_stream:
            pcm_payload = b"".join(chunk_data for _, chunk_data in audio_chunks)
            sample_rate = session_sample_rate or DEFAULT_SAMPLE_RATE
            return _build_wav_from_pcm(pcm_payload, sample_rate)

        # Combine all chunks in the sliding window
        buffer_parts = [chunk_data for _, chunk_data in audio_chunks]
        return b"".join(buffer_parts)
    
    def reset_session(clear_headers: bool = True) -> None:
        nonlocal audio_chunks, total_audio_bytes, chunk_counter
        nonlocal last_full_transcript, current_transcript, detected_language
        nonlocal last_sequence_processed, last_transcription_time, consecutive_errors
        nonlocal repeated_transcript_count, last_repeated_transcript, session_sample_rate
        nonlocal is_wav_stream
        
        audio_chunks.clear()
        total_audio_bytes = 0
        chunk_counter = 0
        last_full_transcript = ""
        current_transcript = ""
        detected_language = None
        last_sequence_processed = -1
        last_transcription_time = 0.0
        consecutive_errors = 0
        repeated_transcript_count = 0
        last_repeated_transcript = ""
        session_sample_rate = None
        is_wav_stream = False
        
        if clear_headers and channel_id:
            STREAMING_INITIAL_SEGMENTS.pop(channel_id, None)
            MP4_INITIALIZATION_SEGMENTS.pop(channel_id, None)

    try:
        while True:
            try:
                message = await websocket.receive_json()
            except ValueError:
                await websocket.send_json(
                    {"type": "error", "detail": "Invalid message payload."}
                )
                continue

            message_type = (message.get("type") or "").lower()

            if message_type == "start":
                print(f"DEBUG: Received start message: {message}")
                # Initialize new session
                requested_channel_id = message.get("channel_id")
                if isinstance(requested_channel_id, bytes):
                    try:
                        requested_channel_id = requested_channel_id.decode()
                    except Exception:
                        requested_channel_id = ""
                
                new_channel_id = (
                    str(requested_channel_id)
                    if isinstance(requested_channel_id, str) and requested_channel_id
                    else str(uuid4())
                )
                
                print(f"DEBUG: Setting channel_id to: {new_channel_id}")
                
                # Only preserve headers if same channel
                preserve_headers = channel_id == new_channel_id
                reset_session(clear_headers=not preserve_headers)
                
                channel_id = new_channel_id
                language_hint = _normalize_language_hint(message.get("language"))
                
                await websocket.send_json({
                    "type": "session_started",
                    "channel_id": channel_id
                })
                continue

            if message_type == "chunk":
                print(f"DEBUG: Received chunk message")
                if channel_id is None:
                    print(f"DEBUG: No channel_id, sending error")
                    await websocket.send_json(
                        {"type": "error", "detail": "Streaming session not initialized."}
                    )
                    continue

                # Decode chunk data
                chunk_data = message.get("data")
                if not isinstance(chunk_data, str):
                    print(f"DEBUG: Invalid chunk data type: {type(chunk_data)}")
                    await websocket.send_json(
                        {"type": "error", "detail": "Missing audio chunk data."}
                    )
                    continue

                try:
                    chunk_bytes = base64.b64decode(chunk_data, validate=True)
                    print(f"DEBUG: Decoded chunk of {len(chunk_bytes)} bytes")
                except binascii.Error:
                    print(f"DEBUG: Base64 decode error")
                    await websocket.send_json(
                        {"type": "error", "detail": "Invalid audio chunk encoding."}
                    )
                    continue

                sequence_value = message.get("sequence")
                is_final = bool(message.get("is_final") or message.get("isFinal"))
                sample_rate = message.get("sample_rate") or message.get("sampleRate")
                print(f"DEBUG: sequence={sequence_value}, is_final={is_final}, sample_rate={sample_rate}")

                # Skip duplicate sequences
                if (isinstance(sequence_value, int) and 
                    sequence_value <= last_sequence_processed and not is_final):
                    continue

                # Update MIME type
                chunk_mime = message.get("mime_type") or message.get("mimeType")
                if isinstance(chunk_mime, str) and chunk_mime:
                    if mime_type != chunk_mime:
                        print(f"DEBUG: MIME type changed from '{mime_type}' to '{chunk_mime}'")
                    mime_type = chunk_mime
                    print(f"DEBUG: Using MIME type: {mime_type}")

                # Validate sample rate if provided
                if sample_rate is not None and not isinstance(sample_rate, int):
                    try:
                        sample_rate = int(sample_rate)
                    except (ValueError, TypeError):
                        sample_rate = None
                
                # Update session sample rate for dynamic window calculations
                if sample_rate is not None and sample_rate > 0:
                    if session_sample_rate != sample_rate:
                        print(f"DEBUG: Sample rate updated: {session_sample_rate} -> {sample_rate}Hz")
                        session_sample_rate = sample_rate

                # Handle initialization segments (MP4 only)
                if chunk_bytes and channel_id and mime_type:
                    mime_lower = mime_type.lower()
                    if "mp4" in mime_lower and channel_id not in STREAMING_INITIAL_SEGMENTS:
                        init_segment = _extract_mp4_initialization_segment(chunk_bytes)
                        if init_segment:
                            STREAMING_INITIAL_SEGMENTS[channel_id] = init_segment
                            print(f"DEBUG: Cached MP4 init segment for channel {channel_id} ({len(init_segment)} bytes)")
                    else:
                        print(f"DEBUG: Using simple format: {mime_type} - no init segment needed")

                # Add chunk to sliding window buffer
                if chunk_bytes:
                    add_audio_chunk(chunk_bytes)

                # Handle final empty chunk
                if not chunk_bytes and is_final:
                    if current_transcript:
                        await websocket.send_json({
                            "type": "transcript",
                            "text": "",
                            "fullText": current_transcript,
                            "language": detected_language or language_hint or None,
                            "isFinal": True,
                            **({"sequence": sequence_value} if isinstance(sequence_value, int) else {})
                        })
                    reset_session(clear_headers=False)
                    continue

                # Skip if buffer too small and not final (but be very lenient for streaming)
                print(f"DEBUG: total_audio_bytes={total_audio_bytes}, MIN_CHUNK_SIZE={MIN_CHUNK_SIZE}, is_final={is_final}")
                # Be very lenient - allow very small chunks for frequent intermediate updates
                min_size_threshold = MIN_CHUNK_SIZE // 4 if is_final else MIN_CHUNK_SIZE // 3
                if total_audio_bytes < min_size_threshold and not is_final:
                    print(f"DEBUG: Skipping - buffer too small ({total_audio_bytes} < {min_size_threshold})")
                    continue
                
                # Throttle transcription to prevent overload
                current_time = time.time()
                time_since_last = current_time - last_transcription_time
                if time_since_last < MIN_TRANSCRIPTION_INTERVAL and not is_final:
                    print(f"DEBUG: Throttling transcription - only {time_since_last:.2f}s since last")
                    continue
                
                last_transcription_time = current_time
                
                # Additional cleanup: ensure sliding window doesn't grow too large
                if len(audio_chunks) > 50:  # If we have more than 50 chunks, be more aggressive
                    cleanup_threshold = STREAMING_WINDOW_SECONDS * 0.7  # Keep only 70% of window
                    chunks_before = len(audio_chunks)
                    while audio_chunks and len(audio_chunks) > 10:
                        chunk_time, chunk_data = audio_chunks[0]
                        if current_time - chunk_time > cleanup_threshold:
                            audio_chunks.popleft()
                            total_audio_bytes -= len(chunk_data)
                        else:
                            break
                    chunks_after = len(audio_chunks)
                    if chunks_before != chunks_after:
                        effective_rate = session_sample_rate or DEFAULT_SAMPLE_RATE
                        print(f"DEBUG: Aggressive cleanup: {chunks_before} -> {chunks_after} chunks @ {effective_rate}Hz")

                # Prepare audio data for transcription (sliding window)
                buffer_data = get_audio_buffer()
                
                # Add initialization header if needed (MP4 only)
                if channel_id in STREAMING_INITIAL_SEGMENTS and mime_type:
                    mime_lower = mime_type.lower()
                    init_segment = STREAMING_INITIAL_SEGMENTS[channel_id]
                    
                    # Validate that init segment is reasonable size (not containing lots of audio)
                    if len(init_segment) > 50000:  # 50KB limit for init segments
                        print(f"DEBUG: Init segment too large ({len(init_segment)} bytes), clearing cache")
                        STREAMING_INITIAL_SEGMENTS.pop(channel_id, None)
                    elif "mp4" in mime_lower and not buffer_data.startswith(init_segment):
                        buffer_data = init_segment + buffer_data
                        print(f"DEBUG: Added MP4 init segment to buffer ({len(init_segment)} bytes)")
                    else:
                        print(f"DEBUG: Simple format {mime_type} - no header prepending needed")

                # Calculate approximate duration for logging (using session sample rate)
                bytes_per_second = calculate_bytes_per_second(session_sample_rate)
                audio_duration = len(buffer_data) / bytes_per_second if buffer_data and bytes_per_second > 0 else 0
                effective_rate = session_sample_rate or DEFAULT_SAMPLE_RATE
                print(f"DEBUG: About to transcribe {len(buffer_data)} bytes (~{audio_duration:.1f}s @ {effective_rate}Hz), sequence: {sequence_value}, final: {is_final}")
                logger.info(f"Transcribing {len(buffer_data)} bytes (~{audio_duration:.1f}s @ {effective_rate}Hz), sequence: {sequence_value}, final: {is_final}")

                # Simple transcription - no complex fallback needed
                # Client should send WAV or other simple formats
                
                # Transcribe with same settings as chunked uploads for consistency
                try:
                    transcript, language_detected = await _transcribe_audio_blob(
                        buffer_data,
                        mime_type,
                        channel_id,
                        language_hint,
                        enable_vad=False,  # Keep VAD disabled for streaming to avoid premature truncation
                        is_streaming=True,  # Enable streaming-specific format handling
                        sample_rate=sample_rate,  # Pass sample rate for server-side resampling
                    )
                except TranscriptionError as exc:
                    error_detail = exc.detail
                    print(f"DEBUG: Transcription error with {mime_type}: {error_detail}")
                    logger.error(f"Transcription error: {error_detail}")
                    
                    consecutive_errors += 1
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        print(f"DEBUG: Too many consecutive errors ({consecutive_errors}), sending error")
                        await websocket.send_json({"type": "error", "detail": f"Persistent transcription failures: {error_detail}"})
                        consecutive_errors = 0  # Reset counter
                        continue
                        
                    if _is_recoverable_streaming_failure(error_detail):
                        print(f"DEBUG: Recoverable error, continuing...")
                        continue
                    await websocket.send_json({"type": "error", "detail": error_detail})
                    continue
                except Exception as exc:
                    logger.error(f"Unexpected transcription error: {exc}")
                    await websocket.send_json({"type": "error", "detail": str(exc)})
                    continue

                # Reset error counter on successful transcription
                consecutive_errors = 0
                
                if language_detected:
                    detected_language = language_detected

                transcript = transcript.strip()
                print(f"DEBUG: Got transcript: '{transcript}', current: '{current_transcript}'")
                logger.info(f"Got transcript: '{transcript}', current: '{current_transcript}'")
                
                # Check for repeated transcripts to auto-finalize (but be less aggressive)
                if transcript and transcript == last_repeated_transcript:
                    repeated_transcript_count += 1
                    print(f"DEBUG: Repeated transcript #{repeated_transcript_count}: '{transcript}'")
                    logger.info(f"Repeated transcript #{repeated_transcript_count}: '{transcript}'")
                    
                    # Auto-finalize if we've seen the same transcript enough times
                    if repeated_transcript_count >= AUTO_FINAL_REPEAT_THRESHOLD and not is_final:
                        print(f"DEBUG: Auto-finalizing after {repeated_transcript_count} repetitions")
                        logger.info(f"Auto-finalizing transcript after {repeated_transcript_count} repetitions: '{transcript}'")
                        is_final = True
                        repeated_transcript_count = 0  # Reset counter
                        last_repeated_transcript = ""
                elif transcript:
                    # Different transcript, reset counter (but be more lenient about "sameness")
                    # Only reset if the transcripts are substantially different
                    if not last_repeated_transcript or len(transcript.split()) != len(last_repeated_transcript.split()):
                        repeated_transcript_count = 0
                    last_repeated_transcript = transcript
                
                # Compute delta from previous transcript
                previous_transcript = current_transcript
                delta = _compute_delta(previous_transcript, transcript)
                has_new_content = bool(delta) or transcript != previous_transcript
                
                # Detect if this is a significant change (new sentence, not just extension)
                is_significant_change = False
                if current_transcript and transcript:
                    # More lenient significant change detection - only trigger on major differences
                    current_words = current_transcript.strip().split()
                    new_words = transcript.strip().split()
                    
                    # Only consider it significant if:
                    # 1. New transcript is much shorter (likely a restart)
                    # 2. Or the beginning has changed substantially
                    if (len(new_words) < len(current_words) * 0.5 or 
                        (len(current_words) > 3 and len(new_words) > 3 and 
                         " ".join(current_words[:3]) != " ".join(new_words[:3]))):
                        is_significant_change = True
                        print(f"DEBUG: Detected significant change - finalizing previous transcript")
                
                print(f"DEBUG: Delta: '{delta}', has_new_content: {has_new_content}, significant_change: {is_significant_change}")
                logger.info(f"Delta: '{delta}', has_new_content: {has_new_content}, significant_change: {is_significant_change}")
                
                # If significant change detected, first finalize the previous transcript
                if is_significant_change and current_transcript:
                    final_response = {
                        "type": "transcript",
                        "text": "",
                        "fullText": current_transcript,
                        "language": detected_language or language_hint or None,
                        "isFinal": True,
                    }
                    if isinstance(sequence_value, int):
                        final_response["sequence"] = sequence_value
                    
                    print(f"DEBUG: Sending final response for previous: {final_response}")
                    logger.info(f"Sending final response for previous: {final_response}")
                    await websocket.send_json(final_response)
                    
                    # Small delay to ensure message ordering
                    import asyncio
                    await asyncio.sleep(0.01)
                
                # Update current transcript
                if transcript:
                    current_transcript = transcript

                # Send transcriptions immediately like chunked uploads - no complex logic
                should_send = bool(transcript.strip())  # Simple and fast like /transcribe endpoint
                
                if should_send:
                    # For intermediate updates, always send the full transcript as text for better UX
                    # The frontend can handle the incremental display
                    response = {
                        "type": "transcript",
                        "text": transcript,  # Always send full transcript for intermediate updates
                        "fullText": current_transcript,
                        "language": detected_language or language_hint or None,
                        "isFinal": is_final,
                    }
                    if isinstance(sequence_value, int):
                        response["sequence"] = sequence_value
                    
                    print(f"DEBUG: Sending response: {response}")
                    logger.info(f"Sending response: {response}")
                    await websocket.send_json(response)
                else:
                    print(f"DEBUG: Skipping response - empty transcript")
                    logger.info("Skipping response - empty transcript")

                # Update sequence
                if isinstance(sequence_value, int):
                    last_sequence_processed = sequence_value

                # Reset for final chunks
                if is_final:
                    reset_session(clear_headers=False)

                continue

            if message_type == "stop":
                if current_transcript:
                    await websocket.send_json({
                        "type": "transcript",
                        "text": "",
                        "fullText": current_transcript,
                        "language": detected_language or language_hint or None,
                        "isFinal": True,
                    })
                break

            await websocket.send_json(
                {"type": "error", "detail": "Unsupported message type."}
            )
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        try:
            await websocket.send_json({"type": "error", "detail": str(e)})
        except:
            pass
    finally:
        if channel_id:
            MP4_INITIALIZATION_SEGMENTS.pop(channel_id, None)
            STREAMING_INITIAL_SEGMENTS.pop(channel_id, None)
