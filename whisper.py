from contextlib import suppress
from io import BytesIO
import logging
import mimetypes
import os
from pathlib import Path
import tempfile
from typing import Dict
from uuid import uuid4

import ffmpeg
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from faster_whisper import BatchedInferencePipeline, WhisperModel
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger("whisper")

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3")
try:
    WHISPER_BATCH_SIZE = max(int(os.getenv("WHISPER_BATCH_SIZE", "4")), 1)
except ValueError:
    WHISPER_BATCH_SIZE = 4

app = FastAPI()
model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device="cpu",
    compute_type="int8",
    num_workers=WHISPER_BATCH_SIZE,
)
batched_pipeline = (
    BatchedInferencePipeline(model=model) if WHISPER_BATCH_SIZE > 1 else None
)

MP4_INITIALIZATION_SEGMENTS: Dict[str, bytes] = {}


def _write_temp_file(audio_bytes: bytes, content_type: str | None) -> Path:
    suffix = mimetypes.guess_extension(content_type or "") or ".bin"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as tmp_file:
        tmp_file.write(audio_bytes)
    return Path(path)


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
    offset = 0
    total = len(data)

    while offset + 8 <= total:
        size = int.from_bytes(data[offset : offset + 4], "big")
        box_type = data[offset + 4 : offset + 8]

        if box_type in {b"moof", b"mdat"}:
            return data[:offset] if offset > 0 else None

        if size == 0:
            break

        if size == 1:
            if offset + 16 > total:
                break
            size = int.from_bytes(data[offset + 8 : offset + 16], "big")

        if size <= 0:
            break

        offset += size

    return None


def _probe_audio(path: Path) -> None:
    try:
        ffmpeg.probe(str(path))
    except ffmpeg.Error as exc:
        message = exc.stderr.decode().strip()
        raise ValueError(f"ffprobe failed: {message}") from exc


def _decode_to_wav(path: Path) -> bytes:
    wav_bytes, _ = (
        ffmpeg.input(str(path))
        .output("pipe:1", format="wav", acodec="pcm_s16le", ac=1, ar="16000")
        .run(
            capture_stdout=True,
            capture_stderr=True,
            cmd=["ffmpeg", "-nostdin", "-loglevel", "error"],
        )
    )
    return wav_bytes


@app.post("/transcribe")
async def transcribe_audio(
    channel_id: str | None = Form(None),
    language: str | None = Form(None),
    file: UploadFile = File(...),
):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    content_type = file.content_type or ""
    lower_content_type = content_type.lower()
    key = channel_id or None
    language_hint = (language or "").strip()
    if language_hint.lower() == "auto":
        language_hint = ""

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

    temp_path = _write_temp_file(audio_bytes, content_type)
    probe_note: str | None = None

    try:
        try:
            await run_in_threadpool(_probe_audio, temp_path)
        except ValueError as exc:
            probe_note = str(exc)
            logger.warning(probe_note)

        wav_bytes = await run_in_threadpool(_decode_to_wav, temp_path)
    except ffmpeg.Error as exc:
        detail = exc.stderr.decode().strip() or "Audio decoding failed."
        if debug_dump_path:
            detail = f"{detail} | raw saved: {debug_dump_path}"
        if probe_note:
            detail = f"{detail} | {probe_note}"
        if key:
            MP4_INITIALIZATION_SEGMENTS.pop(key, None)
        raise HTTPException(status_code=400, detail=detail) from exc
    finally:
        with suppress(FileNotFoundError):
            temp_path.unlink()

    wav_stream = BytesIO(wav_bytes)

    def do_transcribe():
        wav_stream.seek(0)
        kwargs: dict[str, object] = {"beam_size": 1}
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

    return {"text": transcript, "language": info.language}
