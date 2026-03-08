from __future__ import annotations

import asyncio
import html
import importlib
import json
import shutil
import tempfile
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.request import urlopen

from pydantic import BaseModel, Field

_WHISPER_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_WHISPER_MODEL_LOCK = Lock()


class TranscribeVideoRequest(BaseModel):
    video_id: str = Field(min_length=1)
    model_size: str = Field(default="turbo", min_length=1)
    language: str | None = None
    compute_type: str = Field(default="int8", min_length=1)


class TranscribeVideoResult(BaseModel):
    is_successful: bool
    text: str | None = None
    language: str | None = None
    error_message: str | None = None


class CaptionTranscriptResult(BaseModel):
    text: str | None = None
    language: str | None = None
    error_message: str | None = None


async def transcribe_video(*, request: TranscribeVideoRequest) -> TranscribeVideoResult:
    yt_dlp_runtime_error = get_yt_dlp_runtime_error_message()
    if yt_dlp_runtime_error:
        return TranscribeVideoResult(is_successful=False, error_message=yt_dlp_runtime_error)

    caption_result = await asyncio.to_thread(
        fetch_caption_transcript,
        video_id=request.video_id,
        preferred_language=request.language,
    )
    if caption_result.text:
        return TranscribeVideoResult(
            is_successful=True,
            text=caption_result.text,
            language=caption_result.language,
        )

    whisper_runtime_error = get_whisper_runtime_error_message()
    if whisper_runtime_error:
        if caption_result.error_message:
            return TranscribeVideoResult(
                is_successful=False,
                error_message=(
                    f"{caption_result.error_message} "
                    f"Local transcription fallback is unavailable: {whisper_runtime_error}"
                ),
            )

        return TranscribeVideoResult(is_successful=False, error_message=whisper_runtime_error)

    audio_path: str | None = None
    try:
        audio_path = await asyncio.to_thread(download_video_audio, video_id=request.video_id)
        transcription_payload = await asyncio.to_thread(
            run_transcription,
            audio_path=audio_path,
            model_size=request.model_size,
            compute_type=request.compute_type,
            language=request.language,
        )
    except Exception as exc:
        return TranscribeVideoResult(
            is_successful=False,
            error_message=f"Transcription failed for {request.video_id}: {exc}",
        )
    finally:
        await asyncio.to_thread(delete_audio_artifacts, audio_path=audio_path)

    return TranscribeVideoResult(
        is_successful=True,
        text=transcription_payload["text"],
        language=transcription_payload["language"],
    )


def fetch_caption_transcript(*, video_id: str, preferred_language: str | None) -> CaptionTranscriptResult:
    try:
        yt_dlp_module = importlib.import_module("yt_dlp")
    except Exception:
        return CaptionTranscriptResult(error_message="yt-dlp is not installed.")

    video_url = f"https://www.youtube.com/watch?v={video_id}"
    download_options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
    }

    try:
        with yt_dlp_module.YoutubeDL(download_options) as downloader:
            info = downloader.extract_info(video_url, download=False)
    except Exception as exc:
        return CaptionTranscriptResult(error_message=f"Could not read YouTube caption metadata: {exc}")

    all_tracks = collect_caption_tracks(info=info)
    if not all_tracks:
        return CaptionTranscriptResult(error_message="No YouTube captions were available for this video.")

    selected_track = select_caption_track(
        tracks=all_tracks,
        preferred_language=preferred_language,
    )
    if selected_track is None:
        return CaptionTranscriptResult(error_message="No usable YouTube caption track was found.")

    try:
        raw_caption_payload = download_text_from_url(url=selected_track["url"])
    except Exception as exc:
        return CaptionTranscriptResult(error_message=f"Could not download YouTube captions: {exc}")

    parsed_caption_text = parse_caption_payload(
        raw_payload=raw_caption_payload,
        extension=selected_track["ext"],
    )
    if not parsed_caption_text:
        return CaptionTranscriptResult(error_message="YouTube caption track did not contain usable text.")

    return CaptionTranscriptResult(
        text=parsed_caption_text,
        language=selected_track["language"],
    )


def collect_caption_tracks(*, info: dict) -> list[dict[str, str]]:
    raw_tracks: list[dict[str, str]] = []

    for source_key in ["subtitles", "automatic_captions"]:
        source_value = info.get(source_key)
        if not isinstance(source_value, dict):
            continue

        for language, language_tracks in source_value.items():
            if not isinstance(language_tracks, list):
                continue

            for track in language_tracks:
                if not isinstance(track, dict):
                    continue

                url = track.get("url")
                if not isinstance(url, str) or not url.strip():
                    continue

                extension = track.get("ext")
                if not isinstance(extension, str):
                    extension = ""

                raw_tracks.append(
                    {
                        "language": language,
                        "url": url,
                        "ext": extension.lower().strip(),
                    }
                )

    return raw_tracks


def select_caption_track(*, tracks: list[dict[str, str]], preferred_language: str | None) -> dict[str, str] | None:
    if not tracks:
        return None

    language_to_tracks: dict[str, list[dict[str, str]]] = {}
    for track in tracks:
        language = track["language"]
        language_to_tracks.setdefault(language, []).append(track)

    ordered_languages = build_language_priority(
        available_languages=list(language_to_tracks.keys()),
        preferred_language=preferred_language,
    )

    for language in ordered_languages:
        candidates = language_to_tracks.get(language)
        if not candidates:
            continue

        best_track = choose_best_track_format(tracks=candidates)
        if best_track is not None:
            return best_track

    return None


def build_language_priority(*, available_languages: list[str], preferred_language: str | None) -> list[str]:
    if not available_languages:
        return []

    normalized_available = [language.strip() for language in available_languages if language.strip()]
    if not normalized_available:
        return []

    ordered_languages: list[str] = []

    def push_if_missing(*, language: str) -> None:
        if language not in ordered_languages:
            ordered_languages.append(language)

    if preferred_language:
        normalized_preferred = preferred_language.strip().lower()
        if normalized_preferred:
            for language in normalized_available:
                if language.lower() == normalized_preferred:
                    push_if_missing(language=language)
            for language in normalized_available:
                lower_language = language.lower()
                if lower_language.startswith(f"{normalized_preferred}-"):
                    push_if_missing(language=language)

    for language in normalized_available:
        if language.lower() == "en" or language.lower().startswith("en-"):
            push_if_missing(language=language)

    for language in normalized_available:
        push_if_missing(language=language)

    return ordered_languages


def choose_best_track_format(*, tracks: list[dict[str, str]]) -> dict[str, str] | None:
    if not tracks:
        return None

    format_priority = {
        "json3": 0,
        "vtt": 1,
        "srv3": 2,
        "ttml": 3,
        "srt": 4,
    }

    sorted_tracks = sorted(
        tracks,
        key=lambda track: format_priority.get(track.get("ext", ""), 50),
    )
    return sorted_tracks[0]


def download_text_from_url(*, url: str) -> str:
    with urlopen(url, timeout=20) as response:
        return response.read().decode("utf-8", errors="ignore")


def parse_caption_payload(*, raw_payload: str, extension: str) -> str:
    if not raw_payload.strip():
        return ""

    if extension == "json3":
        return parse_json3_caption_payload(raw_payload=raw_payload)

    return parse_line_based_caption_payload(raw_payload=raw_payload)


def parse_json3_caption_payload(*, raw_payload: str) -> str:
    try:
        payload = json.loads(raw_payload)
    except Exception:
        return ""

    raw_fragments: list[str] = []
    for event in payload.get("events", []):
        if not isinstance(event, dict):
            continue

        event_segments = event.get("segs")
        if not isinstance(event_segments, list):
            continue

        for segment in event_segments:
            if not isinstance(segment, dict):
                continue

            text_value = segment.get("utf8")
            if isinstance(text_value, str) and text_value.strip():
                raw_fragments.append(text_value.replace("\n", " "))

    return normalize_caption_text(raw_text=" ".join(raw_fragments))


def parse_line_based_caption_payload(*, raw_payload: str) -> str:
    raw_lines = raw_payload.splitlines()
    kept_lines: list[str] = []

    for line in raw_lines:
        normalized_line = line.strip()
        if not normalized_line:
            continue

        upper_line = normalized_line.upper()
        if upper_line == "WEBVTT":
            continue

        if upper_line.startswith("NOTE"):
            continue

        if normalized_line.startswith("Kind:") or normalized_line.startswith("Language:"):
            continue

        if "-->" in normalized_line:
            continue

        if normalized_line.isdigit():
            continue

        kept_lines.append(normalized_line)

    return normalize_caption_text(raw_text=" ".join(kept_lines))


def normalize_caption_text(*, raw_text: str) -> str:
    unescaped_text = html.unescape(raw_text)
    collapsed_text = " ".join(unescaped_text.split())
    return collapsed_text.strip()


def get_yt_dlp_runtime_error_message() -> str | None:
    if not has_module(module_name="yt_dlp"):
        return "yt-dlp is not installed."

    return None


def get_whisper_runtime_error_message() -> str | None:
    if shutil.which("ffmpeg") is None:
        return "ffmpeg is not installed or not on PATH."

    if not has_module(module_name="faster_whisper"):
        return "faster-whisper is not installed."

    return None


def has_module(*, module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except Exception:
        return False

    return True


def download_video_audio(*, video_id: str) -> str:
    yt_dlp_module = importlib.import_module("yt_dlp")
    temp_directory_path = Path(tempfile.mkdtemp(prefix="video_finder_audio_"))
    output_template = str(temp_directory_path / "audio.%(ext)s")
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    download_options = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }

    with yt_dlp_module.YoutubeDL(download_options) as downloader:
        downloader.download([video_url])

    downloaded_files = [path for path in temp_directory_path.iterdir() if path.is_file()]
    if not downloaded_files:
        raise RuntimeError("Could not download audio stream.")

    return str(downloaded_files[0])


def run_transcription(
    *,
    audio_path: str,
    model_size: str,
    compute_type: str,
    language: str | None,
) -> dict[str, str | None]:
    model = get_whisper_model(model_size=model_size, compute_type=compute_type)
    segments, info = model.transcribe(
        audio_path,
        language=language,
        vad_filter=True,
    )

    raw_segment_texts = [
        segment.text.strip()
        for segment in segments
        if isinstance(segment.text, str) and segment.text.strip()
    ]
    full_text = " ".join(raw_segment_texts).strip()
    if not full_text:
        raise RuntimeError("Transcription produced empty text.")

    detected_language = language or getattr(info, "language", None)
    return {
        "text": full_text,
        "language": detected_language,
    }


def get_whisper_model(*, model_size: str, compute_type: str):
    cache_key = (model_size, compute_type)

    with _WHISPER_MODEL_LOCK:
        cached_model = _WHISPER_MODEL_CACHE.get(cache_key)
        if cached_model is not None:
            return cached_model

        faster_whisper_module = importlib.import_module("faster_whisper")
        model = faster_whisper_module.WhisperModel(
            model_size,
            device="auto",
            compute_type=compute_type,
        )
        _WHISPER_MODEL_CACHE[cache_key] = model

    return model


def delete_audio_artifacts(*, audio_path: str | None) -> None:
    if not audio_path:
        return

    path = Path(audio_path)
    parent_directory_path = path.parent

    if path.exists():
        path.unlink(missing_ok=True)

    if parent_directory_path.exists() and parent_directory_path.name.startswith("video_finder_audio_"):
        shutil.rmtree(parent_directory_path, ignore_errors=True)
