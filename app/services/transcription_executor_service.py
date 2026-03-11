from __future__ import annotations

import asyncio
import html
import importlib
import json
import logging
import math
import mimetypes
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

import aiosqlite
import httpx
from pydantic import BaseModel, Field

from app.core.config import get_app_config
from app.services.groq_rate_limit_service import (
    GroqCapacityReservationRequest,
    GroqReservationOutcomeRequest,
    record_groq_reservation_outcome,
    reserve_groq_capacity,
)

GROQ_TRANSCRIPTIONS_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
logger = logging.getLogger(__name__)


class TranscribeVideoRequest(BaseModel):
    video_id: str = Field(min_length=1)
    language: str | None = None


class TranscribeVideoResult(BaseModel):
    is_successful: bool
    text: str | None = None
    language: str | None = None
    error_message: str | None = None


class CaptionTranscriptResult(BaseModel):
    text: str | None = None
    language: str | None = None
    error_message: str | None = None


class GroqTranscriptionRequestError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


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

    groq_api_key = get_groq_api_key()
    if not groq_api_key:
        groq_runtime_error = "GROQ_API_KEY is not configured."
        if caption_result.error_message:
            return TranscribeVideoResult(
                is_successful=False,
                error_message=(
                    f"{caption_result.error_message} "
                    f"Groq transcription fallback is unavailable: {groq_runtime_error}"
                ),
            )

        return TranscribeVideoResult(is_successful=False, error_message=groq_runtime_error)

    audio_path: str | None = None
    groq_reservation_id: str | None = None
    groq_request_was_successful = False
    groq_request_status_code: int | None = None
    groq_request_error_message: str | None = None
    app_config = get_app_config()
    try:
        audio_path = await asyncio.to_thread(download_video_audio, video_id=request.video_id)
        audio_duration_seconds = await resolve_audio_duration_seconds(video_id=request.video_id)
        if audio_duration_seconds is None:
            raise RuntimeError(
                "Could not determine audio duration for Groq transcription. "
                "Local rate limiting refuses unknown audio lengths."
            )

        reservation_result = await reserve_groq_capacity(
            request=GroqCapacityReservationRequest(
                db_path=app_config.app_db_path,
                model=app_config.groq_transcription_model,
                audio_seconds=audio_duration_seconds,
            )
        )
        if not reservation_result.is_allowed:
            raise RuntimeError(
                reservation_result.error_message
                or "Groq transcription is temporarily rate limited."
            )

        groq_reservation_id = reservation_result.reservation_id
        transcription_payload = await asyncio.to_thread(
            run_groq_transcription,
            audio_path=audio_path,
            language=request.language,
            api_key=groq_api_key,
            model=app_config.groq_transcription_model,
        )
        groq_request_was_successful = True
    except GroqTranscriptionRequestError as exc:
        groq_request_status_code = exc.status_code
        groq_request_error_message = str(exc)
        return TranscribeVideoResult(
            is_successful=False,
            error_message=f"Transcription failed for {request.video_id}: {exc}",
        )
    except Exception as exc:
        groq_request_error_message = str(exc)
        return TranscribeVideoResult(
            is_successful=False,
            error_message=f"Transcription failed for {request.video_id}: {exc}",
        )
    finally:
        if groq_reservation_id:
            try:
                await record_groq_reservation_outcome(
                    request=GroqReservationOutcomeRequest(
                        db_path=app_config.app_db_path,
                        reservation_id=groq_reservation_id,
                        is_successful=groq_request_was_successful,
                        response_status_code=groq_request_status_code,
                        error_message=groq_request_error_message,
                    )
                )
            except Exception:
                logger.exception(
                    "Failed to record Groq rate-limit reservation outcome for %s",
                    request.video_id,
                )
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


def get_groq_api_key() -> str | None:
    raw_api_key = get_app_config().groq_api_key
    if raw_api_key is None:
        return None

    normalized_api_key = raw_api_key.strip()
    if not normalized_api_key:
        return None

    return normalized_api_key


async def resolve_audio_duration_seconds(*, video_id: str) -> int | None:
    cached_duration_seconds = await load_cached_video_duration_seconds(video_id=video_id)
    if cached_duration_seconds is not None:
        return cached_duration_seconds

    return await asyncio.to_thread(fetch_video_duration_seconds_with_ytdlp, video_id=video_id)


async def load_cached_video_duration_seconds(*, video_id: str) -> int | None:
    normalized_video_id = video_id.strip()
    if not normalized_video_id:
        return None

    db_path = get_app_config().app_db_path.strip()
    if not db_path:
        return None

    try:
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT duration_seconds
                FROM video_metadata
                WHERE video_id = ?
                """,
                (normalized_video_id,),
            )
            row = await cursor.fetchone()
    except Exception:
        return None

    if row is None:
        return None

    duration_seconds = row["duration_seconds"]
    if not isinstance(duration_seconds, int) or duration_seconds < 1:
        return None

    return duration_seconds


def fetch_video_duration_seconds_with_ytdlp(*, video_id: str) -> int | None:
    try:
        yt_dlp_module = importlib.import_module("yt_dlp")
    except Exception:
        return None

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
    except Exception:
        return None

    duration_value = info.get("duration")
    if not isinstance(duration_value, (int, float)) or duration_value <= 0:
        return None

    return int(math.ceil(duration_value))


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


def run_groq_transcription(
    *,
    audio_path: str,
    language: str | None,
    api_key: str,
    model: str,
) -> dict[str, str | None]:
    audio_file_path = Path(audio_path)
    if not audio_file_path.exists():
        raise RuntimeError("Downloaded audio file is missing.")

    audio_size_bytes = audio_file_path.stat().st_size
    if audio_size_bytes > GROQ_MAX_UPLOAD_BYTES:
        raise RuntimeError(
            "Downloaded audio is "
            f"{format_bytes_as_megabytes(byte_count=audio_size_bytes)}, exceeding Groq's 25 MB "
            "direct upload limit. Audio chunking is not implemented."
        )

    content_type, _ = mimetypes.guess_type(audio_file_path.name)
    form_data = {
        "model": model,
        "response_format": "json",
        "temperature": "0",
    }
    if language:
        form_data["language"] = language

    try:
        with audio_file_path.open("rb") as audio_file:
            response = httpx.post(
                GROQ_TRANSCRIPTIONS_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                data=form_data,
                files={
                    "file": (
                        audio_file_path.name,
                        audio_file,
                        content_type or "application/octet-stream",
                    )
                },
                timeout=httpx.Timeout(90.0, connect=10.0),
            )
    except httpx.TimeoutException as exc:
        raise GroqTranscriptionRequestError("Groq transcription request timed out.") from exc
    except httpx.HTTPError as exc:
        raise GroqTranscriptionRequestError(
            f"Groq transcription request failed: {exc}"
        ) from exc

    if response.status_code >= 400:
        raise GroqTranscriptionRequestError(
            format_groq_api_error(response=response),
            status_code=response.status_code,
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise GroqTranscriptionRequestError(
            "Groq transcription returned invalid JSON."
        ) from exc

    raw_text = payload.get("text")
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise GroqTranscriptionRequestError("Groq transcription produced empty text.")

    return {
        "text": normalize_caption_text(raw_text=raw_text),
        "language": language or None,
    }


def format_bytes_as_megabytes(*, byte_count: int) -> str:
    return f"{byte_count / (1024 * 1024):.1f} MB"


def format_groq_api_error(*, response: httpx.Response) -> str:
    message = extract_groq_error_message(response=response)
    if message:
        return f"Groq transcription failed with status {response.status_code}: {message}"

    return f"Groq transcription failed with status {response.status_code}."


def extract_groq_error_message(*, response: httpx.Response) -> str | None:
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        error_payload = payload.get("error")
        if isinstance(error_payload, dict):
            error_message = error_payload.get("message")
            if isinstance(error_message, str) and error_message.strip():
                return error_message.strip()

        direct_message = payload.get("message")
        if isinstance(direct_message, str) and direct_message.strip():
            return direct_message.strip()

    body_text = response.text.strip()
    if not body_text:
        return None

    compact_text = " ".join(body_text.split())
    if len(compact_text) > 300:
        return f"{compact_text[:297]}..."

    return compact_text


def delete_audio_artifacts(*, audio_path: str | None) -> None:
    if not audio_path:
        return

    path = Path(audio_path)
    parent_directory_path = path.parent

    if path.exists():
        path.unlink(missing_ok=True)

    if parent_directory_path.exists() and parent_directory_path.name.startswith("video_finder_audio_"):
        shutil.rmtree(parent_directory_path, ignore_errors=True)
