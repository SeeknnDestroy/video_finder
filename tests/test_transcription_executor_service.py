from __future__ import annotations

import importlib
from pathlib import Path

import httpx
import pytest

from app.core import config as config_module
from app.services.groq_rate_limit_service import GroqCapacityReservationResult
from app.services.transcription_executor_service import (
    GROQ_MAX_UPLOAD_BYTES,
    CaptionTranscriptResult,
    TranscribeVideoRequest,
    build_yt_dlp_options,
    delete_audio_artifacts,
    download_video_audio,
    fetch_caption_transcript,
    format_youtube_access_error,
    run_groq_transcription,
    transcribe_video,
)


@pytest.mark.asyncio
async def test_transcribe_video_uses_captions_before_groq(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.transcription_executor_service.get_yt_dlp_runtime_error_message",
        lambda: None,
    )
    monkeypatch.setattr(
        "app.services.transcription_executor_service.fetch_caption_transcript",
        lambda video_id, preferred_language: CaptionTranscriptResult(
            text="caption first result",
            language="en",
            error_message=None,
        ),
    )

    def fail_if_called() -> None:
        raise AssertionError("Groq fallback should not be used when captions exist.")

    monkeypatch.setattr(
        "app.services.transcription_executor_service.get_groq_api_key",
        fail_if_called,
    )

    result = await transcribe_video(
        request=TranscribeVideoRequest(video_id="video-one")
    )

    assert result.is_successful is True
    assert result.text == "caption first result"
    assert result.language == "en"


@pytest.mark.asyncio
async def test_transcribe_video_reports_missing_groq_key_after_caption_miss(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.transcription_executor_service.get_yt_dlp_runtime_error_message",
        lambda: None,
    )
    monkeypatch.setattr(
        "app.services.transcription_executor_service.fetch_caption_transcript",
        lambda video_id, preferred_language: CaptionTranscriptResult(
            text=None,
            language=None,
            error_message="No YouTube captions were available for this video.",
        ),
    )
    monkeypatch.setattr(
        "app.services.transcription_executor_service.get_groq_api_key",
        lambda: None,
    )

    result = await transcribe_video(
        request=TranscribeVideoRequest(video_id="video-two")
    )

    assert result.is_successful is False
    assert result.error_message is not None
    assert "No YouTube captions were available" in result.error_message
    assert "GROQ_API_KEY is not configured" in result.error_message


@pytest.mark.asyncio
async def test_transcribe_video_adds_cookie_hint_when_private_video_blocks_captions_and_groq_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("YT_DLP_COOKIES_FROM_BROWSER", "")
    monkeypatch.setenv("YT_DLP_COOKIES_PROFILE", "")
    monkeypatch.setenv("YT_DLP_COOKIES_FILE", "")
    config_module.load_dotenv_files.cache_clear()
    monkeypatch.setattr(
        "app.services.transcription_executor_service.get_yt_dlp_runtime_error_message",
        lambda: None,
    )
    monkeypatch.setattr(
        "app.services.transcription_executor_service.fetch_caption_transcript",
        lambda video_id, preferred_language: CaptionTranscriptResult(
            text=None,
            language=None,
            error_message=(
                "Could not read YouTube caption metadata: ERROR: [youtube] private123: "
                "Private video. Sign in if you've been granted access"
            ),
        ),
    )
    monkeypatch.setattr(
        "app.services.transcription_executor_service.get_groq_api_key",
        lambda: None,
    )

    result = await transcribe_video(
        request=TranscribeVideoRequest(video_id="private123")
    )

    assert result.is_successful is False
    assert result.error_message is not None
    assert "YT_DLP_COOKIES_FROM_BROWSER" in result.error_message
    assert "YT_DLP_COOKIES_FILE" in result.error_message
    assert "GROQ_API_KEY is not configured" in result.error_message


@pytest.mark.asyncio
async def test_transcribe_video_runs_groq_fallback_after_caption_miss(monkeypatch) -> None:
    deleted_audio_paths: list[str | None] = []
    recorded_outcomes: list[tuple[bool, int | None, str | None]] = []

    monkeypatch.setattr(
        "app.services.transcription_executor_service.get_yt_dlp_runtime_error_message",
        lambda: None,
    )
    monkeypatch.setattr(
        "app.services.transcription_executor_service.fetch_caption_transcript",
        lambda video_id, preferred_language: CaptionTranscriptResult(
            text=None,
            language=None,
            error_message="No YouTube captions were available for this video.",
        ),
    )
    monkeypatch.setattr(
        "app.services.transcription_executor_service.get_groq_api_key",
        lambda: "groq-secret",
    )
    monkeypatch.setattr(
        "app.services.transcription_executor_service.download_video_audio",
        lambda video_id: "/tmp/video-one.webm",
    )
    async def fake_resolve_audio_duration_seconds(*, video_id: str) -> int | None:
        return 25

    async def fake_reserve_groq_capacity(*, request):
        assert request.audio_seconds == 25
        assert request.model == "whisper-large-v3-turbo"
        return GroqCapacityReservationResult(is_allowed=True, reservation_id="reservation-1")

    async def fake_record_outcome(*, request) -> None:
        recorded_outcomes.append(
            (request.is_successful, request.response_status_code, request.error_message)
        )

    def fake_run_groq_transcription(
        *,
        audio_path: str,
        language: str | None,
        api_key: str,
        model: str,
    ):
        assert audio_path == "/tmp/video-one.webm"
        assert language == "tr"
        assert api_key == "groq-secret"
        assert model == "whisper-large-v3-turbo"
        return {"text": "merhaba dunya", "language": "tr"}

    monkeypatch.setattr(
        "app.services.transcription_executor_service.resolve_audio_duration_seconds",
        fake_resolve_audio_duration_seconds,
    )
    monkeypatch.setattr(
        "app.services.transcription_executor_service.reserve_groq_capacity",
        fake_reserve_groq_capacity,
    )
    monkeypatch.setattr(
        "app.services.transcription_executor_service.run_groq_transcription",
        fake_run_groq_transcription,
    )
    monkeypatch.setattr(
        "app.services.transcription_executor_service.record_groq_reservation_outcome",
        fake_record_outcome,
    )
    monkeypatch.setattr(
        "app.services.transcription_executor_service.delete_audio_artifacts",
        lambda *, audio_path: deleted_audio_paths.append(audio_path),
    )

    result = await transcribe_video(
        request=TranscribeVideoRequest(video_id="video-one", language="tr")
    )

    assert result.is_successful is True
    assert result.text == "merhaba dunya"
    assert result.language == "tr"
    assert deleted_audio_paths == ["/tmp/video-one.webm"]
    assert recorded_outcomes == [(True, None, None)]


def test_run_groq_transcription_passes_language_hint_and_form_fields(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio_path = tmp_path / "audio.webm"
    audio_path.write_bytes(b"small audio")

    def fake_post(url, *, headers, data, files, timeout):
        assert url == "https://api.groq.com/openai/v1/audio/transcriptions"
        assert headers == {"Authorization": "Bearer groq-secret"}
        assert data == {
            "model": "whisper-large-v3-turbo",
            "response_format": "json",
            "temperature": "0",
            "language": "tr",
        }
        uploaded_name, uploaded_file, uploaded_content_type = files["file"]
        assert uploaded_name == "audio.webm"
        assert uploaded_file.read() == b"small audio"
        assert uploaded_content_type == "video/webm"
        assert timeout is not None
        return httpx.Response(200, json={"text": "merhaba dunya"})

    monkeypatch.setattr("app.services.transcription_executor_service.httpx.post", fake_post)

    result = run_groq_transcription(
        audio_path=str(audio_path),
        language="tr",
        api_key="groq-secret",
        model="whisper-large-v3-turbo",
    )

    assert result == {"text": "merhaba dunya", "language": "tr"}


def test_run_groq_transcription_rejects_oversized_audio_before_api_call(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"")
    audio_path.write_bytes(b"x" * (GROQ_MAX_UPLOAD_BYTES + 1))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Groq API should not be called for oversized audio.")

    monkeypatch.setattr("app.services.transcription_executor_service.httpx.post", fail_if_called)

    with pytest.raises(RuntimeError) as exc_info:
        run_groq_transcription(
            audio_path=str(audio_path),
            language=None,
            api_key="groq-secret",
            model="whisper-large-v3-turbo",
        )

    assert "25 MB direct upload limit" in str(exc_info.value)


def test_run_groq_transcription_surfaces_api_errors(tmp_path: Path, monkeypatch) -> None:
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"small audio")

    def fake_post(*args, **kwargs):
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    monkeypatch.setattr("app.services.transcription_executor_service.httpx.post", fake_post)

    with pytest.raises(RuntimeError) as exc_info:
        run_groq_transcription(
            audio_path=str(audio_path),
            language=None,
            api_key="groq-secret",
            model="whisper-large-v3-turbo",
        )

    assert "status 429" in str(exc_info.value)
    assert "rate limited" in str(exc_info.value)


def test_run_groq_transcription_rejects_empty_text(tmp_path: Path, monkeypatch) -> None:
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"small audio")

    def fake_post(*args, **kwargs):
        return httpx.Response(200, json={"text": "   "})

    monkeypatch.setattr("app.services.transcription_executor_service.httpx.post", fake_post)

    with pytest.raises(RuntimeError) as exc_info:
        run_groq_transcription(
            audio_path=str(audio_path),
            language=None,
            api_key="groq-secret",
            model="whisper-large-v3-turbo",
        )

    assert "empty text" in str(exc_info.value)


@pytest.mark.asyncio
async def test_transcribe_video_rejects_unknown_audio_duration_before_groq_call(monkeypatch) -> None:
    deleted_audio_paths: list[str | None] = []

    monkeypatch.setattr(
        "app.services.transcription_executor_service.get_yt_dlp_runtime_error_message",
        lambda: None,
    )
    monkeypatch.setattr(
        "app.services.transcription_executor_service.fetch_caption_transcript",
        lambda video_id, preferred_language: CaptionTranscriptResult(
            text=None,
            language=None,
            error_message="No YouTube captions were available for this video.",
        ),
    )
    monkeypatch.setattr(
        "app.services.transcription_executor_service.get_groq_api_key",
        lambda: "groq-secret",
    )
    monkeypatch.setattr(
        "app.services.transcription_executor_service.download_video_audio",
        lambda video_id: "/tmp/video-unknown.webm",
    )
    async def fake_resolve_audio_duration_seconds(*, video_id: str) -> int | None:
        return None

    monkeypatch.setattr(
        "app.services.transcription_executor_service.resolve_audio_duration_seconds",
        fake_resolve_audio_duration_seconds,
    )

    async def fail_if_reserved(*, request):
        raise AssertionError("Rate limiter should not reserve unknown audio durations.")

    monkeypatch.setattr(
        "app.services.transcription_executor_service.reserve_groq_capacity",
        fail_if_reserved,
    )
    monkeypatch.setattr(
        "app.services.transcription_executor_service.delete_audio_artifacts",
        lambda *, audio_path: deleted_audio_paths.append(audio_path),
    )

    result = await transcribe_video(
        request=TranscribeVideoRequest(video_id="video-unknown")
    )

    assert result.is_successful is False
    assert result.error_message is not None
    assert "unknown audio lengths" in result.error_message
    assert deleted_audio_paths == ["/tmp/video-unknown.webm"]


@pytest.mark.asyncio
async def test_transcribe_video_reports_local_rate_limit_preflight(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.transcription_executor_service.get_yt_dlp_runtime_error_message",
        lambda: None,
    )
    monkeypatch.setattr(
        "app.services.transcription_executor_service.fetch_caption_transcript",
        lambda video_id, preferred_language: CaptionTranscriptResult(
            text=None,
            language=None,
            error_message="No YouTube captions were available for this video.",
        ),
    )
    monkeypatch.setattr(
        "app.services.transcription_executor_service.get_groq_api_key",
        lambda: "groq-secret",
    )
    monkeypatch.setattr(
        "app.services.transcription_executor_service.download_video_audio",
        lambda video_id: "/tmp/video-rate-limited.webm",
    )
    async def fake_resolve_audio_duration_seconds(*, video_id: str) -> int | None:
        return 25

    monkeypatch.setattr(
        "app.services.transcription_executor_service.resolve_audio_duration_seconds",
        fake_resolve_audio_duration_seconds,
    )

    async def fake_reserve_groq_capacity(*, request):
        return GroqCapacityReservationResult(
            is_allowed=False,
            retry_after_seconds=60,
            error_message="Groq rate limit preflight blocked model whisper-large-v3-turbo.",
        )

    monkeypatch.setattr(
        "app.services.transcription_executor_service.reserve_groq_capacity",
        fake_reserve_groq_capacity,
    )
    monkeypatch.setattr(
        "app.services.transcription_executor_service.delete_audio_artifacts",
        lambda *, audio_path: None,
    )

    result = await transcribe_video(
        request=TranscribeVideoRequest(video_id="video-rate-limited")
    )

    assert result.is_successful is False
    assert result.error_message is not None
    assert "preflight blocked" in result.error_message


def test_build_yt_dlp_options_uses_browser_cookies_when_configured(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("YT_DLP_COOKIES_FROM_BROWSER", "chrome")
    monkeypatch.setenv("YT_DLP_COOKIES_PROFILE", "Profile 1")
    monkeypatch.setenv("YT_DLP_COOKIES_FILE", "/tmp/cookies.txt")
    config_module.load_dotenv_files.cache_clear()

    options = build_yt_dlp_options(quiet=True, noplaylist=True)

    assert options["quiet"] is True
    assert options["noplaylist"] is True
    assert options["cookiesfrombrowser"] == ("chrome", "Profile 1", None, None)
    assert "cookiefile" not in options


def test_build_yt_dlp_options_uses_cookie_file_when_browser_not_configured(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("YT_DLP_COOKIES_FROM_BROWSER", "")
    monkeypatch.setenv("YT_DLP_COOKIES_PROFILE", "")
    monkeypatch.setenv("YT_DLP_COOKIES_FILE", "/tmp/cookies.txt")
    config_module.load_dotenv_files.cache_clear()

    options = build_yt_dlp_options(quiet=True)

    assert options["cookiefile"] == "/tmp/cookies.txt"


def test_fetch_caption_transcript_retries_logged_in_browser_cookies_for_private_video(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("YT_DLP_COOKIES_FROM_BROWSER", "")
    monkeypatch.setenv("YT_DLP_COOKIES_PROFILE", "")
    monkeypatch.setenv("YT_DLP_COOKIES_FILE", "")
    config_module.load_dotenv_files.cache_clear()

    attempts: list[dict[str, object]] = []

    class FakeYoutubeDL:
        def __init__(self, options: dict[str, object]) -> None:
            self.options = dict(options)

        def __enter__(self) -> "FakeYoutubeDL":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def extract_info(self, video_url: str, download: bool = False) -> dict[str, object]:
            assert video_url == "https://www.youtube.com/watch?v=private123"
            assert download is False
            attempts.append(self.options)

            if "cookiesfrombrowser" not in self.options:
                raise RuntimeError(
                    "ERROR: [youtube] private123: Private video. Sign in if you've been granted access"
                )

            return {
                "subtitles": {
                    "en": [
                        {
                            "url": "https://example.com/private123.vtt",
                            "ext": "vtt",
                        }
                    ]
                }
            }

    fake_yt_dlp_module = type("FakeYtDlpModule", (), {"YoutubeDL": FakeYoutubeDL})
    original_import_module = importlib.import_module

    def fake_import_module(module_name: str):
        if module_name == "yt_dlp":
            return fake_yt_dlp_module
        return original_import_module(module_name)

    monkeypatch.setattr(
        "app.services.transcription_executor_service.importlib.import_module",
        fake_import_module,
    )
    monkeypatch.setattr(
        "app.services.transcription_executor_service.platform.system",
        lambda: "Darwin",
    )
    monkeypatch.setattr(
        "app.services.transcription_executor_service.download_text_from_url",
        lambda url: "WEBVTT\n\nhello from private captions",
    )

    result = fetch_caption_transcript(video_id="private123", preferred_language=None)

    assert result.text == "hello from private captions"
    assert result.language == "en"
    assert len(attempts) == 2
    assert "cookiesfrombrowser" not in attempts[0]
    assert attempts[1]["cookiesfrombrowser"] == ("chrome",)


def test_download_video_audio_retries_logged_in_browser_cookies_for_private_video(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("YT_DLP_COOKIES_FROM_BROWSER", "")
    monkeypatch.setenv("YT_DLP_COOKIES_PROFILE", "")
    monkeypatch.setenv("YT_DLP_COOKIES_FILE", "")
    config_module.load_dotenv_files.cache_clear()

    attempts: list[dict[str, object]] = []

    class FakeYoutubeDL:
        def __init__(self, options: dict[str, object]) -> None:
            self.options = dict(options)

        def __enter__(self) -> "FakeYoutubeDL":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def download(self, urls: list[str]) -> None:
            assert urls == ["https://www.youtube.com/watch?v=private123"]
            attempts.append(self.options)

            if "cookiesfrombrowser" not in self.options:
                raise RuntimeError(
                    "ERROR: [youtube] private123: Private video. Sign in if you've been granted access"
                )

            output_template = str(self.options["outtmpl"])
            output_path = Path(output_template.replace("%(ext)s", "webm"))
            output_path.write_bytes(b"private audio bytes")

    fake_yt_dlp_module = type("FakeYtDlpModule", (), {"YoutubeDL": FakeYoutubeDL})
    original_import_module = importlib.import_module

    def fake_import_module(module_name: str):
        if module_name == "yt_dlp":
            return fake_yt_dlp_module
        return original_import_module(module_name)

    monkeypatch.setattr(
        "app.services.transcription_executor_service.importlib.import_module",
        fake_import_module,
    )
    monkeypatch.setattr(
        "app.services.transcription_executor_service.platform.system",
        lambda: "Darwin",
    )

    audio_path = download_video_audio(video_id="private123")

    assert Path(audio_path).exists()
    assert len(attempts) == 2
    assert "cookiesfrombrowser" not in attempts[0]
    assert attempts[1]["cookiesfrombrowser"] == ("chrome",)

    delete_audio_artifacts(audio_path=audio_path)


@pytest.mark.asyncio
async def test_transcribe_video_skips_unrecoverable_removed_video(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.transcription_executor_service.get_yt_dlp_runtime_error_message",
        lambda: None,
    )
    monkeypatch.setattr(
        "app.services.transcription_executor_service.fetch_caption_transcript",
        lambda video_id, preferred_language: CaptionTranscriptResult(
            text=None,
            language=None,
            error_message=(
                "Could not read YouTube caption metadata: ERROR: [youtube] abc123: "
                "Video unavailable. This video has been removed by the uploader"
            ),
        ),
    )

    result = await transcribe_video(request=TranscribeVideoRequest(video_id="abc123"))

    assert result.is_successful is False
    assert result.should_skip is True
    assert result.error_message is not None
    assert "removed by the uploader" in result.error_message


def test_format_youtube_access_error_adds_cookie_hint_for_private_video(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("YT_DLP_COOKIES_FROM_BROWSER", "")
    monkeypatch.setenv("YT_DLP_COOKIES_PROFILE", "")
    monkeypatch.setenv("YT_DLP_COOKIES_FILE", "")
    config_module.load_dotenv_files.cache_clear()

    message = format_youtube_access_error(
        error_message=(
            "ERROR: [youtube] private123: Private video. Sign in if you've been granted access"
        )
    )

    assert "YT_DLP_COOKIES_FROM_BROWSER" in message
    assert "YT_DLP_COOKIES_FILE" in message
    assert "sign into youtube" in message.lower()
