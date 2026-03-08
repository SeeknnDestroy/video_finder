from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from app.services.transcription_executor_service import (
    GROQ_MAX_UPLOAD_BYTES,
    CaptionTranscriptResult,
    TranscribeVideoRequest,
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
async def test_transcribe_video_runs_groq_fallback_after_caption_miss(monkeypatch) -> None:
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
        lambda video_id: "/tmp/video-one.webm",
    )

    def fake_run_groq_transcription(*, audio_path: str, language: str | None, api_key: str):
        assert audio_path == "/tmp/video-one.webm"
        assert language == "tr"
        assert api_key == "groq-secret"
        return {"text": "merhaba dunya", "language": "tr"}

    monkeypatch.setattr(
        "app.services.transcription_executor_service.run_groq_transcription",
        fake_run_groq_transcription,
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
        )

    assert "empty text" in str(exc_info.value)
