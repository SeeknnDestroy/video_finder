from __future__ import annotations

import pytest

from app.services.transcription_executor_service import (
    CaptionTranscriptResult,
    TranscribeVideoRequest,
    transcribe_video,
)


@pytest.mark.asyncio
async def test_transcribe_video_uses_captions_before_local_whisper(monkeypatch) -> None:
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
    monkeypatch.setattr(
        "app.services.transcription_executor_service.get_whisper_runtime_error_message",
        lambda: "ffmpeg is not installed or not on PATH.",
    )

    result = await transcribe_video(
        request=TranscribeVideoRequest(video_id="video-one")
    )

    assert result.is_successful is True
    assert result.text == "caption first result"
    assert result.language == "en"


@pytest.mark.asyncio
async def test_transcribe_video_reports_ffmpeg_only_after_caption_miss(monkeypatch) -> None:
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
        "app.services.transcription_executor_service.get_whisper_runtime_error_message",
        lambda: "ffmpeg is not installed or not on PATH.",
    )

    result = await transcribe_video(
        request=TranscribeVideoRequest(video_id="video-two")
    )

    assert result.is_successful is False
    assert result.error_message is not None
    assert "No YouTube captions were available" in result.error_message
    assert "ffmpeg is not installed" in result.error_message
