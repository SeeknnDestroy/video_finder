from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.db.database import DatabaseConnectionRequest, InitializeDatabaseRequest, get_database_connection, initialize_database
from app.models.metadata_models import EnsureMetadataCacheRequest, FetchVideoMetadataRequest
from app.services.metadata_service import ensure_metadata_cached, fetch_video_metadata, parse_iso8601_duration


def test_parse_iso8601_duration() -> None:
    assert parse_iso8601_duration(duration_text="PT45S") == 45
    assert parse_iso8601_duration(duration_text="PT1M30S") == 90
    assert parse_iso8601_duration(duration_text="PT2H") == 7200
    assert parse_iso8601_duration(duration_text="invalid") is None


@pytest.mark.asyncio
async def test_ensure_metadata_cached_warns_without_api_key(tmp_path) -> None:
    db_path = tmp_path / "metadata_no_key.db"
    initialize_result = await initialize_database(
        request=InitializeDatabaseRequest(db_path=str(db_path))
    )
    assert initialize_result.is_successful

    async with get_database_connection(
        request=DatabaseConnectionRequest(db_path=str(db_path))
    ) as db:
        result = await ensure_metadata_cached(
            request=EnsureMetadataCacheRequest(
                db=db,
                video_ids=["video-one"],
                api_key=None,
            )
        )

    assert result.fetched_count == 0
    assert result.warning_message is not None


@pytest.mark.asyncio
async def test_fetch_video_metadata_handles_partial_batch_failures(monkeypatch) -> None:
    class DummyResponse:
        def __init__(self, *, status_code: int, payload: dict, text: str = ""):
            self.status_code = status_code
            self._payload = payload
            self.text = text

        def json(self) -> dict:
            return self._payload

    class DummyClient:
        def __init__(self):
            self.call_index = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def get(self, url, params):
            _ = url
            self.call_index += 1
            if self.call_index == 1:
                return DummyResponse(status_code=500, payload={}, text="server error")

            returned_video_id = params["id"].split(",")[0]
            return DummyResponse(
                status_code=200,
                payload={
                    "items": [
                        {
                            "id": returned_video_id,
                            "snippet": {
                                "title": "Recovered title",
                                "channelTitle": "Recovered channel",
                                "thumbnails": {"default": {"url": "https://example.com/thumb.jpg"}},
                            },
                            "contentDetails": {"duration": "PT30S"},
                        }
                    ]
                },
            )

    monkeypatch.setattr(
        "app.services.metadata_service.httpx.AsyncClient",
        lambda timeout: DummyClient(),
    )

    video_ids = [f"video-{index}" for index in range(55)]
    result = await fetch_video_metadata(
        request=FetchVideoMetadataRequest(video_ids=video_ids, api_key="test-key")
    )

    assert result.warning_message is not None
    assert len(result.items) == 1
    assert result.items[0].duration_seconds == 30
    assert result.items[0].fetched_at <= datetime.now(timezone.utc)
