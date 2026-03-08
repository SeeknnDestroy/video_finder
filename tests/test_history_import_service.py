from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.db.database import DatabaseConnectionRequest, InitializeDatabaseRequest, get_database_connection, initialize_database
from app.models.history_models import ParseHistoryRequest, UpsertHistoryRequest, WatchedEventInput
from app.services.history_import_service import parse_watch_history, upsert_watch_history


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "watch_history.json"


def test_parse_watch_history_handles_valid_and_invalid_items() -> None:
    raw_bytes = FIXTURE_PATH.read_bytes()
    result = parse_watch_history(request=ParseHistoryRequest(raw_bytes=raw_bytes))

    assert result.error_message is None
    assert len(result.items) == 2
    assert result.skipped_count == 2
    assert result.items[0].video_id == "abc123xyz11"
    assert result.items[0].source_title == "Quick Pasta Recipe"


@pytest.mark.asyncio
async def test_upsert_watch_history_dedupes_records(tmp_path) -> None:
    db_path = tmp_path / "history.db"
    initialize_result = await initialize_database(
        request=InitializeDatabaseRequest(db_path=str(db_path))
    )
    assert initialize_result.is_successful

    watch_items = [
        WatchedEventInput(
            video_id="abc123xyz11",
            watched_at=datetime(2025, 1, 5, 12, 0, tzinfo=timezone.utc),
            source_title="Quick Pasta Recipe",
            source_channel="CookLab",
            source_url="https://www.youtube.com/watch?v=abc123xyz11",
        )
    ]

    async with get_database_connection(
        request=DatabaseConnectionRequest(db_path=str(db_path))
    ) as db:
        first_result = await upsert_watch_history(
            request=UpsertHistoryRequest(db=db, items=watch_items)
        )
        second_result = await upsert_watch_history(
            request=UpsertHistoryRequest(db=db, items=watch_items)
        )

    assert first_result.inserted_count == 1
    assert first_result.deduped_count == 0
    assert second_result.inserted_count == 0
    assert second_result.deduped_count == 1
