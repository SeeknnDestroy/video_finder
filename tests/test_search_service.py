from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from app.db.database import DatabaseConnectionRequest, InitializeDatabaseRequest, get_database_connection, initialize_database
from app.models.search_models import SearchVideosRequest, SearchVideosServiceRequest
from app.services.search_service import resolve_date_range, search_videos


@pytest.mark.asyncio
async def test_search_title_tokens_require_all_terms(tmp_path) -> None:
    db_path = tmp_path / "search_tokens.db"
    initialize_result = await initialize_database(
        request=InitializeDatabaseRequest(db_path=str(db_path))
    )
    assert initialize_result.is_successful

    now = datetime.now(timezone.utc)
    watched_at = now.isoformat()

    async with get_database_connection(
        request=DatabaseConnectionRequest(db_path=str(db_path))
    ) as db:
        await db.execute(
            """
            INSERT INTO watched_events (video_id, watched_at, source_title, source_channel, source_url, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "video-one",
                watched_at,
                "Quick Pasta Recipe",
                "CookLab",
                "https://www.youtube.com/watch?v=video-one",
                watched_at,
            ),
        )
        await db.execute(
            """
            INSERT INTO watched_events (video_id, watched_at, source_title, source_channel, source_url, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "video-two",
                watched_at,
                "Quick Workout",
                "FitNow",
                "https://www.youtube.com/watch?v=video-two",
                watched_at,
            ),
        )
        await db.commit()

        search_result = await search_videos(
            request=SearchVideosServiceRequest(
                db=db,
                search=SearchVideosRequest(title_query="quick recipe", date_preset="6m"),
                api_key=None,
            )
        )

    assert len(search_result.items) == 1
    assert search_result.items[0].video_id == "video-one"


@pytest.mark.asyncio
async def test_duration_filter_skips_missing_metadata(tmp_path) -> None:
    db_path = tmp_path / "search_duration.db"
    initialize_result = await initialize_database(
        request=InitializeDatabaseRequest(db_path=str(db_path))
    )
    assert initialize_result.is_successful

    now = datetime.now(timezone.utc).isoformat()

    async with get_database_connection(
        request=DatabaseConnectionRequest(db_path=str(db_path))
    ) as db:
        await db.execute(
            """
            INSERT INTO watched_events (video_id, watched_at, source_title, source_channel, source_url, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "video-no-metadata",
                now,
                "Uncached Short",
                "Unknown",
                "https://www.youtube.com/watch?v=video-no-metadata",
                now,
            ),
        )
        await db.commit()

        search_result = await search_videos(
            request=SearchVideosServiceRequest(
                db=db,
                search=SearchVideosRequest(duration_max_seconds=60, date_preset="6m"),
                api_key=None,
            )
        )

    assert len(search_result.items) == 0
    assert any("missing duration metadata" in warning.lower() for warning in search_result.warnings)


def test_resolve_date_range_rejects_invalid_custom_range() -> None:
    result = resolve_date_range(
        search=SearchVideosRequest(
            date_preset="custom",
            watched_from=date(2025, 1, 5),
            watched_to=date(2025, 1, 4),
        )
    )

    assert result.error_message is not None


def test_resolve_date_range_has_no_default_filter() -> None:
    result = resolve_date_range(search=SearchVideosRequest())

    assert result.error_message is None
    assert result.applied_preset is None
    assert result.watched_from is None
    assert result.watched_to_exclusive is None


def test_search_request_rejects_invalid_duration_range() -> None:
    with pytest.raises(ValidationError):
        SearchVideosRequest(duration_min_seconds=120, duration_max_seconds=60)
