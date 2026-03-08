from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.db.database import (
    DatabaseConnectionRequest,
    InitializeDatabaseRequest,
    get_database_connection,
    initialize_database,
)
from app.models.job_models import (
    CreateTranscriptionJobRequest,
    CreateTranscriptionJobServiceRequest,
    SpokenSearchRequest,
    SpokenSearchServiceRequest,
)
from app.services.transcription_service import create_transcription_job, search_spoken_transcripts


async def insert_watched_event(*, db, video_id: str, watched_at: datetime, title: str) -> None:
    await db.execute(
        """
        INSERT INTO watched_events (video_id, watched_at, source_title, source_channel, source_url, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            video_id,
            watched_at.isoformat(),
            title,
            "Channel",
            f"https://www.youtube.com/watch?v={video_id}",
            datetime.now(timezone.utc).isoformat(),
        ),
    )


@pytest.mark.asyncio
async def test_create_transcription_job_allows_backfill_and_applies_cap(tmp_path) -> None:
    db_path = tmp_path / "transcription_jobs_cap.db"
    initialize_result = await initialize_database(request=InitializeDatabaseRequest(db_path=str(db_path)))
    assert initialize_result.is_successful

    now = datetime.now(timezone.utc)

    async with get_database_connection(request=DatabaseConnectionRequest(db_path=str(db_path))) as db:
        await insert_watched_event(db=db, video_id="video-1", watched_at=now, title="One")
        await insert_watched_event(db=db, video_id="video-2", watched_at=now - timedelta(minutes=1), title="Two")
        await insert_watched_event(db=db, video_id="video-3", watched_at=now - timedelta(minutes=2), title="Three")
        await db.commit()

        result = await create_transcription_job(
            request=CreateTranscriptionJobServiceRequest(
                db=db,
                payload=CreateTranscriptionJobRequest(),
                max_candidates=2,
            )
        )

        assert result.job_id is not None
        assert result.queued_count == 2

        cursor = await db.execute(
            "SELECT COUNT(*) AS item_count FROM transcription_job_items WHERE job_id = ?",
            (result.job_id,),
        )
        row = await cursor.fetchone()

    assert row["item_count"] == 2


@pytest.mark.asyncio
async def test_create_transcription_job_skips_existing_transcripts_without_force(tmp_path) -> None:
    db_path = tmp_path / "transcription_jobs_existing.db"
    initialize_result = await initialize_database(request=InitializeDatabaseRequest(db_path=str(db_path)))
    assert initialize_result.is_successful

    now = datetime.now(timezone.utc)

    async with get_database_connection(request=DatabaseConnectionRequest(db_path=str(db_path))) as db:
        await insert_watched_event(db=db, video_id="video-1", watched_at=now, title="One")
        await db.execute(
            """
            INSERT INTO transcripts (video_id, language, text, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("video-1", "en", "already transcribed", now.isoformat(), now.isoformat()),
        )
        await db.commit()

        result = await create_transcription_job(
            request=CreateTranscriptionJobServiceRequest(
                db=db,
                payload=CreateTranscriptionJobRequest(force_retranscribe=False),
                max_candidates=10,
            )
        )

        assert result.job_id is None
        assert result.queued_count == 0
        assert result.error_message is not None

        cursor = await db.execute("SELECT COUNT(*) AS job_count FROM transcription_jobs")
        row = await cursor.fetchone()

    assert row["job_count"] == 0


@pytest.mark.asyncio
async def test_spoken_search_returns_matches_and_auto_queues_missing_transcripts(tmp_path) -> None:
    db_path = tmp_path / "spoken_search_queue.db"
    initialize_result = await initialize_database(request=InitializeDatabaseRequest(db_path=str(db_path)))
    assert initialize_result.is_successful

    now = datetime.now(timezone.utc)

    async with get_database_connection(request=DatabaseConnectionRequest(db_path=str(db_path))) as db:
        await insert_watched_event(db=db, video_id="spoken-hit", watched_at=now, title="Hit")
        await insert_watched_event(
            db=db,
            video_id="missing-transcript",
            watched_at=now - timedelta(minutes=1),
            title="Missing",
        )
        await db.execute(
            """
            INSERT INTO transcripts (video_id, language, text, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "spoken-hit",
                "en",
                "hello there this is a spoken transcript",
                now.isoformat(),
                now.isoformat(),
            ),
        )
        await db.commit()

        result = await search_spoken_transcripts(
            request=SpokenSearchServiceRequest(
                db=db,
                payload=SpokenSearchRequest(phrase="hello", limit=20),
                max_candidates=10,
            )
        )

        assert result.error_message is None
        assert len(result.items) == 1
        assert result.items[0].video_id == "spoken-hit"
        assert result.queued_count == 1
        assert result.candidate_count == 2
        assert result.transcript_available_count == 1
        assert result.needs_transcription_count == 1
        assert result.job_id is not None

        cursor = await db.execute(
            """
            SELECT video_id
            FROM transcription_job_items
            WHERE job_id = ?
            """,
            (result.job_id,),
        )
        rows = await cursor.fetchall()

    assert [row["video_id"] for row in rows] == ["missing-transcript"]


@pytest.mark.asyncio
async def test_spoken_search_language_filter_counts_language_specific_transcripts(tmp_path) -> None:
    db_path = tmp_path / "spoken_search_language_counts.db"
    initialize_result = await initialize_database(request=InitializeDatabaseRequest(db_path=str(db_path)))
    assert initialize_result.is_successful

    now = datetime.now(timezone.utc)

    async with get_database_connection(request=DatabaseConnectionRequest(db_path=str(db_path))) as db:
        await insert_watched_event(db=db, video_id="video-tr-needed", watched_at=now, title="Need TR")
        await db.execute(
            """
            INSERT INTO transcripts (video_id, language, text, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "video-tr-needed",
                "en",
                "hello this transcript is in english",
                now.isoformat(),
                now.isoformat(),
            ),
        )
        await db.commit()

        result = await search_spoken_transcripts(
            request=SpokenSearchServiceRequest(
                db=db,
                payload=SpokenSearchRequest(phrase="hello", language="tr", limit=20),
                max_candidates=10,
            )
        )

        assert result.error_message is None
        assert result.candidate_count == 1
        assert result.transcript_available_count == 0
        assert result.needs_transcription_count == 1
        assert result.queued_count == 1
