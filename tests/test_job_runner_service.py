from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.db.database import (
    DatabaseConnectionRequest,
    InitializeDatabaseRequest,
    get_database_connection,
    initialize_database,
)
from app.services.job_runner_service import WorkerRunRequest, run_transcription_worker
from app.services.transcription_executor_service import TranscribeVideoResult


async def seed_job_item(*, db, job_id: str, video_id: str, payload_json: str = "{}") -> None:
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
        """
        INSERT INTO transcription_jobs (job_id, status, payload_json, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (job_id, "queued", payload_json, now),
    )
    await db.execute(
        """
        INSERT INTO transcription_job_items (job_id, video_id, status)
        VALUES (?, ?, ?)
        """,
        (job_id, video_id, "queued"),
    )
    await db.commit()


@pytest.mark.asyncio
async def test_worker_successfully_transcribes_and_updates_fts(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "worker_success.db"
    initialize_result = await initialize_database(request=InitializeDatabaseRequest(db_path=str(db_path)))
    assert initialize_result.is_successful

    async with get_database_connection(request=DatabaseConnectionRequest(db_path=str(db_path))) as db:
        await seed_job_item(db=db, job_id="job-success", video_id="video-success")

    async def fake_transcribe_video(*, request):
        assert request.video_id == "video-success"
        return TranscribeVideoResult(
            is_successful=True,
            language="en",
            text="alpha beta gamma",
        )

    monkeypatch.setattr("app.services.job_runner_service.transcribe_video", fake_transcribe_video)

    run_result = await run_transcription_worker(
        request=WorkerRunRequest(
            db_path=str(db_path),
            run_once=True,
            max_concurrency=1,
            poll_seconds=0.01,
        )
    )

    assert run_result.is_successful
    assert run_result.processed_count == 1

    async with get_database_connection(request=DatabaseConnectionRequest(db_path=str(db_path))) as db:
        item_cursor = await db.execute(
            "SELECT status FROM transcription_job_items WHERE job_id = ? AND video_id = ?",
            ("job-success", "video-success"),
        )
        item_row = await item_cursor.fetchone()

        transcript_cursor = await db.execute(
            "SELECT text FROM transcripts WHERE video_id = ?",
            ("video-success",),
        )
        transcript_row = await transcript_cursor.fetchone()

        job_cursor = await db.execute(
            "SELECT status FROM transcription_jobs WHERE job_id = ?",
            ("job-success",),
        )
        job_row = await job_cursor.fetchone()

        fts_cursor = await db.execute(
            "SELECT COUNT(*) AS hit_count FROM transcripts_fts WHERE transcripts_fts MATCH ?",
            ("alpha",),
        )
        fts_row = await fts_cursor.fetchone()

    assert item_row["status"] == "completed"
    assert transcript_row["text"] == "alpha beta gamma"
    assert job_row["status"] == "completed"
    assert fts_row["hit_count"] == 1


@pytest.mark.asyncio
async def test_worker_marks_items_failed_on_transcription_error(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "worker_failure.db"
    initialize_result = await initialize_database(request=InitializeDatabaseRequest(db_path=str(db_path)))
    assert initialize_result.is_successful

    async with get_database_connection(request=DatabaseConnectionRequest(db_path=str(db_path))) as db:
        await seed_job_item(db=db, job_id="job-failure", video_id="video-failure")

    async def fake_transcribe_video(*, request):
        assert request.video_id == "video-failure"
        return TranscribeVideoResult(
            is_successful=False,
            error_message="boom",
        )

    monkeypatch.setattr("app.services.job_runner_service.transcribe_video", fake_transcribe_video)

    run_result = await run_transcription_worker(
        request=WorkerRunRequest(
            db_path=str(db_path),
            run_once=True,
            max_concurrency=1,
            poll_seconds=0.01,
        )
    )

    assert run_result.is_successful
    assert run_result.processed_count == 1

    async with get_database_connection(request=DatabaseConnectionRequest(db_path=str(db_path))) as db:
        item_cursor = await db.execute(
            "SELECT status, error_message FROM transcription_job_items WHERE job_id = ? AND video_id = ?",
            ("job-failure", "video-failure"),
        )
        item_row = await item_cursor.fetchone()

        job_cursor = await db.execute(
            "SELECT status, error_message FROM transcription_jobs WHERE job_id = ?",
            ("job-failure",),
        )
        job_row = await job_cursor.fetchone()

    assert item_row["status"] == "failed"
    assert item_row["error_message"] == "boom"
    assert job_row["status"] == "failed"
    assert job_row["error_message"] is not None


@pytest.mark.asyncio
async def test_worker_uses_job_language_override_from_payload(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "worker_language_override.db"
    initialize_result = await initialize_database(request=InitializeDatabaseRequest(db_path=str(db_path)))
    assert initialize_result.is_successful

    async with get_database_connection(request=DatabaseConnectionRequest(db_path=str(db_path))) as db:
        await seed_job_item(
            db=db,
            job_id="job-language",
            video_id="video-language",
            payload_json='{"language":"tr"}',
        )

    async def fake_transcribe_video(*, request):
        assert request.video_id == "video-language"
        assert request.language == "tr"
        return TranscribeVideoResult(
            is_successful=True,
            language="tr",
            text="merhaba dunya",
        )

    monkeypatch.setattr("app.services.job_runner_service.transcribe_video", fake_transcribe_video)

    run_result = await run_transcription_worker(
        request=WorkerRunRequest(
            db_path=str(db_path),
            language="en",
            run_once=True,
            max_concurrency=1,
            poll_seconds=0.01,
        )
    )

    assert run_result.is_successful
    assert run_result.processed_count == 1
