from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from app.core import config as config_module
from app.db.database import (
    DatabaseConnectionRequest,
    InitializeDatabaseRequest,
    get_database_connection,
    initialize_database,
)
from app.services.groq_rate_limit_service import (
    GroqCapacityReservationRequest,
    GroqReservationOutcomeRequest,
    record_groq_reservation_outcome,
    reserve_groq_capacity,
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


async def seed_job_items(*, db, job_id: str, video_ids: list[str], payload_json: str = "{}") -> None:
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
        """
        INSERT INTO transcription_jobs (job_id, status, payload_json, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (job_id, "queued", payload_json, now),
    )
    for video_id in video_ids:
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


@pytest.mark.asyncio
async def test_worker_marks_items_skipped_on_unrecoverable_video_error(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "worker_skip.db"
    initialize_result = await initialize_database(request=InitializeDatabaseRequest(db_path=str(db_path)))
    assert initialize_result.is_successful

    async with get_database_connection(request=DatabaseConnectionRequest(db_path=str(db_path))) as db:
        await seed_job_item(db=db, job_id="job-skip", video_id="video-skip")

    async def fake_transcribe_video(*, request):
        assert request.video_id == "video-skip"
        return TranscribeVideoResult(
            is_successful=False,
            should_skip=True,
            error_message="Video unavailable. This video has been removed by the uploader",
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
            ("job-skip", "video-skip"),
        )
        item_row = await item_cursor.fetchone()

        job_cursor = await db.execute(
            "SELECT status, error_message FROM transcription_jobs WHERE job_id = ?",
            ("job-skip",),
        )
        job_row = await job_cursor.fetchone()

    assert item_row["status"] == "skipped"
    assert "removed by the uploader" in item_row["error_message"]
    assert job_row["status"] == "completed_with_errors"
    assert "skipped" in job_row["error_message"]


@pytest.mark.asyncio
async def test_worker_rate_limit_behavior_with_concurrency_one(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "worker_rate_limit_single.db"
    initialize_result = await initialize_database(request=InitializeDatabaseRequest(db_path=str(db_path)))
    assert initialize_result.is_successful

    async with get_database_connection(request=DatabaseConnectionRequest(db_path=str(db_path))) as db:
        await seed_job_items(
            db=db,
            job_id="job-rate-limit-single",
            video_ids=["video-one", "video-two"],
        )

    monkeypatch.setitem(
        config_module.GROQ_TRANSCRIPTION_RATE_LIMITS_BY_MODEL,
        "whisper-large-v3-turbo",
        config_module.GroqTranscriptionRateLimits(
            requests_per_minute=10,
            requests_per_day=100,
            audio_seconds_per_hour=100,
            audio_seconds_per_day=1_000,
        ),
    )

    fixed_now = datetime(2026, 3, 11, 14, 0, 0, tzinfo=timezone.utc)

    async def fake_transcribe_video(*, request):
        reservation_result = await reserve_groq_capacity(
            request=GroqCapacityReservationRequest(
                db_path=str(db_path),
                model="whisper-large-v3-turbo",
                audio_seconds=60,
                now=fixed_now,
            )
        )
        if not reservation_result.is_allowed:
            return TranscribeVideoResult(
                is_successful=False,
                error_message=reservation_result.error_message,
            )

        await record_groq_reservation_outcome(
            request=GroqReservationOutcomeRequest(
                db_path=str(db_path),
                reservation_id=reservation_result.reservation_id or "",
                is_successful=True,
                completed_at=fixed_now,
            )
        )
        return TranscribeVideoResult(
            is_successful=True,
            language="en",
            text=f"transcript for {request.video_id}",
        )

    monkeypatch.setattr("app.services.job_runner_service.transcribe_video", fake_transcribe_video)

    first_run_result = await run_transcription_worker(
        request=WorkerRunRequest(
            db_path=str(db_path),
            run_once=True,
            max_concurrency=1,
            poll_seconds=0.01,
        )
    )
    second_run_result = await run_transcription_worker(
        request=WorkerRunRequest(
            db_path=str(db_path),
            run_once=True,
            max_concurrency=1,
            poll_seconds=0.01,
        )
    )

    assert first_run_result.is_successful
    assert second_run_result.is_successful
    assert first_run_result.processed_count == 1
    assert second_run_result.processed_count == 1

    async with get_database_connection(request=DatabaseConnectionRequest(db_path=str(db_path))) as db:
        item_cursor = await db.execute(
            """
            SELECT status, COUNT(*) AS item_count
            FROM transcription_job_items
            WHERE job_id = ?
            GROUP BY status
            """,
            ("job-rate-limit-single",),
        )
        item_rows = await item_cursor.fetchall()

    status_counts = {row["status"]: row["item_count"] for row in item_rows}
    assert status_counts["completed"] == 1
    assert status_counts["failed"] == 1


@pytest.mark.asyncio
async def test_worker_rate_limit_behavior_with_parallel_concurrency(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "worker_rate_limit_parallel.db"
    initialize_result = await initialize_database(request=InitializeDatabaseRequest(db_path=str(db_path)))
    assert initialize_result.is_successful

    async with get_database_connection(request=DatabaseConnectionRequest(db_path=str(db_path))) as db:
        await seed_job_items(
            db=db,
            job_id="job-rate-limit-parallel",
            video_ids=["video-one", "video-two", "video-three"],
        )

    monkeypatch.setitem(
        config_module.GROQ_TRANSCRIPTION_RATE_LIMITS_BY_MODEL,
        "whisper-large-v3-turbo",
        config_module.GroqTranscriptionRateLimits(
            requests_per_minute=10,
            requests_per_day=100,
            audio_seconds_per_hour=100,
            audio_seconds_per_day=1_000,
        ),
    )

    fixed_now = datetime(2026, 3, 11, 15, 0, 0, tzinfo=timezone.utc)

    async def fake_transcribe_video(*, request):
        reservation_result = await reserve_groq_capacity(
            request=GroqCapacityReservationRequest(
                db_path=str(db_path),
                model="whisper-large-v3-turbo",
                audio_seconds=60,
                now=fixed_now,
            )
        )
        if not reservation_result.is_allowed:
            return TranscribeVideoResult(
                is_successful=False,
                error_message=reservation_result.error_message,
            )

        await asyncio.sleep(0.02)
        await record_groq_reservation_outcome(
            request=GroqReservationOutcomeRequest(
                db_path=str(db_path),
                reservation_id=reservation_result.reservation_id or "",
                is_successful=True,
                completed_at=fixed_now,
            )
        )
        return TranscribeVideoResult(
            is_successful=True,
            language="en",
            text=f"parallel transcript for {request.video_id}",
        )

    monkeypatch.setattr("app.services.job_runner_service.transcribe_video", fake_transcribe_video)

    run_result = await run_transcription_worker(
        request=WorkerRunRequest(
            db_path=str(db_path),
            run_once=True,
            max_concurrency=3,
            poll_seconds=0.01,
        )
    )

    assert run_result.is_successful
    assert run_result.processed_count == 3

    async with get_database_connection(request=DatabaseConnectionRequest(db_path=str(db_path))) as db:
        item_cursor = await db.execute(
            """
            SELECT status, COUNT(*) AS item_count
            FROM transcription_job_items
            WHERE job_id = ?
            GROUP BY status
            """,
            ("job-rate-limit-parallel",),
        )
        item_rows = await item_cursor.fetchall()

    status_counts = {row["status"]: row["item_count"] for row in item_rows}
    assert status_counts["completed"] == 1
    assert status_counts["failed"] == 2
