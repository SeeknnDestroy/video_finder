from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from app.core.config import GroqTranscriptionRateLimits
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


def set_test_limits(monkeypatch, *, rpm: int, rpd: int, audio_hour: int, audio_day: int) -> None:
    monkeypatch.setitem(
        __import__("app.core.config", fromlist=["unused"]).GROQ_TRANSCRIPTION_RATE_LIMITS_BY_MODEL,
        "whisper-large-v3-turbo",
        GroqTranscriptionRateLimits(
            requests_per_minute=rpm,
            requests_per_day=rpd,
            audio_seconds_per_hour=audio_hour,
            audio_seconds_per_day=audio_day,
        ),
    )


@pytest.mark.asyncio
async def test_reserve_groq_capacity_blocks_second_request_in_same_minute(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "groq_rate_limit_requests.db"
    initialize_result = await initialize_database(
        request=InitializeDatabaseRequest(db_path=str(db_path))
    )
    assert initialize_result.is_successful
    set_test_limits(monkeypatch, rpm=1, rpd=10, audio_hour=1_000, audio_day=10_000)

    fixed_now = datetime(2026, 3, 11, 10, 0, 0, tzinfo=timezone.utc)
    first_result = await reserve_groq_capacity(
        request=GroqCapacityReservationRequest(
            db_path=str(db_path),
            model="whisper-large-v3-turbo",
            audio_seconds=30,
            now=fixed_now,
        )
    )
    second_result = await reserve_groq_capacity(
        request=GroqCapacityReservationRequest(
            db_path=str(db_path),
            model="whisper-large-v3-turbo",
            audio_seconds=30,
            now=fixed_now,
        )
    )

    assert first_result.is_allowed is True
    assert second_result.is_allowed is False
    assert second_result.error_message is not None
    assert "requests per minute" in second_result.error_message
    assert second_result.retry_after_seconds == 60


@pytest.mark.asyncio
async def test_reserve_groq_capacity_blocks_when_audio_budget_is_exhausted(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "groq_rate_limit_audio.db"
    initialize_result = await initialize_database(
        request=InitializeDatabaseRequest(db_path=str(db_path))
    )
    assert initialize_result.is_successful
    set_test_limits(monkeypatch, rpm=10, rpd=100, audio_hour=100, audio_day=500)

    fixed_now = datetime(2026, 3, 11, 11, 0, 0, tzinfo=timezone.utc)
    first_result = await reserve_groq_capacity(
        request=GroqCapacityReservationRequest(
            db_path=str(db_path),
            model="whisper-large-v3-turbo",
            audio_seconds=70,
            now=fixed_now,
        )
    )
    second_result = await reserve_groq_capacity(
        request=GroqCapacityReservationRequest(
            db_path=str(db_path),
            model="whisper-large-v3-turbo",
            audio_seconds=40,
            now=fixed_now,
        )
    )

    assert first_result.is_allowed is True
    assert second_result.is_allowed is False
    assert second_result.error_message is not None
    assert "audio seconds per hour" in second_result.error_message
    assert second_result.retry_after_seconds == 3600


@pytest.mark.asyncio
async def test_reserve_groq_capacity_is_safe_under_concurrent_callers(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "groq_rate_limit_concurrency.db"
    initialize_result = await initialize_database(
        request=InitializeDatabaseRequest(db_path=str(db_path))
    )
    assert initialize_result.is_successful
    set_test_limits(monkeypatch, rpm=1, rpd=10, audio_hour=1_000, audio_day=10_000)

    fixed_now = datetime(2026, 3, 11, 12, 0, 0, tzinfo=timezone.utc)

    async def reserve_once():
        return await reserve_groq_capacity(
            request=GroqCapacityReservationRequest(
                db_path=str(db_path),
                model="whisper-large-v3-turbo",
                audio_seconds=30,
                now=fixed_now,
            )
        )

    results = await asyncio.gather(reserve_once(), reserve_once(), reserve_once())

    assert sum(1 for result in results if result.is_allowed) == 1
    assert sum(1 for result in results if not result.is_allowed) == 2


@pytest.mark.asyncio
async def test_record_groq_reservation_outcome_persists_result_details(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "groq_rate_limit_outcome.db"
    initialize_result = await initialize_database(
        request=InitializeDatabaseRequest(db_path=str(db_path))
    )
    assert initialize_result.is_successful
    set_test_limits(monkeypatch, rpm=10, rpd=100, audio_hour=1_000, audio_day=10_000)

    fixed_now = datetime(2026, 3, 11, 13, 0, 0, tzinfo=timezone.utc)
    reservation_result = await reserve_groq_capacity(
        request=GroqCapacityReservationRequest(
            db_path=str(db_path),
            model="whisper-large-v3-turbo",
            audio_seconds=45,
            now=fixed_now,
        )
    )
    assert reservation_result.is_allowed is True
    assert reservation_result.reservation_id is not None

    await record_groq_reservation_outcome(
        request=GroqReservationOutcomeRequest(
            db_path=str(db_path),
            reservation_id=reservation_result.reservation_id,
            is_successful=False,
            response_status_code=429,
            error_message="rate limited upstream",
            completed_at=fixed_now,
        )
    )

    async with get_database_connection(
        request=DatabaseConnectionRequest(db_path=str(db_path))
    ) as db:
        cursor = await db.execute(
            """
            SELECT outcome, response_status_code, error_message
            FROM groq_transcription_usage
            WHERE reservation_id = ?
            """,
            (reservation_result.reservation_id,),
        )
        row = await cursor.fetchone()

    assert row["outcome"] == "failed"
    assert row["response_status_code"] == 429
    assert row["error_message"] == "rate limited upstream"
