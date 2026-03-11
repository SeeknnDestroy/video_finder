from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import aiosqlite
from pydantic import BaseModel, Field

from app.core.config import get_groq_transcription_rate_limits


class GroqCapacityReservationRequest(BaseModel):
    db_path: str
    model: str
    audio_seconds: int = Field(ge=1)
    now: datetime | None = None


class GroqCapacityReservationResult(BaseModel):
    is_allowed: bool
    reservation_id: str | None = None
    retry_after_seconds: int | None = None
    error_message: str | None = None


class GroqReservationOutcomeRequest(BaseModel):
    db_path: str
    reservation_id: str
    is_successful: bool
    response_status_code: int | None = None
    error_message: str | None = None
    completed_at: datetime | None = None


class WindowPressure(BaseModel):
    label: str
    current_value: int
    limit_value: int
    retry_after_seconds: int


class GroqRateLimitWindowSnapshot(BaseModel):
    current: int
    limit: int
    remaining: int


class GroqRateLimitSnapshot(BaseModel):
    model: str
    requests_per_minute: GroqRateLimitWindowSnapshot
    requests_per_day: GroqRateLimitWindowSnapshot
    audio_seconds_per_hour: GroqRateLimitWindowSnapshot
    audio_seconds_per_day: GroqRateLimitWindowSnapshot
    total_reservations: int
    last_reserved_at: str | None = None


async def reserve_groq_capacity(
    *,
    request: GroqCapacityReservationRequest,
) -> GroqCapacityReservationResult:
    normalized_db_path = request.db_path.strip()
    if not normalized_db_path:
        return GroqCapacityReservationResult(
            is_allowed=False,
            error_message="Groq rate limiter requires a database path.",
        )

    resolved_now = normalize_datetime_value(raw_value=request.now)
    limits = get_groq_transcription_rate_limits(model=request.model)

    async with aiosqlite.connect(normalized_db_path) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("BEGIN IMMEDIATE")
        try:
            blocking_pressures = [
                pressure
                for pressure in [
                    await evaluate_request_window(
                        db=db,
                        model=request.model,
                        cutoff=resolved_now - timedelta(minutes=1),
                        label="requests per minute",
                        current_increment=1,
                        limit_value=limits.requests_per_minute,
                        now=resolved_now,
                        window_seconds=60,
                    ),
                    await evaluate_request_window(
                        db=db,
                        model=request.model,
                        cutoff=resolved_now - timedelta(days=1),
                        label="requests per day",
                        current_increment=1,
                        limit_value=limits.requests_per_day,
                        now=resolved_now,
                        window_seconds=86_400,
                    ),
                    await evaluate_audio_window(
                        db=db,
                        model=request.model,
                        cutoff=resolved_now - timedelta(hours=1),
                        label="audio seconds per hour",
                        audio_increment=request.audio_seconds,
                        limit_value=limits.audio_seconds_per_hour,
                        now=resolved_now,
                        window_seconds=3_600,
                    ),
                    await evaluate_audio_window(
                        db=db,
                        model=request.model,
                        cutoff=resolved_now - timedelta(days=1),
                        label="audio seconds per day",
                        audio_increment=request.audio_seconds,
                        limit_value=limits.audio_seconds_per_day,
                        now=resolved_now,
                        window_seconds=86_400,
                    ),
                ]
                if pressure is not None
            ]

            if blocking_pressures:
                await db.rollback()
                retry_after_seconds = max(
                    pressure.retry_after_seconds for pressure in blocking_pressures
                )
                pressure_summaries = ", ".join(
                    (
                        f"{pressure.label} would exceed {pressure.limit_value} "
                        f"(current={pressure.current_value})"
                    )
                    for pressure in blocking_pressures
                )
                return GroqCapacityReservationResult(
                    is_allowed=False,
                    retry_after_seconds=retry_after_seconds,
                    error_message=(
                        f"Groq rate limit preflight blocked model {request.model}: "
                        f"{pressure_summaries}. Retry in about {retry_after_seconds}s."
                    ),
                )

            reservation_id = uuid4().hex
            await db.execute(
                """
                INSERT INTO groq_transcription_usage (
                    reservation_id,
                    model,
                    audio_seconds,
                    reserved_at,
                    outcome
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    reservation_id,
                    request.model,
                    request.audio_seconds,
                    resolved_now.isoformat(),
                    "reserved",
                ),
            )
            await db.commit()
            return GroqCapacityReservationResult(
                is_allowed=True,
                reservation_id=reservation_id,
            )
        except Exception:
            await db.rollback()
            raise


async def record_groq_reservation_outcome(*, request: GroqReservationOutcomeRequest) -> None:
    normalized_db_path = request.db_path.strip()
    normalized_reservation_id = request.reservation_id.strip()
    if not normalized_db_path or not normalized_reservation_id:
        return

    resolved_completed_at = normalize_datetime_value(raw_value=request.completed_at)
    outcome = "succeeded" if request.is_successful else "failed"

    async with aiosqlite.connect(normalized_db_path) as db:
        await db.execute(
            """
            UPDATE groq_transcription_usage
            SET
                completed_at = ?,
                outcome = ?,
                response_status_code = ?,
                error_message = ?
            WHERE reservation_id = ?
            """,
            (
                resolved_completed_at.isoformat(),
                outcome,
                request.response_status_code,
                normalize_error_message(error_message=request.error_message),
                normalized_reservation_id,
            ),
        )
        await db.commit()


async def get_groq_rate_limit_snapshot(*, db_path: str, model: str) -> GroqRateLimitSnapshot:
    normalized_db_path = db_path.strip()
    resolved_now = normalize_datetime_value(raw_value=None)
    limits = get_groq_transcription_rate_limits(model=model)

    async with aiosqlite.connect(normalized_db_path) as db:
        db.row_factory = aiosqlite.Row
        minute_rows = await fetch_usage_rows(
            db=db,
            model=model,
            cutoff=resolved_now - timedelta(minutes=1),
        )
        day_rows = await fetch_usage_rows(
            db=db,
            model=model,
            cutoff=resolved_now - timedelta(days=1),
        )
        hour_rows = await fetch_usage_rows(
            db=db,
            model=model,
            cutoff=resolved_now - timedelta(hours=1),
        )
        total_cursor = await db.execute(
            """
            SELECT COUNT(*) AS total_reservations, MAX(reserved_at) AS last_reserved_at
            FROM groq_transcription_usage
            WHERE model = ?
            """,
            (model,),
        )
        totals_row = await total_cursor.fetchone()

    request_minute_count = len(minute_rows)
    request_day_count = len(day_rows)
    audio_hour_count = sum(max(int(row["audio_seconds"]), 0) for row in hour_rows)
    audio_day_count = sum(max(int(row["audio_seconds"]), 0) for row in day_rows)

    return GroqRateLimitSnapshot(
        model=model,
        requests_per_minute=build_window_snapshot(
            current_value=request_minute_count,
            limit_value=limits.requests_per_minute,
        ),
        requests_per_day=build_window_snapshot(
            current_value=request_day_count,
            limit_value=limits.requests_per_day,
        ),
        audio_seconds_per_hour=build_window_snapshot(
            current_value=audio_hour_count,
            limit_value=limits.audio_seconds_per_hour,
        ),
        audio_seconds_per_day=build_window_snapshot(
            current_value=audio_day_count,
            limit_value=limits.audio_seconds_per_day,
        ),
        total_reservations=int(totals_row["total_reservations"]) if totals_row else 0,
        last_reserved_at=(
            str(totals_row["last_reserved_at"])
            if totals_row and totals_row["last_reserved_at"] is not None
            else None
        ),
    )


async def evaluate_request_window(
    *,
    db: aiosqlite.Connection,
    model: str,
    cutoff: datetime,
    label: str,
    current_increment: int,
    limit_value: int,
    now: datetime,
    window_seconds: int,
) -> WindowPressure | None:
    rows = await fetch_usage_rows(db=db, model=model, cutoff=cutoff)
    current_value = len(rows)
    if current_value + current_increment <= limit_value:
        return None

    retry_after_seconds = compute_request_retry_after_seconds(
        rows=rows,
        now=now,
        window_seconds=window_seconds,
    )
    return WindowPressure(
        label=label,
        current_value=current_value,
        limit_value=limit_value,
        retry_after_seconds=retry_after_seconds,
    )


async def evaluate_audio_window(
    *,
    db: aiosqlite.Connection,
    model: str,
    cutoff: datetime,
    label: str,
    audio_increment: int,
    limit_value: int,
    now: datetime,
    window_seconds: int,
) -> WindowPressure | None:
    rows = await fetch_usage_rows(db=db, model=model, cutoff=cutoff)
    current_value = sum(max(int(row["audio_seconds"]), 0) for row in rows)
    if current_value + audio_increment <= limit_value:
        return None

    retry_after_seconds = compute_audio_retry_after_seconds(
        rows=rows,
        required_audio_seconds=(current_value + audio_increment) - limit_value,
        now=now,
        window_seconds=window_seconds,
    )
    return WindowPressure(
        label=label,
        current_value=current_value,
        limit_value=limit_value,
        retry_after_seconds=retry_after_seconds,
    )


async def fetch_usage_rows(
    *,
    db: aiosqlite.Connection,
    model: str,
    cutoff: datetime,
) -> list[aiosqlite.Row]:
    cursor = await db.execute(
        """
        SELECT
            audio_seconds,
            reserved_at
        FROM groq_transcription_usage
        WHERE model = ?
        AND reserved_at >= ?
        ORDER BY reserved_at ASC
        """,
        (model, cutoff.isoformat()),
    )
    return await cursor.fetchall()


def compute_request_retry_after_seconds(
    *,
    rows: list[aiosqlite.Row],
    now: datetime,
    window_seconds: int,
) -> int:
    if not rows:
        return 1

    earliest_reserved_at = parse_datetime(raw_value=rows[0]["reserved_at"])
    return max(1, int((earliest_reserved_at - now).total_seconds()) + window_seconds)


def compute_audio_retry_after_seconds(
    *,
    rows: list[aiosqlite.Row],
    required_audio_seconds: int,
    now: datetime,
    window_seconds: int,
) -> int:
    if not rows:
        return 1

    released_audio_seconds = 0
    for row in rows:
        released_audio_seconds += max(int(row["audio_seconds"]), 0)
        if released_audio_seconds >= required_audio_seconds:
            reserved_at = parse_datetime(raw_value=row["reserved_at"])
            return max(
                1,
                int((reserved_at - now).total_seconds()) + window_seconds,
            )

    last_reserved_at = parse_datetime(raw_value=rows[-1]["reserved_at"])
    return max(1, int((last_reserved_at - now).total_seconds()) + window_seconds)


def parse_datetime(*, raw_value: str) -> datetime:
    normalized_value = raw_value.strip()
    if normalized_value.endswith("Z"):
        normalized_value = normalized_value[:-1] + "+00:00"

    parsed_value = datetime.fromisoformat(normalized_value)
    if parsed_value.tzinfo is None:
        return parsed_value.replace(tzinfo=timezone.utc)

    return parsed_value.astimezone(timezone.utc)


def normalize_datetime_value(*, raw_value: datetime | None) -> datetime:
    if raw_value is None:
        return datetime.now(timezone.utc)

    if raw_value.tzinfo is None:
        return raw_value.replace(tzinfo=timezone.utc)

    return raw_value.astimezone(timezone.utc)


def normalize_error_message(*, error_message: str | None) -> str | None:
    if error_message is None:
        return None

    normalized_error_message = error_message.strip()
    if not normalized_error_message:
        return None

    return normalized_error_message


def build_window_snapshot(
    *,
    current_value: int,
    limit_value: int,
) -> GroqRateLimitWindowSnapshot:
    return GroqRateLimitWindowSnapshot(
        current=current_value,
        limit=limit_value,
        remaining=max(0, limit_value - current_value),
    )
