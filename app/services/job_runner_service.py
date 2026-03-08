from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from time import perf_counter

import aiosqlite
from pydantic import BaseModel, Field

from app.services.transcription_executor_service import TranscribeVideoRequest, transcribe_video

logger = logging.getLogger(__name__)

JOB_STATUS_QUEUED = "queued"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_COMPLETED_WITH_ERRORS = "completed_with_errors"
JOB_STATUS_FAILED = "failed"

ITEM_STATUS_QUEUED = "queued"
ITEM_STATUS_PROCESSING = "processing"
ITEM_STATUS_COMPLETED = "completed"
ITEM_STATUS_FAILED = "failed"
ITEM_STATUS_SKIPPED = "skipped"


class WorkerRunRequest(BaseModel):
    db_path: str
    model_size: str = "turbo"
    language: str | None = None
    compute_type: str = "int8"
    max_concurrency: int = Field(default=1, ge=1)
    poll_seconds: float = Field(default=2.0, gt=0)
    run_once: bool = False


class WorkerRunResult(BaseModel):
    is_successful: bool
    processed_count: int = 0
    error_message: str | None = None


class ClaimedJobItem(BaseModel):
    job_id: str
    video_id: str
    language_override: str | None = None


class ItemProcessResult(BaseModel):
    job_id: str
    video_id: str
    status: str
    duration_seconds: float


class BatchProcessSummary(BaseModel):
    processed_count: int
    completed_count: int
    failed_count: int
    duration_seconds: float


class JobStatusSnapshot(BaseModel):
    job_id: str
    status: str
    total_count: int
    queued_count: int
    processing_count: int
    completed_count: int
    failed_count: int
    skipped_count: int


async def run_transcription_worker(*, request: WorkerRunRequest) -> WorkerRunResult:
    if not request.db_path.strip():
        return WorkerRunResult(is_successful=False, error_message="Database path is missing.")

    processed_count = 0
    logger.info(
        "Transcription worker started model=%s compute_type=%s concurrency=%s fallback_language=%s poll_seconds=%.2f",
        request.model_size,
        request.compute_type,
        request.max_concurrency,
        request.language,
        request.poll_seconds,
    )

    try:
        async with aiosqlite.connect(request.db_path) as db:
            db.row_factory = aiosqlite.Row

            while True:
                claimed_items = await claim_next_job_items(
                    db=db,
                    max_items=request.max_concurrency,
                )
                if not claimed_items:
                    if request.run_once:
                        return WorkerRunResult(is_successful=True, processed_count=processed_count)

                    await asyncio.sleep(request.poll_seconds)
                    continue

                batch_summary = await process_claimed_items(
                    db=db,
                    claimed_items=claimed_items,
                    model_size=request.model_size,
                    language=request.language,
                    compute_type=request.compute_type,
                    max_concurrency=request.max_concurrency,
                )
                processed_count += batch_summary.processed_count
                logger.info(
                    "Transcription batch finished processed=%s completed=%s failed=%s duration=%.2fs",
                    batch_summary.processed_count,
                    batch_summary.completed_count,
                    batch_summary.failed_count,
                    batch_summary.duration_seconds,
                )

                if request.run_once:
                    return WorkerRunResult(is_successful=True, processed_count=processed_count)
    except asyncio.CancelledError:
        logger.info("Transcription worker cancelled")
        return WorkerRunResult(is_successful=True, processed_count=processed_count)
    except Exception as exc:
        logger.exception("Transcription worker failed")
        return WorkerRunResult(
            is_successful=False,
            processed_count=processed_count,
            error_message=f"Transcription worker failed: {exc}",
        )


async def claim_next_job_items(*, db: aiosqlite.Connection, max_items: int) -> list[ClaimedJobItem]:
    if max_items < 1:
        return []

    try:
        await db.execute("BEGIN IMMEDIATE")
        cursor = await db.execute(
            """
            SELECT
                items.job_id AS job_id,
                items.video_id AS video_id,
                jobs.payload_json AS payload_json
            FROM transcription_job_items AS items
            INNER JOIN transcription_jobs AS jobs
                ON jobs.job_id = items.job_id
            WHERE items.status = ?
            ORDER BY items.rowid
            LIMIT ?
            """,
            (ITEM_STATUS_QUEUED, max_items),
        )
        rows = await cursor.fetchall()

        if not rows:
            await db.commit()
            return []

        now = datetime.now(timezone.utc).isoformat()
        claimed_items: list[ClaimedJobItem] = []
        touched_job_ids: set[str] = set()

        for row in rows:
            job_id = row["job_id"]
            video_id = row["video_id"]
            update_cursor = await db.execute(
                """
                UPDATE transcription_job_items
                SET
                    status = ?,
                    error_message = NULL
                WHERE job_id = ?
                AND video_id = ?
                AND status = ?
                """,
                (
                    ITEM_STATUS_PROCESSING,
                    job_id,
                    video_id,
                    ITEM_STATUS_QUEUED,
                ),
            )
            if update_cursor.rowcount == 0:
                continue

            claimed_items.append(
                ClaimedJobItem(
                    job_id=job_id,
                    video_id=video_id,
                    language_override=parse_job_language(payload_json=row["payload_json"]),
                )
            )
            touched_job_ids.add(job_id)

        for job_id in touched_job_ids:
            await db.execute(
                """
                UPDATE transcription_jobs
                SET
                    status = ?,
                    started_at = COALESCE(started_at, ?),
                    finished_at = NULL,
                    error_message = NULL
                WHERE job_id = ?
                """,
                (JOB_STATUS_RUNNING, now, job_id),
            )

        await db.commit()
        return claimed_items
    except Exception:
        await db.rollback()
        raise


async def process_claimed_items(
    *,
    db: aiosqlite.Connection,
    claimed_items: list[ClaimedJobItem],
    model_size: str,
    language: str | None,
    compute_type: str,
    max_concurrency: int,
) -> BatchProcessSummary:
    if not claimed_items:
        return BatchProcessSummary(
            processed_count=0,
            completed_count=0,
            failed_count=0,
            duration_seconds=0,
        )

    semaphore = asyncio.Semaphore(max_concurrency)

    async def process_one_item(*, claimed_item: ClaimedJobItem) -> ItemProcessResult:
        async with semaphore:
            return await process_job_item(
                db=db,
                claimed_item=claimed_item,
                model_size=model_size,
                language=language,
                compute_type=compute_type,
            )

    batch_started_at = perf_counter()
    item_results = await asyncio.gather(
        *(process_one_item(claimed_item=claimed_item) for claimed_item in claimed_items),
        return_exceptions=False,
    )
    batch_duration_seconds = perf_counter() - batch_started_at

    for job_id in {item.job_id for item in claimed_items}:
        snapshot = await recompute_job_status(db=db, job_id=job_id)
        logger.info(
            "Job progress job_id=%s status=%s completed=%s failed=%s processing=%s queued=%s total=%s",
            snapshot.job_id,
            snapshot.status,
            snapshot.completed_count,
            snapshot.failed_count,
            snapshot.processing_count,
            snapshot.queued_count,
            snapshot.total_count,
        )

    completed_count = sum(1 for result in item_results if result.status == ITEM_STATUS_COMPLETED)
    failed_count = sum(1 for result in item_results if result.status == ITEM_STATUS_FAILED)

    return BatchProcessSummary(
        processed_count=len(item_results),
        completed_count=completed_count,
        failed_count=failed_count,
        duration_seconds=batch_duration_seconds,
    )


async def process_job_item(
    *,
    db: aiosqlite.Connection,
    claimed_item: ClaimedJobItem,
    model_size: str,
    language: str | None,
    compute_type: str,
) -> ItemProcessResult:
    item_started_at = perf_counter()
    resolved_language = claimed_item.language_override or language

    transcription_result = await transcribe_video(
        request=TranscribeVideoRequest(
            video_id=claimed_item.video_id,
            model_size=model_size,
            language=resolved_language,
            compute_type=compute_type,
        )
    )

    now = datetime.now(timezone.utc).isoformat()

    try:
        if transcription_result.is_successful and transcription_result.text:
            await db.execute(
                """
                INSERT INTO transcripts (
                    video_id,
                    language,
                    text,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                    language = excluded.language,
                    text = excluded.text,
                    updated_at = excluded.updated_at
                """,
                (
                    claimed_item.video_id,
                    transcription_result.language,
                    transcription_result.text,
                    now,
                    now,
                ),
            )
            await db.execute(
                """
                UPDATE transcription_job_items
                SET
                    status = ?,
                    error_message = NULL
                WHERE job_id = ?
                AND video_id = ?
                """,
                (ITEM_STATUS_COMPLETED, claimed_item.job_id, claimed_item.video_id),
            )
            await db.commit()
            duration_seconds = perf_counter() - item_started_at
            logger.info(
                "Transcription item success job_id=%s video_id=%s resolved_language=%s detected_language=%s duration=%.2fs",
                claimed_item.job_id,
                claimed_item.video_id,
                resolved_language,
                transcription_result.language,
                duration_seconds,
            )
            return ItemProcessResult(
                job_id=claimed_item.job_id,
                video_id=claimed_item.video_id,
                status=ITEM_STATUS_COMPLETED,
                duration_seconds=duration_seconds,
            )

        error_message = transcription_result.error_message or "Transcription failed."
        await db.execute(
            """
            UPDATE transcription_job_items
            SET
                status = ?,
                error_message = ?
            WHERE job_id = ?
            AND video_id = ?
            """,
            (ITEM_STATUS_FAILED, error_message, claimed_item.job_id, claimed_item.video_id),
        )
        await db.commit()
        duration_seconds = perf_counter() - item_started_at
        logger.warning(
            "Transcription item failed job_id=%s video_id=%s resolved_language=%s duration=%.2fs error=%s",
            claimed_item.job_id,
            claimed_item.video_id,
            resolved_language,
            duration_seconds,
            error_message,
        )
        return ItemProcessResult(
            job_id=claimed_item.job_id,
            video_id=claimed_item.video_id,
            status=ITEM_STATUS_FAILED,
            duration_seconds=duration_seconds,
        )
    except Exception as exc:
        await db.rollback()
        logger.exception("Failed to persist job item result", exc_info=exc)
        await db.execute(
            """
            UPDATE transcription_job_items
            SET
                status = ?,
                error_message = ?
            WHERE job_id = ?
            AND video_id = ?
            """,
            (
                ITEM_STATUS_FAILED,
                f"Failed to persist transcript result: {exc}",
                claimed_item.job_id,
                claimed_item.video_id,
            ),
        )
        await db.commit()
        duration_seconds = perf_counter() - item_started_at
        return ItemProcessResult(
            job_id=claimed_item.job_id,
            video_id=claimed_item.video_id,
            status=ITEM_STATUS_FAILED,
            duration_seconds=duration_seconds,
        )


async def recompute_job_status(*, db: aiosqlite.Connection, job_id: str) -> JobStatusSnapshot:
    status_counts = await load_job_item_status_counts(db=db, job_id=job_id)

    queued_count = status_counts.get(ITEM_STATUS_QUEUED, 0)
    processing_count = status_counts.get(ITEM_STATUS_PROCESSING, 0)
    completed_count = status_counts.get(ITEM_STATUS_COMPLETED, 0)
    failed_count = status_counts.get(ITEM_STATUS_FAILED, 0)
    skipped_count = status_counts.get(ITEM_STATUS_SKIPPED, 0)

    total_count = queued_count + processing_count + completed_count + failed_count + skipped_count
    finished_at: str | None = None
    error_message: str | None = None

    if total_count == 0:
        status = JOB_STATUS_FAILED
        finished_at = datetime.now(timezone.utc).isoformat()
        error_message = "Job has no items."
    elif processing_count > 0 or (queued_count > 0 and (completed_count + failed_count + skipped_count) > 0):
        status = JOB_STATUS_RUNNING
    elif queued_count > 0:
        status = JOB_STATUS_QUEUED
    elif failed_count == total_count:
        status = JOB_STATUS_FAILED
        finished_at = datetime.now(timezone.utc).isoformat()
        error_message = "All transcription items failed."
    elif failed_count > 0:
        status = JOB_STATUS_COMPLETED_WITH_ERRORS
        finished_at = datetime.now(timezone.utc).isoformat()
        error_message = f"{failed_count} transcription items failed."
    else:
        status = JOB_STATUS_COMPLETED
        finished_at = datetime.now(timezone.utc).isoformat()

    await db.execute(
        """
        UPDATE transcription_jobs
        SET
            status = ?,
            finished_at = ?,
            error_message = ?
        WHERE job_id = ?
        """,
        (status, finished_at, error_message, job_id),
    )
    await db.commit()

    return JobStatusSnapshot(
        job_id=job_id,
        status=status,
        total_count=total_count,
        queued_count=queued_count,
        processing_count=processing_count,
        completed_count=completed_count,
        failed_count=failed_count,
        skipped_count=skipped_count,
    )


async def load_job_item_status_counts(*, db: aiosqlite.Connection, job_id: str) -> dict[str, int]:
    cursor = await db.execute(
        """
        SELECT
            status,
            COUNT(*) AS status_count
        FROM transcription_job_items
        WHERE job_id = ?
        GROUP BY status
        """,
        (job_id,),
    )
    rows = await cursor.fetchall()
    return {row["status"]: row["status_count"] for row in rows}


def parse_job_language(*, payload_json: str | None) -> str | None:
    if payload_json is None:
        return None

    normalized_payload = payload_json.strip()
    if not normalized_payload:
        return None

    try:
        payload = json.loads(normalized_payload)
    except Exception:
        return None

    language_value = payload.get("language")
    if not isinstance(language_value, str):
        return None

    normalized_language = language_value.strip()
    if not normalized_language:
        return None

    return normalized_language
