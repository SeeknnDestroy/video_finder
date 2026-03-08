from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import aiosqlite
from pydantic import BaseModel, ConfigDict, Field

from app.models.job_models import (
    CreateTranscriptionJobRequest,
    CreateTranscriptionJobResult,
    CreateTranscriptionJobServiceRequest,
    GetJobStatusServiceRequest,
    JobStatusResult,
    SpokenSearchResult,
    SpokenSearchServiceRequest,
)
from app.models.metadata_models import LoadMetadataMapRequest
from app.models.metadata_models import EnsureMetadataCacheRequest
from app.models.search_models import SearchVideosRequest
from app.services.metadata_service import ensure_metadata_cached, load_video_metadata_map
from app.services.search_service import (
    fetch_candidate_events,
    parse_watched_at,
    resolve_channel,
    resolve_date_range,
    resolve_title,
    title_matches,
    tokenize_title_query,
)

JOB_STATUS_QUEUED = "queued"
ITEM_STATUS_QUEUED = "queued"
ITEM_STATUS_PROCESSING = "processing"
ITEM_STATUS_COMPLETED = "completed"
ITEM_STATUS_FAILED = "failed"
ITEM_STATUS_SKIPPED = "skipped"

IN_FLIGHT_ITEM_STATUSES = (ITEM_STATUS_QUEUED, ITEM_STATUS_PROCESSING)


class CandidateVideo(BaseModel):
    video_id: str
    watched_at: datetime
    title: str
    channel_title: str | None = None
    duration_seconds: int | None = None
    thumbnail_url: str | None = None


class CandidateSelectionRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    db: aiosqlite.Connection
    search: SearchVideosRequest
    api_key: str | None = None


class CandidateSelectionResult(BaseModel):
    items: list[CandidateVideo] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error_message: str | None = None


class QueueVideoIdsRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    db: aiosqlite.Connection
    video_ids: list[str]
    payload_json: str
    language: str | None = None
    max_candidates: int = Field(default=200, ge=1)
    force_retranscribe: bool = False


class QueueVideoIdsResult(BaseModel):
    job_id: str | None = None
    queued_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    error_message: str | None = None


async def create_transcription_job(*, request: CreateTranscriptionJobServiceRequest) -> CreateTranscriptionJobResult:
    search_request = build_search_request_from_job_payload(payload=request.payload)
    candidate_result = await select_candidate_videos(
        request=CandidateSelectionRequest(db=request.db, search=search_request, api_key=request.api_key)
    )
    if candidate_result.error_message:
        return CreateTranscriptionJobResult(error_message=candidate_result.error_message)

    queue_result = await queue_video_ids(
        request=QueueVideoIdsRequest(
            db=request.db,
            video_ids=[item.video_id for item in candidate_result.items],
            payload_json=request.payload.model_dump_json(),
            language=request.payload.language,
            max_candidates=request.max_candidates,
            force_retranscribe=request.payload.force_retranscribe,
        )
    )
    if queue_result.error_message:
        return CreateTranscriptionJobResult(error_message=queue_result.error_message)

    if queue_result.queued_count == 0:
        warning_message = "No eligible videos matched the filters for transcription."
        if queue_result.warnings:
            warning_message = f"{warning_message} {' '.join(queue_result.warnings)}"
        return CreateTranscriptionJobResult(queued_count=0, error_message=warning_message)

    return CreateTranscriptionJobResult(
        job_id=queue_result.job_id,
        queued_count=queue_result.queued_count,
        error_message=None if not queue_result.warnings else " ".join(queue_result.warnings),
    )


async def search_spoken_transcripts(*, request: SpokenSearchServiceRequest) -> SpokenSearchResult:
    if not request.payload.phrase.strip():
        return SpokenSearchResult(error_message="Phrase is required.")

    search_request = SearchVideosRequest(
        title_query=request.payload.title_query,
        duration_min_seconds=request.payload.duration_min_seconds,
        duration_max_seconds=request.payload.duration_max_seconds,
        watched_from=request.payload.watched_from,
        watched_to=request.payload.watched_to,
        date_preset=request.payload.date_preset,
        limit=None,
    )
    candidate_result = await select_candidate_videos(
        request=CandidateSelectionRequest(db=request.db, search=search_request, api_key=request.api_key)
    )
    if candidate_result.error_message:
        return SpokenSearchResult(
            error_message=candidate_result.error_message,
            warnings=candidate_result.warnings,
        )

    candidate_video_ids = [item.video_id for item in candidate_result.items]
    transcript_available_video_ids = await fetch_transcript_available_video_ids(
        db=request.db,
        video_ids=candidate_video_ids,
        language=request.payload.language,
    )
    needs_transcription_count = max(0, len(candidate_video_ids) - len(transcript_available_video_ids))

    transcript_rows, transcript_query_error = await query_transcript_matches(
        db=request.db,
        phrase=request.payload.phrase,
        language=request.payload.language,
        max_rows=max(request.payload.limit * 10, 500),
    )
    if transcript_query_error:
        return SpokenSearchResult(
            error_message=transcript_query_error,
            warnings=candidate_result.warnings,
        )

    candidate_by_video_id = {item.video_id: item for item in candidate_result.items}
    ranked_rows: list[dict[str, object | None]] = []

    for row in transcript_rows:
        video_id = row["video_id"]
        candidate = candidate_by_video_id.get(video_id)
        if candidate is None:
            continue

        rank_score = row["rank_score"]
        normalized_rank_score = float(rank_score) if rank_score is not None else 0.0
        ranked_rows.append(
            {
                "video_id": video_id,
                "rank_score": normalized_rank_score,
                "snippet": row["snippet"],
                "candidate": candidate,
            }
        )

    ranked_rows.sort(
        key=lambda row: (
            row["rank_score"],
            -row["candidate"].watched_at.timestamp(),
        )
    )

    items: list[dict[str, object | None]] = []
    for row in ranked_rows[: request.payload.limit]:
        candidate = row["candidate"]
        video_id = row["video_id"]
        items.append(
            {
                "video_id": video_id,
                "title": candidate.title,
                "video_url": f"https://www.youtube.com/watch?v={video_id}",
                "thumbnail_url": candidate.thumbnail_url,
                "channel_title": candidate.channel_title,
                "duration_seconds": candidate.duration_seconds,
                "snippet": row["snippet"],
                "watched_at": candidate.watched_at.isoformat(),
            }
        )

    queue_result = await queue_video_ids(
        request=QueueVideoIdsRequest(
            db=request.db,
            video_ids=candidate_video_ids,
            payload_json=json.dumps(
                {
                    "source": "spoken_search_auto_queue",
                    "phrase": request.payload.phrase,
                    "title_query": request.payload.title_query,
                    "duration_min_seconds": request.payload.duration_min_seconds,
                    "duration_max_seconds": request.payload.duration_max_seconds,
                    "watched_from": request.payload.watched_from.isoformat() if request.payload.watched_from else None,
                    "watched_to": request.payload.watched_to.isoformat() if request.payload.watched_to else None,
                    "date_preset": request.payload.date_preset,
                    "language": request.payload.language,
                }
            ),
            language=request.payload.language,
            max_candidates=request.max_candidates,
            force_retranscribe=False,
        )
    )

    warning_messages = [*candidate_result.warnings, *queue_result.warnings]
    if queue_result.error_message:
        warning_messages.append(queue_result.error_message)

    return SpokenSearchResult(
        items=items,
        queued_count=queue_result.queued_count,
        candidate_count=len(candidate_video_ids),
        transcript_available_count=len(transcript_available_video_ids),
        needs_transcription_count=needs_transcription_count,
        warnings=warning_messages,
        job_id=queue_result.job_id,
    )


async def get_transcription_job_status(
    *,
    request: GetJobStatusServiceRequest,
) -> JobStatusResult | None:
    job_cursor = await request.db.execute(
        """
        SELECT
            job_id,
            status,
            created_at,
            started_at,
            finished_at,
            error_message
        FROM transcription_jobs
        WHERE job_id = ?
        """,
        (request.job_id,),
    )
    job_row = await job_cursor.fetchone()
    if job_row is None:
        return None

    counts_by_status = await fetch_job_item_counts_by_status(db=request.db, job_id=request.job_id)
    items: list[dict[str, str | None]] = []

    if request.include_items:
        item_cursor = await request.db.execute(
            """
            SELECT
                video_id,
                status,
                error_message
            FROM transcription_job_items
            WHERE job_id = ?
            ORDER BY
                CASE status
                    WHEN 'failed' THEN 0
                    WHEN 'processing' THEN 1
                    WHEN 'queued' THEN 2
                    WHEN 'skipped' THEN 3
                    ELSE 4
                END,
                video_id
            """,
            (request.job_id,),
        )
        item_rows = await item_cursor.fetchall()
        items = [
            {
                "video_id": row["video_id"],
                "status": row["status"],
                "error_message": row["error_message"],
            }
            for row in item_rows
        ]

    return JobStatusResult(
        job_id=job_row["job_id"],
        status=job_row["status"],
        created_at=job_row["created_at"],
        started_at=job_row["started_at"],
        finished_at=job_row["finished_at"],
        error_message=job_row["error_message"],
        total_count=sum(counts_by_status.values()),
        queued_count=counts_by_status.get(ITEM_STATUS_QUEUED, 0),
        processing_count=counts_by_status.get(ITEM_STATUS_PROCESSING, 0),
        completed_count=counts_by_status.get(ITEM_STATUS_COMPLETED, 0),
        failed_count=counts_by_status.get(ITEM_STATUS_FAILED, 0),
        skipped_count=counts_by_status.get(ITEM_STATUS_SKIPPED, 0),
        items=items,
    )


def build_search_request_from_job_payload(*, payload: CreateTranscriptionJobRequest) -> SearchVideosRequest:
    return SearchVideosRequest(
        title_query=payload.title_query,
        duration_min_seconds=payload.duration_min_seconds,
        duration_max_seconds=payload.duration_max_seconds,
        watched_from=payload.watched_from,
        watched_to=payload.watched_to,
        date_preset=payload.date_preset,
        limit=None,
    )


async def select_candidate_videos(*, request: CandidateSelectionRequest) -> CandidateSelectionResult:
    date_range_result = resolve_date_range(search=request.search)
    if date_range_result.error_message:
        return CandidateSelectionResult(error_message=date_range_result.error_message)

    candidate_rows = await fetch_candidate_events(
        db=request.db,
        watched_from=date_range_result.watched_from,
        watched_to_exclusive=date_range_result.watched_to_exclusive,
    )
    unique_video_ids = list(dict.fromkeys(row["video_id"] for row in candidate_rows if row["video_id"]))
    metadata_cache_result = await ensure_metadata_cached(
        request=EnsureMetadataCacheRequest(
            db=request.db,
            video_ids=unique_video_ids,
            api_key=request.api_key,
        )
    )
    metadata_result = await load_video_metadata_map(
        request=LoadMetadataMapRequest(db=request.db, video_ids=unique_video_ids)
    )
    metadata_by_video_id = metadata_result.metadata_by_video_id

    title_tokens = tokenize_title_query(title_query=request.search.title_query)
    has_duration_filter = (
        request.search.duration_min_seconds is not None
        or request.search.duration_max_seconds is not None
    )

    warnings: list[str] = []
    if metadata_cache_result.warning_message:
        warnings.append(metadata_cache_result.warning_message)
    skipped_missing_duration_count = 0
    seen_video_ids: set[str] = set()
    filtered_items: list[CandidateVideo] = []

    for row in candidate_rows:
        video_id = row["video_id"]
        if not video_id or video_id in seen_video_ids:
            continue

        seen_video_ids.add(video_id)
        metadata = metadata_by_video_id.get(video_id)

        resolved_title = resolve_title(raw_row=row, metadata=metadata)
        if title_tokens and not title_matches(title=resolved_title, tokens=title_tokens):
            continue

        resolved_duration = metadata.duration_seconds if metadata else None
        if has_duration_filter:
            if resolved_duration is None:
                skipped_missing_duration_count += 1
                continue

            if (
                request.search.duration_min_seconds is not None
                and resolved_duration < request.search.duration_min_seconds
            ):
                continue

            if (
                request.search.duration_max_seconds is not None
                and resolved_duration > request.search.duration_max_seconds
            ):
                continue

        watched_at = parse_watched_at(raw_value=row["watched_at"])
        if watched_at is None:
            continue

        filtered_items.append(
            CandidateVideo(
                video_id=video_id,
                watched_at=watched_at,
                title=resolved_title,
                channel_title=resolve_channel(raw_row=row, metadata=metadata),
                duration_seconds=resolved_duration,
                thumbnail_url=metadata.thumbnail_url if metadata else None,
            )
        )

    if has_duration_filter and skipped_missing_duration_count > 0:
        warnings.append(
            f"Skipped {skipped_missing_duration_count} videos with missing duration metadata."
        )

    return CandidateSelectionResult(items=filtered_items, warnings=warnings)


async def queue_video_ids(*, request: QueueVideoIdsRequest) -> QueueVideoIdsResult:
    unique_video_ids = list(dict.fromkeys(video_id for video_id in request.video_ids if video_id))
    if not unique_video_ids:
        return QueueVideoIdsResult(job_id=None, queued_count=0)

    warnings: list[str] = []
    in_flight_video_ids = await fetch_in_flight_video_ids(db=request.db, video_ids=unique_video_ids)
    filtered_video_ids = [video_id for video_id in unique_video_ids if video_id not in in_flight_video_ids]

    if not request.force_retranscribe:
        existing_transcript_video_ids = await fetch_existing_transcript_video_ids(
            db=request.db,
            video_ids=filtered_video_ids,
            language=request.language,
        )
        filtered_video_ids = [
            video_id
            for video_id in filtered_video_ids
            if video_id not in existing_transcript_video_ids
        ]

    if len(filtered_video_ids) > request.max_candidates:
        warnings.append(
            f"Capped queued videos to {request.max_candidates} out of {len(filtered_video_ids)} candidates."
        )
        filtered_video_ids = filtered_video_ids[: request.max_candidates]

    if not filtered_video_ids:
        return QueueVideoIdsResult(job_id=None, queued_count=0, warnings=warnings)

    job_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    try:
        await request.db.execute(
            """
            INSERT INTO transcription_jobs (
                job_id,
                status,
                payload_json,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (job_id, JOB_STATUS_QUEUED, request.payload_json, created_at),
        )
        for video_id in filtered_video_ids:
            await request.db.execute(
                """
                INSERT INTO transcription_job_items (
                    job_id,
                    video_id,
                    status
                )
                VALUES (?, ?, ?)
                ON CONFLICT(job_id, video_id) DO NOTHING
                """,
                (job_id, video_id, ITEM_STATUS_QUEUED),
            )
        await request.db.commit()
    except Exception as exc:
        await request.db.rollback()
        return QueueVideoIdsResult(
            error_message=f"Failed to create transcription job: {exc}",
            warnings=warnings,
        )

    return QueueVideoIdsResult(
        job_id=job_id,
        queued_count=len(filtered_video_ids),
        warnings=warnings,
    )


async def fetch_in_flight_video_ids(*, db: aiosqlite.Connection, video_ids: list[str]) -> set[str]:
    if not video_ids:
        return set()

    placeholders = ", ".join("?" for _ in video_ids)
    status_placeholders = ", ".join("?" for _ in IN_FLIGHT_ITEM_STATUSES)
    cursor = await db.execute(
        f"""
        SELECT DISTINCT video_id
        FROM transcription_job_items
        WHERE video_id IN ({placeholders})
        AND status IN ({status_placeholders})
        """,
        [*video_ids, *IN_FLIGHT_ITEM_STATUSES],
    )
    rows = await cursor.fetchall()
    return {row["video_id"] for row in rows}


async def fetch_existing_transcript_video_ids(
    *,
    db: aiosqlite.Connection,
    video_ids: list[str],
    language: str | None,
) -> set[str]:
    return await fetch_transcript_available_video_ids(
        db=db,
        video_ids=video_ids,
        language=language,
    )


async def fetch_transcript_available_video_ids(
    *,
    db: aiosqlite.Connection,
    video_ids: list[str],
    language: str | None,
) -> set[str]:
    if not video_ids:
        return set()

    placeholders = ", ".join("?" for _ in video_ids)
    query_parts = [
        f"""
        SELECT video_id
        FROM transcripts
        WHERE video_id IN ({placeholders})
        """
    ]
    query_params: list[object] = [*video_ids]

    normalized_language = normalize_language_value(language=language)
    if normalized_language:
        query_parts.append("AND (lower(language) = ? OR lower(language) LIKE ?)")
        query_params.append(normalized_language)
        query_params.append(f"{normalized_language}-%")

    cursor = await db.execute("\n".join(query_parts), query_params)
    rows = await cursor.fetchall()
    return {row["video_id"] for row in rows}


async def query_transcript_matches(
    *,
    db: aiosqlite.Connection,
    phrase: str,
    language: str | None,
    max_rows: int,
) -> tuple[list[aiosqlite.Row], str | None]:
    query_parts = [
        """
        SELECT
            transcripts.video_id AS video_id,
            snippet(transcripts_fts, 1, '<mark>', '</mark>', ' … ', 14) AS snippet,
            bm25(transcripts_fts) AS rank_score
        FROM transcripts_fts
        INNER JOIN transcripts
            ON transcripts.video_id = transcripts_fts.video_id
        WHERE transcripts_fts MATCH ?
        """
    ]
    query_params: list[object] = [phrase]

    normalized_language = normalize_language_value(language=language)
    if normalized_language:
        query_parts.append("AND (lower(transcripts.language) = ? OR lower(transcripts.language) LIKE ?)")
        query_params.append(normalized_language)
        query_params.append(f"{normalized_language}-%")

    query_parts.append("ORDER BY rank_score ASC")
    query_parts.append("LIMIT ?")
    query_params.append(max_rows)

    try:
        cursor = await db.execute("\n".join(query_parts), query_params)
        rows = await cursor.fetchall()
    except Exception as exc:
        return [], f"Invalid spoken query syntax: {exc}"

    return rows, None


async def fetch_job_item_counts_by_status(*, db: aiosqlite.Connection, job_id: str) -> dict[str, int]:
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


def normalize_language_value(*, language: str | None) -> str | None:
    if language is None:
        return None

    normalized_language = language.strip().lower()
    if not normalized_language:
        return None

    return normalized_language
