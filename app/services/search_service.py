from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

from app.models.search_models import (
    ResolveDateRangeResult,
    SearchVideoItem,
    SearchVideosRequest,
    SearchVideosResult,
    SearchVideosServiceRequest,
)
from app.services.metadata_service import ensure_metadata_cached, load_video_metadata_map
from app.models.metadata_models import EnsureMetadataCacheRequest, LoadMetadataMapRequest

CANDIDATE_EVENT_LIMIT = 5000
DATE_PRESET_DAYS = {
    "7d": 7,
    "30d": 30,
    "6m": 183,
    "1y": 365,
}


async def search_videos(*, request: SearchVideosServiceRequest) -> SearchVideosResult:
    date_range = resolve_date_range(search=request.search)
    if date_range.error_message:
        return SearchVideosResult(
            items=[],
            warnings=[date_range.error_message],
            queued_count=0,
            candidate_count=0,
        )

    candidate_rows = await fetch_candidate_events(
        db=request.db,
        watched_from=date_range.watched_from,
        watched_to_exclusive=date_range.watched_to_exclusive,
    )

    unique_video_ids = list(dict.fromkeys(row["video_id"] for row in candidate_rows if row["video_id"]))
    metadata_cache_result = await ensure_metadata_cached(
        request=EnsureMetadataCacheRequest(
            db=request.db,
            video_ids=unique_video_ids,
            api_key=request.api_key,
        )
    )
    metadata_map_result = await load_video_metadata_map(
        request=LoadMetadataMapRequest(db=request.db, video_ids=unique_video_ids)
    )
    metadata_by_video_id = metadata_map_result.metadata_by_video_id

    has_duration_filter = (
        request.search.duration_min_seconds is not None
        or request.search.duration_max_seconds is not None
    )
    title_tokens = tokenize_title_query(title_query=request.search.title_query)

    warnings: list[str] = []
    if metadata_cache_result.warning_message:
        warnings.append(metadata_cache_result.warning_message)

    filtered_items: list[SearchVideoItem] = []
    skipped_missing_duration_count = 0

    for row in candidate_rows:
        video_id = row["video_id"]
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
            SearchVideoItem(
                video_id=video_id,
                video_url=f"https://www.youtube.com/watch?v={video_id}",
                title=resolved_title,
                channel_title=resolve_channel(raw_row=row, metadata=metadata),
                watched_at=watched_at,
                duration_seconds=resolved_duration,
                thumbnail_url=metadata.thumbnail_url if metadata else None,
            )
        )

        if request.search.limit is not None and len(filtered_items) >= request.search.limit:
            break

    if has_duration_filter and skipped_missing_duration_count > 0:
        warnings.append(
            f"Skipped {skipped_missing_duration_count} videos with missing duration metadata."
        )

    if len(candidate_rows) >= CANDIDATE_EVENT_LIMIT:
        warnings.append(
            f"Candidate set reached {CANDIDATE_EVENT_LIMIT} events. Narrow the date range for full coverage."
        )

    return SearchVideosResult(
        items=filtered_items,
        warnings=warnings,
        queued_count=0,
        candidate_count=len(candidate_rows),
    )


async def fetch_candidate_events(*, db, watched_from: datetime | None, watched_to_exclusive: datetime | None):
    query_parts = [
        """
        SELECT
            id,
            video_id,
            watched_at,
            source_title,
            source_channel,
            source_url
        FROM watched_events
        WHERE 1=1
        """
    ]
    query_params: list[object] = []

    if watched_from is not None:
        query_parts.append("AND watched_at >= ?")
        query_params.append(watched_from.isoformat())

    if watched_to_exclusive is not None:
        query_parts.append("AND watched_at < ?")
        query_params.append(watched_to_exclusive.isoformat())

    query_parts.append("ORDER BY watched_at DESC")
    query_parts.append("LIMIT ?")
    query_params.append(CANDIDATE_EVENT_LIMIT)

    final_query = "\n".join(query_parts)
    cursor = await db.execute(final_query, query_params)
    return await cursor.fetchall()


def resolve_date_range(*, search: SearchVideosRequest) -> ResolveDateRangeResult:
    now_utc = datetime.now(timezone.utc)

    if search.date_preset == "custom":
        return build_custom_date_range(
            watched_from=search.watched_from,
            watched_to=search.watched_to,
            applied_preset="custom",
        )

    if search.date_preset in DATE_PRESET_DAYS:
        preset_days = DATE_PRESET_DAYS[search.date_preset]
        watched_from_date = (now_utc - timedelta(days=preset_days)).date()
        watched_to_date = now_utc.date()
        return build_custom_date_range(
            watched_from=watched_from_date,
            watched_to=watched_to_date,
            applied_preset=search.date_preset,
        )

    if search.watched_from is not None or search.watched_to is not None:
        return build_custom_date_range(
            watched_from=search.watched_from,
            watched_to=search.watched_to,
            applied_preset="custom",
        )

    return ResolveDateRangeResult(
        watched_from=None,
        watched_to_exclusive=None,
        applied_preset=None,
    )


def build_custom_date_range(
    *,
    watched_from: date | None,
    watched_to: date | None,
    applied_preset: str,
) -> ResolveDateRangeResult:
    watched_from_datetime = convert_date_to_datetime_start(raw_date=watched_from)
    watched_to_exclusive = convert_date_to_datetime_exclusive_end(raw_date=watched_to)

    if watched_from_datetime and watched_to_exclusive and watched_from_datetime >= watched_to_exclusive:
        return ResolveDateRangeResult(
            error_message="Invalid watched-date range: watched_from must be on or before watched_to.",
            applied_preset=applied_preset,
        )

    return ResolveDateRangeResult(
        watched_from=watched_from_datetime,
        watched_to_exclusive=watched_to_exclusive,
        applied_preset=applied_preset,
    )


def convert_date_to_datetime_start(*, raw_date: date | None) -> datetime | None:
    if raw_date is None:
        return None

    return datetime.combine(raw_date, time.min, tzinfo=timezone.utc)


def convert_date_to_datetime_exclusive_end(*, raw_date: date | None) -> datetime | None:
    if raw_date is None:
        return None

    next_day = raw_date + timedelta(days=1)
    return datetime.combine(next_day, time.min, tzinfo=timezone.utc)


def tokenize_title_query(*, title_query: str | None) -> list[str]:
    if not title_query:
        return []

    tokens = [token.strip().lower() for token in title_query.split() if token.strip()]
    return tokens


def title_matches(*, title: str, tokens: list[str]) -> bool:
    normalized_title = title.lower()
    return all(token in normalized_title for token in tokens)


def resolve_title(*, raw_row, metadata) -> str:
    if metadata and metadata.title:
        return metadata.title

    source_title = raw_row["source_title"]
    if isinstance(source_title, str) and source_title.strip():
        return source_title.strip()

    return "Untitled video"


def resolve_channel(*, raw_row, metadata) -> str | None:
    if metadata and metadata.channel_title:
        return metadata.channel_title

    source_channel = raw_row["source_channel"]
    if isinstance(source_channel, str) and source_channel.strip():
        return source_channel.strip()

    return None


def parse_watched_at(*, raw_value: str | None) -> datetime | None:
    if raw_value is None:
        return None

    normalized_value = raw_value.strip()
    if not normalized_value:
        return None

    if normalized_value.endswith("Z"):
        normalized_value = normalized_value.replace("Z", "+00:00")

    try:
        parsed_value = datetime.fromisoformat(normalized_value)
    except ValueError:
        return None

    if parsed_value.tzinfo is None:
        return parsed_value.replace(tzinfo=timezone.utc)

    return parsed_value.astimezone(timezone.utc)
