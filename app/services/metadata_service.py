from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import httpx

from app.models.metadata_models import (
    EnsureMetadataCacheRequest,
    EnsureMetadataCacheResult,
    FetchVideoMetadataRequest,
    FetchVideoMetadataResult,
    LoadMetadataMapRequest,
    LoadMetadataMapResult,
    VideoMetadataItem,
)

logger = logging.getLogger(__name__)

YOUTUBE_VIDEOS_API_URL = "https://www.googleapis.com/youtube/v3/videos"
MAX_YOUTUBE_BATCH_SIZE = 50
_DURATION_PATTERN = re.compile(
    r"P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
)


async def fetch_video_metadata(*, request: FetchVideoMetadataRequest) -> FetchVideoMetadataResult:
    unique_video_ids = list(dict.fromkeys(video_id for video_id in request.video_ids if video_id))
    if not unique_video_ids:
        return FetchVideoMetadataResult(items=[], missing_video_ids=[])

    collected_items: list[VideoMetadataItem] = []
    seen_video_ids: set[str] = set()
    warnings: list[str] = []

    async with httpx.AsyncClient(timeout=20.0) as client:
        for start_index in range(0, len(unique_video_ids), MAX_YOUTUBE_BATCH_SIZE):
            current_batch = unique_video_ids[start_index : start_index + MAX_YOUTUBE_BATCH_SIZE]
            params = {
                "part": "snippet,contentDetails",
                "id": ",".join(current_batch),
                "key": request.api_key,
                "maxResults": str(MAX_YOUTUBE_BATCH_SIZE),
            }

            try:
                response = await client.get(YOUTUBE_VIDEOS_API_URL, params=params)
            except httpx.HTTPError as exc:
                logger.warning("Metadata request failed for batch", exc_info=exc)
                warnings.append(f"Metadata request failed for batch starting at index {start_index}.")
                continue

            if response.status_code >= 400:
                logger.warning("Metadata API error: %s", response.text)
                warnings.append(
                    f"Metadata API returned status {response.status_code} for batch starting at index {start_index}."
                )
                continue

            payload = response.json()
            if payload.get("error"):
                warnings.append("Metadata API returned an error payload.")
                continue

            for raw_item in payload.get("items", []):
                video_id = raw_item.get("id")
                if not isinstance(video_id, str):
                    continue

                snippet = raw_item.get("snippet") or {}
                content_details = raw_item.get("contentDetails") or {}
                thumbnails = snippet.get("thumbnails") or {}

                seen_video_ids.add(video_id)
                collected_items.append(
                    VideoMetadataItem(
                        video_id=video_id,
                        title=extract_text(snippet.get("title")),
                        channel_title=extract_text(snippet.get("channelTitle")),
                        duration_seconds=parse_iso8601_duration(
                            duration_text=extract_text(content_details.get("duration"))
                        ),
                        thumbnail_url=extract_thumbnail_url(thumbnails=thumbnails),
                        is_available=True,
                        fetched_at=datetime.now(timezone.utc),
                    )
                )

    missing_video_ids = [video_id for video_id in unique_video_ids if video_id not in seen_video_ids]

    warning_message = " ".join(warnings) if warnings else None
    return FetchVideoMetadataResult(
        items=collected_items,
        missing_video_ids=missing_video_ids,
        warning_message=warning_message,
    )


async def ensure_metadata_cached(*, request: EnsureMetadataCacheRequest) -> EnsureMetadataCacheResult:
    unique_video_ids = list(dict.fromkeys(video_id for video_id in request.video_ids if video_id))
    if not unique_video_ids:
        return EnsureMetadataCacheResult(fetched_count=0, cached_count=0)

    existing_video_ids = await fetch_existing_video_ids(db=request.db, video_ids=unique_video_ids)
    missing_video_ids = [video_id for video_id in unique_video_ids if video_id not in existing_video_ids]

    if not missing_video_ids:
        return EnsureMetadataCacheResult(
            fetched_count=0,
            cached_count=len(existing_video_ids),
        )

    if not request.api_key:
        return EnsureMetadataCacheResult(
            fetched_count=0,
            cached_count=len(existing_video_ids),
            warning_message="YOUTUBE_API_KEY is missing. Metadata enrichment was skipped.",
        )

    fetch_result = await fetch_video_metadata(
        request=FetchVideoMetadataRequest(video_ids=missing_video_ids, api_key=request.api_key)
    )

    await persist_metadata_items(db=request.db, items=fetch_result.items)
    resolved_video_ids = {item.video_id for item in fetch_result.items}
    unresolved_video_ids = [
        video_id
        for video_id in missing_video_ids
        if video_id not in resolved_video_ids
    ]
    await persist_unavailable_metadata(db=request.db, video_ids=unresolved_video_ids)
    await request.db.commit()

    return EnsureMetadataCacheResult(
        fetched_count=len(fetch_result.items),
        cached_count=len(existing_video_ids),
        warning_message=fetch_result.warning_message,
    )


async def load_video_metadata_map(*, request: LoadMetadataMapRequest) -> LoadMetadataMapResult:
    unique_video_ids = list(dict.fromkeys(video_id for video_id in request.video_ids if video_id))
    if not unique_video_ids:
        return LoadMetadataMapResult(metadata_by_video_id={})

    placeholders = ", ".join("?" for _ in unique_video_ids)
    cursor = await request.db.execute(
        f"""
        SELECT
            video_id,
            title,
            channel_title,
            duration_seconds,
            thumbnail_url,
            is_available,
            fetched_at
        FROM video_metadata
        WHERE video_id IN ({placeholders})
        """,
        unique_video_ids,
    )
    rows = await cursor.fetchall()

    metadata_by_video_id: dict[str, VideoMetadataItem] = {}
    for row in rows:
        fetched_at = parse_datetime_or_now(raw_value=row["fetched_at"])
        metadata_by_video_id[row["video_id"]] = VideoMetadataItem(
            video_id=row["video_id"],
            title=row["title"],
            channel_title=row["channel_title"],
            duration_seconds=row["duration_seconds"],
            thumbnail_url=row["thumbnail_url"],
            is_available=bool(row["is_available"]),
            fetched_at=fetched_at,
        )

    return LoadMetadataMapResult(metadata_by_video_id=metadata_by_video_id)


async def fetch_existing_video_ids(*, db, video_ids: list[str]) -> set[str]:
    if not video_ids:
        return set()

    placeholders = ", ".join("?" for _ in video_ids)
    cursor = await db.execute(
        f"SELECT video_id FROM video_metadata WHERE video_id IN ({placeholders})",
        video_ids,
    )
    rows = await cursor.fetchall()
    return {row["video_id"] for row in rows}


async def persist_metadata_items(*, db, items: list[VideoMetadataItem]) -> None:
    if not items:
        return

    for item in items:
        await db.execute(
            """
            INSERT INTO video_metadata (
                video_id,
                title,
                channel_title,
                duration_seconds,
                thumbnail_url,
                is_available,
                fetched_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                title = excluded.title,
                channel_title = excluded.channel_title,
                duration_seconds = excluded.duration_seconds,
                thumbnail_url = excluded.thumbnail_url,
                is_available = excluded.is_available,
                fetched_at = excluded.fetched_at
            """,
            (
                item.video_id,
                item.title,
                item.channel_title,
                item.duration_seconds,
                item.thumbnail_url,
                int(item.is_available),
                item.fetched_at.isoformat(),
            ),
        )


async def persist_unavailable_metadata(*, db, video_ids: list[str]) -> None:
    if not video_ids:
        return

    fetched_at = datetime.now(timezone.utc).isoformat()
    for video_id in video_ids:
        await db.execute(
            """
            INSERT INTO video_metadata (
                video_id,
                title,
                channel_title,
                duration_seconds,
                thumbnail_url,
                is_available,
                fetched_at
            )
            VALUES (?, NULL, NULL, NULL, NULL, 0, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                is_available = excluded.is_available,
                fetched_at = excluded.fetched_at
            """,
            (video_id, fetched_at),
        )


def parse_iso8601_duration(*, duration_text: str | None) -> int | None:
    if not duration_text:
        return None

    match = _DURATION_PATTERN.fullmatch(duration_text)
    if not match:
        return None

    days = int(match.group("days") or 0)
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)

    return (days * 86400) + (hours * 3600) + (minutes * 60) + seconds


def extract_thumbnail_url(*, thumbnails: dict) -> str | None:
    for key in ["high", "medium", "default"]:
        value = thumbnails.get(key)
        if isinstance(value, dict):
            url_value = value.get("url")
            if isinstance(url_value, str):
                return url_value

    return None


def extract_text(raw_value: object) -> str | None:
    if not isinstance(raw_value, str):
        return None

    normalized_value = raw_value.strip()
    if not normalized_value:
        return None

    return normalized_value


def parse_datetime_or_now(*, raw_value: str | None) -> datetime:
    if not raw_value:
        return datetime.now(timezone.utc)

    normalized_value = raw_value.strip()
    if normalized_value.endswith("Z"):
        normalized_value = normalized_value.replace("Z", "+00:00")

    try:
        parsed_value = datetime.fromisoformat(normalized_value)
    except ValueError:
        return datetime.now(timezone.utc)

    if parsed_value.tzinfo is None:
        return parsed_value.replace(tzinfo=timezone.utc)

    return parsed_value.astimezone(timezone.utc)
