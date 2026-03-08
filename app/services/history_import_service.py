from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

from app.models.history_models import (
    ParseHistoryRequest,
    ParseHistoryResult,
    UpsertHistoryRequest,
    UpsertHistoryResult,
    WatchedEventInput,
)

logger = logging.getLogger(__name__)


def parse_watch_history(*, request: ParseHistoryRequest) -> ParseHistoryResult:
    if not request.raw_bytes:
        return ParseHistoryResult(error_message="Uploaded history file is empty.")

    try:
        parsed_payload = json.loads(request.raw_bytes.decode("utf-8"))
    except Exception as exc:
        return ParseHistoryResult(error_message=f"Could not parse JSON: {exc}")

    if not isinstance(parsed_payload, list):
        return ParseHistoryResult(error_message="Expected a JSON array in watch-history.json.")

    normalized_items: list[WatchedEventInput] = []
    skipped_count = 0

    for raw_item in parsed_payload:
        if not isinstance(raw_item, dict):
            skipped_count += 1
            continue

        normalized_item = normalize_history_item(raw_item=raw_item)
        if normalized_item is None:
            skipped_count += 1
            continue

        normalized_items.append(normalized_item)

    return ParseHistoryResult(items=normalized_items, skipped_count=skipped_count)


async def upsert_watch_history(*, request: UpsertHistoryRequest) -> UpsertHistoryResult:
    if not request.items:
        return UpsertHistoryResult(inserted_count=0, deduped_count=0, skipped_count=0)

    inserted_count = 0
    deduped_count = 0

    try:
        for item in request.items:
            cursor = await request.db.execute(
                """
                INSERT INTO watched_events (
                    video_id,
                    watched_at,
                    source_title,
                    source_channel,
                    source_url,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_id, watched_at) DO NOTHING
                """,
                (
                    item.video_id,
                    item.watched_at.isoformat(),
                    item.source_title,
                    item.source_channel,
                    item.source_url,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

            if cursor.rowcount == 0:
                deduped_count += 1
                continue

            inserted_count += 1

        await request.db.commit()
    except Exception as exc:
        logger.exception("Failed to upsert history records")
        return UpsertHistoryResult(
            inserted_count=inserted_count,
            deduped_count=deduped_count,
            error_message=f"Failed to persist history records: {exc}",
        )

    return UpsertHistoryResult(
        inserted_count=inserted_count,
        deduped_count=deduped_count,
        skipped_count=0,
    )


def normalize_history_item(*, raw_item: dict) -> WatchedEventInput | None:
    raw_url = raw_item.get("titleUrl")
    raw_time = raw_item.get("time")

    if not isinstance(raw_url, str):
        return None

    if not isinstance(raw_time, str):
        return None

    video_id = extract_video_id(raw_url=raw_url)
    if not video_id:
        return None

    watched_at = parse_datetime(raw_value=raw_time)
    if watched_at is None:
        return None

    raw_title = raw_item.get("title")
    normalized_title = normalize_title(raw_title=raw_title)
    source_channel = extract_channel_name(raw_item=raw_item)

    return WatchedEventInput(
        video_id=video_id,
        watched_at=watched_at,
        source_title=normalized_title,
        source_channel=source_channel,
        source_url=raw_url,
    )


def extract_video_id(*, raw_url: str) -> str | None:
    parsed_url = urlparse(raw_url)
    host = parsed_url.netloc.lower()

    if "youtube.com" not in host and "youtu.be" not in host:
        return None

    if "youtu.be" in host:
        short_id = parsed_url.path.strip("/")
        if short_id:
            return short_id

    if parsed_url.path == "/watch":
        query_values = parse_qs(parsed_url.query)
        query_video_id = query_values.get("v", [None])[0]
        if query_video_id:
            return query_video_id

    if parsed_url.path.startswith("/shorts/"):
        short_path = parsed_url.path.split("/", maxsplit=3)
        if len(short_path) >= 3 and short_path[2]:
            return short_path[2]

    return None


def parse_datetime(*, raw_value: str) -> datetime | None:
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


def normalize_title(*, raw_title: object) -> str | None:
    if not isinstance(raw_title, str):
        return None

    normalized_title = raw_title.strip()
    if not normalized_title:
        return None

    if normalized_title.lower().startswith("watched "):
        normalized_title = normalized_title[8:].strip()

    if not normalized_title:
        return None

    return normalized_title


def extract_channel_name(*, raw_item: dict) -> str | None:
    subtitles = raw_item.get("subtitles")
    if not isinstance(subtitles, list):
        return None

    if not subtitles:
        return None

    first_subtitle = subtitles[0]
    if not isinstance(first_subtitle, dict):
        return None

    raw_name = first_subtitle.get("name")
    if not isinstance(raw_name, str):
        return None

    normalized_name = raw_name.strip()
    if not normalized_name:
        return None

    return normalized_name
