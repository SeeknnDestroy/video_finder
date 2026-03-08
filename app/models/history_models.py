from __future__ import annotations

from datetime import datetime

import aiosqlite
from pydantic import BaseModel, ConfigDict, Field


class ImportHistoryRequest(BaseModel):
    file_name: str


class WatchedEventInput(BaseModel):
    video_id: str
    watched_at: datetime
    source_title: str | None = None
    source_channel: str | None = None
    source_url: str | None = None


class ParseHistoryRequest(BaseModel):
    raw_bytes: bytes


class ParseHistoryResult(BaseModel):
    items: list[WatchedEventInput] = Field(default_factory=list)
    skipped_count: int = 0
    error_message: str | None = None


class UpsertHistoryRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    db: aiosqlite.Connection
    items: list[WatchedEventInput]


class UpsertHistoryResult(BaseModel):
    inserted_count: int = 0
    deduped_count: int = 0
    skipped_count: int = 0
    error_message: str | None = None


class ImportHistoryResult(BaseModel):
    is_successful: bool
    inserted_count: int = 0
    deduped_count: int = 0
    skipped_count: int = 0
    total_parsed_count: int = 0
    error_message: str | None = None
