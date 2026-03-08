from __future__ import annotations

from datetime import datetime

import aiosqlite
from pydantic import BaseModel, ConfigDict, Field


class VideoMetadataItem(BaseModel):
    video_id: str
    title: str | None = None
    channel_title: str | None = None
    duration_seconds: int | None = None
    thumbnail_url: str | None = None
    is_available: bool = True
    fetched_at: datetime


class FetchVideoMetadataRequest(BaseModel):
    video_ids: list[str]
    api_key: str


class FetchVideoMetadataResult(BaseModel):
    items: list[VideoMetadataItem] = Field(default_factory=list)
    missing_video_ids: list[str] = Field(default_factory=list)
    warning_message: str | None = None


class EnsureMetadataCacheRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    db: aiosqlite.Connection
    video_ids: list[str]
    api_key: str | None = None


class EnsureMetadataCacheResult(BaseModel):
    fetched_count: int = 0
    cached_count: int = 0
    warning_message: str | None = None


class LoadMetadataMapRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    db: aiosqlite.Connection
    video_ids: list[str]


class LoadMetadataMapResult(BaseModel):
    metadata_by_video_id: dict[str, VideoMetadataItem] = Field(default_factory=dict)
