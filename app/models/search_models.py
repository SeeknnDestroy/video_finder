from __future__ import annotations

from datetime import date, datetime
from typing import Literal

import aiosqlite
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SearchVideosRequest(BaseModel):
    phrase: str | None = None
    title_query: str | None = None
    channel_query: str | None = None
    duration_min_seconds: int | None = Field(default=None, ge=0, le=43200)
    duration_max_seconds: int | None = Field(default=None, ge=1, le=43200)
    watched_from: date | None = None
    watched_to: date | None = None
    date_preset: Literal["7d", "30d", "6m", "1y", "custom"] | None = None
    limit: int | None = Field(default=None, ge=1, le=200)

    @field_validator(
        "duration_min_seconds",
        "duration_max_seconds",
        "watched_from",
        "watched_to",
        "date_preset",
        "limit",
        "phrase",
        "channel_query",
        mode="before",
    )
    @classmethod
    def normalize_empty_optional_fields(cls, value: object) -> object:
        if not isinstance(value, str):
            return value

        stripped_value = value.strip()
        if not stripped_value:
            return None

        return stripped_value

    @model_validator(mode="after")
    def validate_duration_bounds(self) -> "SearchVideosRequest":
        if self.duration_min_seconds is None or self.duration_max_seconds is None:
            return self

        if self.duration_min_seconds > self.duration_max_seconds:
            raise ValueError("duration_min_seconds must be less than or equal to duration_max_seconds.")

        return self

    @field_validator("title_query", "channel_query")
    @classmethod
    def normalize_keyword_query(cls, value: str | None) -> str | None:
        if value is None:
            return None

        normalized_value = value.strip()
        if not normalized_value:
            return None

        return normalized_value

    @field_validator("phrase")
    @classmethod
    def normalize_phrase(cls, value: str | None) -> str | None:
        if value is None:
            return None

        normalized_value = value.strip()
        if not normalized_value:
            return None

        return normalized_value


class SearchVideoItem(BaseModel):
    video_id: str
    video_url: str
    title: str
    channel_title: str | None = None
    watched_at: datetime
    duration_seconds: int | None = None
    thumbnail_url: str | None = None


class SearchVideosResult(BaseModel):
    items: list[SearchVideoItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    queued_count: int = 0
    candidate_count: int = 0


class ResolveDateRangeResult(BaseModel):
    watched_from: datetime | None = None
    watched_to_exclusive: datetime | None = None
    applied_preset: str | None = None
    error_message: str | None = None


class SearchVideosServiceRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    db: aiosqlite.Connection
    search: SearchVideosRequest
    api_key: str | None = None
