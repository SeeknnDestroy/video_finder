from __future__ import annotations

from datetime import date
from typing import Literal

import aiosqlite
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

JobStatus = Literal["queued", "running", "completed", "completed_with_errors", "failed"]
JobItemStatus = Literal["queued", "processing", "completed", "failed", "skipped"]

DATE_PRESET_VALUES = {"7d", "30d", "6m", "1y", "custom"}


class CreateTranscriptionJobRequest(BaseModel):
    phrase: str | None = None
    title_query: str | None = None
    channel_query: str | None = None
    duration_min_seconds: int | None = Field(default=None, ge=0, le=43200)
    duration_max_seconds: int | None = Field(default=None, ge=1, le=43200)
    watched_from: date | None = None
    watched_to: date | None = None
    date_preset: str | None = None
    force_retranscribe: bool = False
    language: str | None = None

    @field_validator(
        "phrase",
        "title_query",
        "channel_query",
        "duration_min_seconds",
        "duration_max_seconds",
        "watched_from",
        "watched_to",
        "date_preset",
        "language",
        mode="before",
    )
    @classmethod
    def normalize_empty_optional_fields(cls, value: object) -> object:
        if not isinstance(value, str):
            return value

        normalized_value = value.strip()
        if not normalized_value:
            return None

        return normalized_value

    @model_validator(mode="after")
    def validate_duration_bounds(self) -> "CreateTranscriptionJobRequest":
        if self.duration_min_seconds is None or self.duration_max_seconds is None:
            return self

        if self.duration_min_seconds > self.duration_max_seconds:
            raise ValueError("duration_min_seconds must be less than or equal to duration_max_seconds.")

        return self

    @field_validator("date_preset")
    @classmethod
    def validate_date_preset(cls, value: str | None) -> str | None:
        if value is None:
            return None

        if value not in DATE_PRESET_VALUES:
            raise ValueError("date_preset must be one of 7d, 30d, 6m, 1y, custom.")

        return value


class CreateTranscriptionJobResult(BaseModel):
    job_id: str | None = None
    queued_count: int = 0
    error_message: str | None = None


class SpokenSearchRequest(BaseModel):
    phrase: str = Field(min_length=1)
    title_query: str | None = None
    channel_query: str | None = None
    duration_min_seconds: int | None = Field(default=None, ge=0, le=43200)
    duration_max_seconds: int | None = Field(default=None, ge=1, le=43200)
    watched_from: date | None = None
    watched_to: date | None = None
    date_preset: str | None = None
    limit: int = Field(default=50, ge=1, le=200)
    language: str | None = None

    @field_validator(
        "phrase",
        "title_query",
        "channel_query",
        "duration_min_seconds",
        "duration_max_seconds",
        "watched_from",
        "watched_to",
        "date_preset",
        "language",
        mode="before",
    )
    @classmethod
    def normalize_empty_optional_fields(cls, value: object) -> object:
        if not isinstance(value, str):
            return value

        normalized_value = value.strip()
        if not normalized_value:
            return None

        return normalized_value

    @field_validator("phrase")
    @classmethod
    def normalize_phrase(cls, value: str) -> str:
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("phrase is required.")

        return normalized_value

    @model_validator(mode="after")
    def validate_duration_bounds(self) -> "SpokenSearchRequest":
        if self.duration_min_seconds is None or self.duration_max_seconds is None:
            return self

        if self.duration_min_seconds > self.duration_max_seconds:
            raise ValueError("duration_min_seconds must be less than or equal to duration_max_seconds.")

        return self

    @field_validator("date_preset")
    @classmethod
    def validate_date_preset(cls, value: str | None) -> str | None:
        if value is None:
            return None

        if value not in DATE_PRESET_VALUES:
            raise ValueError("date_preset must be one of 7d, 30d, 6m, 1y, custom.")

        return value


class SpokenSearchItem(BaseModel):
    video_id: str
    title: str
    video_url: str
    thumbnail_url: str | None = None
    channel_title: str | None = None
    duration_seconds: int | None = None
    snippet: str | None = None
    watched_at: str | None = None


class SpokenSearchResult(BaseModel):
    items: list[SpokenSearchItem] = Field(default_factory=list)
    queued_count: int = 0
    candidate_count: int = 0
    transcript_available_count: int = 0
    needs_transcription_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    job_id: str | None = None
    error_message: str | None = None


class JobItemStatusSummary(BaseModel):
    video_id: str
    status: JobItemStatus
    error_message: str | None = None


class JobStatusResult(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error_message: str | None = None
    total_count: int = 0
    queued_count: int = 0
    processing_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    items: list[JobItemStatusSummary] = Field(default_factory=list)


class CreateTranscriptionJobServiceRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    db: aiosqlite.Connection
    payload: CreateTranscriptionJobRequest
    max_candidates: int = Field(default=200, ge=1)
    api_key: str | None = None


class SpokenSearchServiceRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    db: aiosqlite.Connection
    payload: SpokenSearchRequest
    max_candidates: int = Field(default=200, ge=1)
    api_key: str | None = None


class GetJobStatusServiceRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    db: aiosqlite.Connection
    job_id: str
    include_items: bool = True
