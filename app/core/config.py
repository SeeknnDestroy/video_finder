from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    app_db_path: str = Field(default="./data/video_finder.db")
    youtube_api_key: str | None = None
    log_level: str = Field(default="INFO")
    transcribe_model_size: str = Field(default="turbo")
    transcribe_language: str | None = None
    transcribe_compute_type: str = Field(default="int8")
    transcribe_worker_concurrency: int = Field(default=1, ge=1)
    transcribe_job_max_candidates: int = Field(default=200, ge=1)
    transcribe_worker_poll_seconds: float = Field(default=2.0, gt=0)
    transcribe_worker_enabled: bool = True


@lru_cache(maxsize=1)
def load_dotenv_files() -> None:
    cwd_dotenv_path = Path.cwd() / ".env"
    if cwd_dotenv_path.exists():
        load_dotenv(dotenv_path=cwd_dotenv_path, override=False)
        return

    load_dotenv(override=False)


def get_app_config() -> AppConfig:
    load_dotenv_files()
    raw_transcribe_language = os.getenv("TRANSCRIBE_LANGUAGE", "").strip()
    raw_transcribe_model_size = os.getenv("TRANSCRIBE_MODEL_SIZE", "turbo").strip()
    raw_transcribe_compute_type = os.getenv("TRANSCRIBE_COMPUTE_TYPE", "int8").strip()

    return AppConfig(
        app_db_path=os.getenv("APP_DB_PATH", "./data/video_finder.db"),
        youtube_api_key=os.getenv("YOUTUBE_API_KEY"),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        transcribe_model_size=raw_transcribe_model_size or "turbo",
        transcribe_language=raw_transcribe_language or None,
        transcribe_compute_type=raw_transcribe_compute_type or "int8",
        transcribe_worker_concurrency=parse_positive_integer_env(
            raw_value=os.getenv("TRANSCRIBE_WORKER_CONCURRENCY"),
            default_value=1,
        ),
        transcribe_job_max_candidates=parse_positive_integer_env(
            raw_value=os.getenv("TRANSCRIBE_JOB_MAX_CANDIDATES"),
            default_value=200,
        ),
        transcribe_worker_poll_seconds=parse_positive_float_env(
            raw_value=os.getenv("TRANSCRIBE_WORKER_POLL_SECONDS"),
            default_value=2.0,
        ),
        transcribe_worker_enabled=parse_boolean_env(
            raw_value=os.getenv("TRANSCRIBE_WORKER_ENABLED"),
            default_value=True,
        ),
    )


def parse_positive_integer_env(*, raw_value: str | None, default_value: int) -> int:
    if raw_value is None:
        return default_value

    normalized_value = raw_value.strip()
    if not normalized_value:
        return default_value

    try:
        parsed_value = int(normalized_value)
    except ValueError:
        return default_value

    if parsed_value < 1:
        return default_value

    return parsed_value


def parse_positive_float_env(*, raw_value: str | None, default_value: float) -> float:
    if raw_value is None:
        return default_value

    normalized_value = raw_value.strip()
    if not normalized_value:
        return default_value

    try:
        parsed_value = float(normalized_value)
    except ValueError:
        return default_value

    if parsed_value <= 0:
        return default_value

    return parsed_value


def parse_boolean_env(*, raw_value: str | None, default_value: bool) -> bool:
    if raw_value is None:
        return default_value

    normalized_value = raw_value.strip().lower()
    if not normalized_value:
        return default_value

    if normalized_value in {"1", "true", "yes", "on"}:
        return True

    if normalized_value in {"0", "false", "no", "off"}:
        return False

    return default_value
