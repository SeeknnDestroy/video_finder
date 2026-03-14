from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

GROQ_MODEL_WHISPER_LARGE_V3 = "whisper-large-v3"
GROQ_MODEL_WHISPER_LARGE_V3_TURBO = "whisper-large-v3-turbo"


class GroqTranscriptionRateLimits(BaseModel):
    requests_per_minute: int = Field(ge=1)
    requests_per_day: int = Field(ge=1)
    audio_seconds_per_hour: int = Field(ge=1)
    audio_seconds_per_day: int = Field(ge=1)


GROQ_TRANSCRIPTION_RATE_LIMITS_BY_MODEL = {
    GROQ_MODEL_WHISPER_LARGE_V3: GroqTranscriptionRateLimits(
        requests_per_minute=300,
        requests_per_day=200_000,
        audio_seconds_per_hour=200_000,
        audio_seconds_per_day=4_000_000,
    ),
    GROQ_MODEL_WHISPER_LARGE_V3_TURBO: GroqTranscriptionRateLimits(
        requests_per_minute=400,
        requests_per_day=200_000,
        audio_seconds_per_hour=400_000,
        audio_seconds_per_day=4_000_000,
    ),
}


class AppConfig(BaseModel):
    app_db_path: str = Field(default="./data/video_finder.db")
    youtube_api_key: str | None = None
    groq_api_key: str | None = None
    groq_transcription_model: str = Field(default=GROQ_MODEL_WHISPER_LARGE_V3_TURBO)
    yt_dlp_cookies_from_browser: str | None = None
    yt_dlp_cookies_profile: str | None = None
    yt_dlp_cookies_file: str | None = None
    log_level: str = Field(default="INFO")
    transcribe_language: str | None = None
    transcribe_worker_concurrency: int = Field(default=1, ge=1)
    transcribe_job_max_candidates: int = Field(default=200, ge=1)
    transcribe_worker_poll_seconds: float = Field(default=2.0, gt=0)
    transcribe_worker_enabled: bool = True

    def groq_transcription_rate_limits(self) -> GroqTranscriptionRateLimits:
        return get_groq_transcription_rate_limits(model=self.groq_transcription_model)


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

    return AppConfig(
        app_db_path=os.getenv("APP_DB_PATH", "./data/video_finder.db"),
        youtube_api_key=os.getenv("YOUTUBE_API_KEY"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        groq_transcription_model=parse_groq_transcription_model_env(
            raw_value=os.getenv("GROQ_TRANSCRIPTION_MODEL"),
            legacy_size_value=os.getenv("TRANSCRIBE_MODEL_SIZE"),
            default_value=GROQ_MODEL_WHISPER_LARGE_V3_TURBO,
        ),
        yt_dlp_cookies_from_browser=normalize_optional_string_env(
            raw_value=os.getenv("YT_DLP_COOKIES_FROM_BROWSER")
        ),
        yt_dlp_cookies_profile=normalize_optional_string_env(
            raw_value=os.getenv("YT_DLP_COOKIES_PROFILE")
        ),
        yt_dlp_cookies_file=normalize_optional_string_env(
            raw_value=os.getenv("YT_DLP_COOKIES_FILE")
        ),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        transcribe_language=raw_transcribe_language or None,
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


def get_groq_transcription_rate_limits(*, model: str) -> GroqTranscriptionRateLimits:
    return GROQ_TRANSCRIPTION_RATE_LIMITS_BY_MODEL.get(
        model,
        GROQ_TRANSCRIPTION_RATE_LIMITS_BY_MODEL[GROQ_MODEL_WHISPER_LARGE_V3_TURBO],
    )


def parse_groq_transcription_model_env(
    *,
    raw_value: str | None,
    legacy_size_value: str | None,
    default_value: str,
) -> str:
    alias_map = {
        "large": GROQ_MODEL_WHISPER_LARGE_V3,
        "turbo": GROQ_MODEL_WHISPER_LARGE_V3_TURBO,
        GROQ_MODEL_WHISPER_LARGE_V3: GROQ_MODEL_WHISPER_LARGE_V3,
        GROQ_MODEL_WHISPER_LARGE_V3_TURBO: GROQ_MODEL_WHISPER_LARGE_V3_TURBO,
    }

    normalized_value = normalize_model_env_value(raw_value)
    if normalized_value in alias_map:
        return alias_map[normalized_value]

    normalized_legacy_value = normalize_model_env_value(legacy_size_value)
    if normalized_legacy_value in alias_map:
        return alias_map[normalized_legacy_value]

    return default_value


def normalize_model_env_value(raw_value: str | None) -> str | None:
    if raw_value is None:
        return None

    normalized_value = raw_value.strip().lower()
    if not normalized_value:
        return None

    return normalized_value


def normalize_optional_string_env(*, raw_value: str | None) -> str | None:
    if raw_value is None:
        return None

    normalized_value = raw_value.strip()
    if not normalized_value:
        return None

    return normalized_value


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
