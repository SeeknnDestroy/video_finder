from __future__ import annotations

from pathlib import Path

from app.core import config as config_module


def test_get_app_config_loads_values_from_dotenv(tmp_path, monkeypatch) -> None:
    dotenv_path = Path(tmp_path) / ".env"
    dotenv_path.write_text(
        "YOUTUBE_API_KEY=from_dotenv\n"
        "GROQ_API_KEY=groq-from-dotenv\n"
        "APP_DB_PATH=./from_dotenv.db\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("APP_DB_PATH", raising=False)
    config_module.load_dotenv_files.cache_clear()

    app_config = config_module.get_app_config()

    assert app_config.youtube_api_key == "from_dotenv"
    assert app_config.groq_api_key == "groq-from-dotenv"
    assert app_config.app_db_path == "./from_dotenv.db"


def test_environment_variables_override_dotenv_values(tmp_path, monkeypatch) -> None:
    dotenv_path = Path(tmp_path) / ".env"
    dotenv_path.write_text("YOUTUBE_API_KEY=from_dotenv\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("YOUTUBE_API_KEY", "from_env")
    config_module.load_dotenv_files.cache_clear()

    app_config = config_module.get_app_config()

    assert app_config.youtube_api_key == "from_env"


def test_transcription_settings_have_groq_defaults(tmp_path, monkeypatch) -> None:
    dotenv_path = Path(tmp_path) / ".env"
    dotenv_path.write_text("YOUTUBE_API_KEY=from_dotenv\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_TRANSCRIPTION_MODEL", raising=False)
    monkeypatch.delenv("TRANSCRIBE_MODEL_SIZE", raising=False)
    monkeypatch.delenv("TRANSCRIBE_WORKER_CONCURRENCY", raising=False)
    monkeypatch.delenv("TRANSCRIBE_JOB_MAX_CANDIDATES", raising=False)
    monkeypatch.delenv("TRANSCRIBE_WORKER_POLL_SECONDS", raising=False)
    monkeypatch.delenv("TRANSCRIBE_WORKER_ENABLED", raising=False)
    config_module.load_dotenv_files.cache_clear()

    app_config = config_module.get_app_config()

    assert app_config.groq_api_key is None
    assert app_config.groq_transcription_model == "whisper-large-v3-turbo"
    assert app_config.transcribe_language is None
    assert app_config.transcribe_worker_concurrency == 1
    assert app_config.transcribe_job_max_candidates == 200
    assert app_config.transcribe_worker_poll_seconds == 2.0
    assert app_config.transcribe_worker_enabled is True


def test_groq_transcription_model_accepts_explicit_model_name(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GROQ_TRANSCRIPTION_MODEL", "whisper-large-v3")
    monkeypatch.delenv("TRANSCRIBE_MODEL_SIZE", raising=False)
    config_module.load_dotenv_files.cache_clear()

    app_config = config_module.get_app_config()

    assert app_config.groq_transcription_model == "whisper-large-v3"
    assert app_config.groq_transcription_rate_limits().requests_per_minute == 300


def test_groq_transcription_model_supports_legacy_size_alias(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GROQ_TRANSCRIPTION_MODEL", raising=False)
    monkeypatch.setenv("TRANSCRIBE_MODEL_SIZE", "turbo")
    config_module.load_dotenv_files.cache_clear()

    app_config = config_module.get_app_config()

    assert app_config.groq_transcription_model == "whisper-large-v3-turbo"
    assert app_config.groq_transcription_rate_limits().audio_seconds_per_hour == 400_000
