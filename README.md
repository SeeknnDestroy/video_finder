<div align="center">
  <h1>YouTube Video Finder</h1>
  <p>Local-first FastAPI app for exploring exported YouTube watch history with metadata enrichment and transcript-aware phrase search.</p>
  <p>
    <a href="https://github.com/SeeknnDestroy/video_finder/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/SeeknnDestroy/video_finder/ci.yml?branch=main&label=CI" alt="CI status"></a>
    <img src="https://img.shields.io/badge/python-3.11%2B-3776AB" alt="Python 3.11+">
    <img src="https://img.shields.io/badge/FastAPI-0.115-009688" alt="FastAPI">
    <img src="https://img.shields.io/badge/SQLite-local%20cache-003B57" alt="SQLite cache">
  </p>
</div>

## Overview

YouTube Video Finder imports a Google Takeout `watch-history.json` export into a local SQLite database, then lets you search it through a small web UI.

The app is designed for personal, local analysis:

- import and deduplicate watched events by `(video_id, watched_at)`
- search by title keywords, date range, duration, and optional result limits
- enrich missing metadata from the YouTube Data API and cache it locally
- search transcript phrases and automatically queue missing transcript work
- use YouTube captions first, then fall back to Groq speech-to-text with configurable Groq Whisper models

## Architecture

```mermaid
flowchart LR
    A["Google Takeout watch-history.json"] --> B["History import service"]
    B --> C["SQLite database"]
    C --> D["Search service"]
    C --> E["Transcript search service"]
    D --> F["Metadata cache"]
    F --> G["YouTube Data API"]
    E --> H["Job queue"]
    H --> I["Background transcription worker"]
    I --> J["yt-dlp + Groq STT"]
    D --> K["FastAPI + Jinja UI"]
    E --> K
```

## Stack

| Layer | Choice | Notes |
| --- | --- | --- |
| App | FastAPI | Single-process local web app |
| Views | Jinja2 + CSS | Server-rendered UI, no frontend build step |
| Storage | SQLite | Watch history, metadata cache, transcript jobs |
| Metadata | YouTube Data API | Optional, only needed for enrichment |
| Transcription | `yt-dlp` + Groq STT | Captions first, then `whisper-large-v3-turbo` cloud fallback |
| Tests | `pytest` | Service and route coverage |

## Quick Start

### Prerequisites

- Python 3.11 or newer
- Optional: Groq API key if you want cloud transcription fallback when captions are unavailable
- Optional: YouTube Data API key for metadata enrichment

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Run the app

```bash
uvicorn app.main:app --reload
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000), upload your Google Takeout `watch-history.json`, and start searching.

### Run the tests

```bash
pytest
```

You can also use the included `Makefile`:

```bash
make install
make run
make test
```

## Configuration

The app loads environment variables from `.env` automatically if the file exists in the current working directory.

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_DB_PATH` | `./data/video_finder.db` | SQLite database path |
| `YOUTUBE_API_KEY` | unset | Enables metadata enrichment for uncached videos |
| `GROQ_API_KEY` | unset | Enables Groq speech-to-text fallback when captions are unavailable |
| `GROQ_TRANSCRIPTION_MODEL` | `whisper-large-v3-turbo` | Active Groq speech-to-text model (`whisper-large-v3` or `whisper-large-v3-turbo`) |
| `YT_DLP_COOKIES_FROM_BROWSER` | unset | Optional advanced override to pin authenticated `yt-dlp` cookie import to one browser, for example `chrome` or `firefox` |
| `YT_DLP_COOKIES_FILE` | unset | Optional advanced override for a Netscape-format cookies file path |
| `LOG_LEVEL` | `INFO` | App logging level |
| `TRANSCRIBE_LANGUAGE` | auto-detect | Optional fixed transcript language |
| `TRANSCRIBE_WORKER_CONCURRENCY` | `1` | Concurrent transcription jobs |
| `TRANSCRIBE_JOB_MAX_CANDIDATES` | `200` | Max candidate videos per queued job |
| `TRANSCRIBE_WORKER_POLL_SECONDS` | `2` | Queue polling interval |
| `TRANSCRIBE_WORKER_ENABLED` | `true` | Starts the in-process worker on app startup |

`GROQ_TRANSCRIPTION_MODEL` also accepts the legacy `TRANSCRIBE_MODEL_SIZE` alias values `turbo` and `large` for backward compatibility.

For private, members-only, or age-restricted videos, the default happy path is simple: log into YouTube in a supported local browser and retry. The app first tries public access, then automatically retries common browser-cookie sources on sign-in-required failures. Use `YT_DLP_COOKIES_FROM_BROWSER` or `YT_DLP_COOKIES_FILE` only if you need to pin a specific browser/profile or use an exported cookie file.

## Search Capabilities

### Metadata-backed search

- title keyword matching with all-term semantics
- duration minimum and maximum filters
- date presets: `7d`, `30d`, `6m`, `1y`
- custom date ranges
- optional result limit up to 200

### Transcript phrase search

- uses transcript matches when they already exist locally
- auto-queues missing transcript work for candidate videos
- exposes queue/progress endpoints for background jobs
- may take longer on the first run while transcripts are prepared

## Project Layout

```text
app/
  core/        configuration, template helpers, static path helpers
  db/          SQLite connection and schema initialization
  models/      Pydantic request and response models
  routers/     FastAPI routes for search, import, and job status
  services/    import, search, metadata, transcription, job orchestration
  templates/   Jinja templates
  static/      CSS assets
  workers/     background transcription worker
tests/         route and service coverage
data/          local database path placeholder
```

## Operational Notes

- This repository intentionally ignores personal/local artifacts such as `.env`, SQLite database files, and raw `watch-history.json` exports.
- Without `YOUTUBE_API_KEY`, the app still works, but metadata-dependent filters may skip uncached videos.
- When captions are unavailable, transcription falls back to the configured Groq model and requires `GROQ_API_KEY`.
- Private or account-restricted YouTube videos first try public access, then automatically retry common logged-in browser cookies. The override env vars are only needed when you want to pin a specific browser/profile or cookie file.
- Permanently unavailable YouTube videos, such as removed or terminated-account videos, are now marked as skipped instead of failed.
- Groq fallback is guarded by a SQLite-backed local preflight limiter so concurrent workers do not oversubscribe Groq request or audio-duration quotas.
- Inspect current Groq limiter usage with the JSON endpoint `/jobs/groq/rate-limit`.
- Built-in Groq limits are enforced per model:
  - `whisper-large-v3`: `300 requests/minute`, `200000 requests/day`, `200000 audio-seconds/hour`, `4000000 audio-seconds/day`
  - `whisper-large-v3-turbo`: `400 requests/minute`, `200000 requests/day`, `400000 audio-seconds/hour`, `4000000 audio-seconds/day`
- The limiter refuses Groq uploads when the app cannot determine the audio duration safely, rather than guessing and risking quota overrun.
- Groq direct file uploads currently have a 25 MB limit; oversized audio fails clearly and is not chunked automatically.
- The background worker runs inside the FastAPI process by default; disable it with `TRANSCRIBE_WORKER_ENABLED=false` if needed.

## CI

GitHub Actions runs the test suite on pushes to `main` and on pull requests against Python 3.11 and 3.12.
