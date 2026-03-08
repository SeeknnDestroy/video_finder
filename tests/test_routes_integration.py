from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "watch_history.json"


def test_import_history_endpoint(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "integration_import.db"
    monkeypatch.setenv("APP_DB_PATH", str(db_path))
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    monkeypatch.setenv("TRANSCRIBE_WORKER_ENABLED", "false")

    with TestClient(app) as client:
        with FIXTURE_PATH.open("rb") as file_handler:
            response = client.post(
                "/import/history",
                files={"history_file": ("watch-history.json", file_handler, "application/json")},
            )

    assert response.status_code == 200
    assert "Import Complete" in response.text

    connection = sqlite3.connect(db_path)
    cursor = connection.execute("SELECT COUNT(*) FROM watched_events")
    row_count = cursor.fetchone()[0]
    connection.close()

    assert row_count == 2


def test_search_endpoint_has_no_default_date_filter(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "integration_search.db"
    monkeypatch.setenv("APP_DB_PATH", str(db_path))
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    monkeypatch.setenv("TRANSCRIBE_WORKER_ENABLED", "false")

    with TestClient(app) as client:
        now = datetime.now(timezone.utc)
        recent = now - timedelta(days=10)
        old = now - timedelta(days=400)

        connection = sqlite3.connect(db_path)
        connection.execute(
            """
            INSERT INTO watched_events (video_id, watched_at, source_title, source_channel, source_url, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "recent-video",
                recent.isoformat(),
                "Recent Match",
                "Channel A",
                "https://www.youtube.com/watch?v=recent-video",
                now.isoformat(),
            ),
        )
        connection.execute(
            """
            INSERT INTO watched_events (video_id, watched_at, source_title, source_channel, source_url, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "old-video",
                old.isoformat(),
                "Old Match",
                "Channel B",
                "https://www.youtube.com/watch?v=old-video",
                now.isoformat(),
            ),
        )
        connection.commit()
        connection.close()

        response = client.get("/search", params={"title_query": "match"})

    assert response.status_code == 200
    assert "Recent Match" in response.text
    assert "Old Match" in response.text


def test_search_endpoint_handles_mixed_cached_and_uncached_metadata(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "integration_metadata.db"
    monkeypatch.setenv("APP_DB_PATH", str(db_path))
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    monkeypatch.setenv("TRANSCRIBE_WORKER_ENABLED", "false")

    with TestClient(app) as client:
        now = datetime.now(timezone.utc).isoformat()
        connection = sqlite3.connect(db_path)
        connection.execute(
            """
            INSERT INTO watched_events (video_id, watched_at, source_title, source_channel, source_url, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "cached-video",
                now,
                "Quick Cached Video",
                "Channel C",
                "https://www.youtube.com/watch?v=cached-video",
                now,
            ),
        )
        connection.execute(
            """
            INSERT INTO watched_events (video_id, watched_at, source_title, source_channel, source_url, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "uncached-video",
                now,
                "Quick Uncached Video",
                "Channel D",
                "https://www.youtube.com/watch?v=uncached-video",
                now,
            ),
        )
        connection.execute(
            """
            INSERT INTO video_metadata (video_id, title, channel_title, duration_seconds, thumbnail_url, is_available, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "cached-video",
                "Quick Cached Video",
                "Channel C",
                45,
                "https://example.com/thumb.jpg",
                1,
                now,
            ),
        )
        connection.commit()
        connection.close()

        response = client.get("/search", params={"title_query": "quick", "duration_max_seconds": 60})

    assert response.status_code == 200
    assert "Quick Cached Video" in response.text
    assert "YOUTUBE_API_KEY is missing" in response.text


def test_search_endpoint_accepts_blank_optional_query_values(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "integration_blank_optional_query.db"
    monkeypatch.setenv("APP_DB_PATH", str(db_path))
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    monkeypatch.setenv("TRANSCRIBE_WORKER_ENABLED", "false")

    with TestClient(app) as client:
        now = datetime.now(timezone.utc).isoformat()
        connection = sqlite3.connect(db_path)
        connection.execute(
            """
            INSERT INTO watched_events (video_id, watched_at, source_title, source_channel, source_url, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "blank-query-video",
                now,
                "Blank Query Test Video",
                "Channel E",
                "https://www.youtube.com/watch?v=blank-query-video",
                now,
            ),
        )
        connection.execute(
            """
            INSERT INTO video_metadata (video_id, title, channel_title, duration_seconds, thumbnail_url, is_available, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "blank-query-video",
                "Blank Query Test Video",
                "Channel E",
                45,
                "https://example.com/blank.jpg",
                1,
                now,
            ),
        )
        connection.commit()
        connection.close()

        response = client.get(
            "/search",
            params={
                "title_query": "",
                "duration_min_seconds": "",
                "duration_max_seconds": "60",
                "date_preset": "6m",
                "watched_from": "",
                "watched_to": "",
                "limit": "",
            },
        )

    assert response.status_code == 200
    assert "Blank Query Test Video" in response.text
