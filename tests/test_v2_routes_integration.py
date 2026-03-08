from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.main import app


def insert_watched_event(*, connection: sqlite3.Connection, video_id: str, watched_at: datetime, title: str) -> None:
    connection.execute(
        """
        INSERT INTO watched_events (video_id, watched_at, source_title, source_channel, source_url, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            video_id,
            watched_at.isoformat(),
            title,
            "Channel",
            f"https://www.youtube.com/watch?v={video_id}",
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def test_v2_json_job_routes_create_and_fetch_status(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "v2_json_jobs.db"
    monkeypatch.setenv("APP_DB_PATH", str(db_path))
    monkeypatch.setenv("TRANSCRIBE_WORKER_ENABLED", "false")
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)

    with TestClient(app) as client:
        connection = sqlite3.connect(db_path)
        now = datetime.now(timezone.utc)
        insert_watched_event(connection=connection, video_id="video-1", watched_at=now, title="One")
        insert_watched_event(connection=connection, video_id="video-2", watched_at=now - timedelta(minutes=1), title="Two")
        connection.commit()
        connection.close()

        create_response = client.post("/jobs/transcribe", json={})

        assert create_response.status_code == 201
        create_payload = create_response.json()
        assert create_payload["job_id"]
        assert create_payload["queued_count"] == 2

        status_response = client.get(f"/jobs/{create_payload['job_id']}")

    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["status"] == "queued"
    assert status_payload["total_count"] == 2
    assert status_payload["queued_count"] == 2


def test_v2_spoken_json_route_returns_matches_and_auto_queue(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "v2_spoken_json.db"
    monkeypatch.setenv("APP_DB_PATH", str(db_path))
    monkeypatch.setenv("TRANSCRIBE_WORKER_ENABLED", "false")
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)

    with TestClient(app) as client:
        connection = sqlite3.connect(db_path)
        now = datetime.now(timezone.utc)
        insert_watched_event(connection=connection, video_id="spoken-hit", watched_at=now, title="Hit")
        insert_watched_event(
            connection=connection,
            video_id="spoken-missing",
            watched_at=now - timedelta(minutes=1),
            title="Missing",
        )
        connection.execute(
            """
            INSERT INTO transcripts (video_id, language, text, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "spoken-hit",
                "en",
                "hello from the transcript",
                now.isoformat(),
                now.isoformat(),
            ),
        )
        connection.commit()
        connection.close()

        response = client.get("/search/spoken", params={"phrase": "hello", "limit": 20})

        assert response.status_code == 200
        payload = response.json()
        assert payload["error_message"] is None
        assert len(payload["items"]) == 1
        assert payload["items"][0]["video_id"] == "spoken-hit"
        assert payload["queued_count"] == 1
        assert payload["candidate_count"] == 2
        assert payload["transcript_available_count"] == 1
        assert payload["needs_transcription_count"] == 1
        assert payload["job_id"] is not None

        job_response = client.get(f"/jobs/{payload['job_id']}")

    assert job_response.status_code == 200
    job_payload = job_response.json()
    assert job_payload["queued_count"] == 1


def test_search_page_handles_transcript_phrase_in_single_form(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "v2_single_form.db"
    monkeypatch.setenv("APP_DB_PATH", str(db_path))
    monkeypatch.setenv("TRANSCRIBE_WORKER_ENABLED", "false")
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)

    with TestClient(app) as client:
        connection = sqlite3.connect(db_path)
        now = datetime.now(timezone.utc)
        insert_watched_event(connection=connection, video_id="phrase-hit", watched_at=now, title="Phrase Hit")
        connection.execute(
            """
            INSERT INTO transcripts (video_id, language, text, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "phrase-hit",
                "en",
                "we discuss one page search behavior",
                now.isoformat(),
                now.isoformat(),
            ),
        )
        connection.commit()
        connection.close()

        response = client.get("/search", params={"phrase": "one page", "limit": 20})

    assert response.status_code == 200
    assert "Transcript phrase (optional)" in response.text
    assert "Search Videos" in response.text
    assert "Transcript phrase search can take longer on first run" in response.text
    assert "Queue Transcription Job" not in response.text
    assert "Found 1 transcript matches" in response.text
    assert "Filtered Videos" in response.text
    assert "Transcript Ready" in response.text
    assert "Need Transcription" in response.text


def test_progress_page_shows_job_counts(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "v2_progress_page.db"
    monkeypatch.setenv("APP_DB_PATH", str(db_path))
    monkeypatch.setenv("TRANSCRIBE_WORKER_ENABLED", "false")
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)

    with TestClient(app) as client:
        connection = sqlite3.connect(db_path)
        now = datetime.now(timezone.utc)
        insert_watched_event(connection=connection, video_id="progress-video", watched_at=now, title="Progress")
        connection.commit()
        connection.close()

        create_response = client.post("/jobs/transcribe", json={})
        assert create_response.status_code == 201
        job_id = create_response.json()["job_id"]

        progress_response = client.get(f"/progress/{job_id}")

    assert progress_response.status_code == 200
    assert "Transcription Progress" in progress_response.text
    assert "Status:</strong> queued" in progress_response.text
    assert "Open live progress page" not in progress_response.text
