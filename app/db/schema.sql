PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS watched_events (
    id INTEGER PRIMARY KEY,
    video_id TEXT NOT NULL,
    watched_at TEXT NOT NULL,
    source_title TEXT,
    source_channel TEXT,
    source_url TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(video_id, watched_at)
);

CREATE INDEX IF NOT EXISTS idx_watched_events_watched_at ON watched_events(watched_at DESC);
CREATE INDEX IF NOT EXISTS idx_watched_events_video_id ON watched_events(video_id);

CREATE TABLE IF NOT EXISTS video_metadata (
    video_id TEXT PRIMARY KEY,
    title TEXT,
    channel_title TEXT,
    duration_seconds INTEGER,
    thumbnail_url TEXT,
    is_available INTEGER NOT NULL DEFAULT 1,
    fetched_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_video_metadata_fetched_at ON video_metadata(fetched_at DESC);

-- v2 planned tables
CREATE TABLE IF NOT EXISTS transcripts (
    video_id TEXT PRIMARY KEY,
    language TEXT,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS transcripts_fts USING fts5(
    video_id UNINDEXED,
    text
);

CREATE TRIGGER IF NOT EXISTS trg_transcripts_ai
AFTER INSERT ON transcripts
BEGIN
    INSERT INTO transcripts_fts (video_id, text)
    VALUES (new.video_id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS trg_transcripts_au
AFTER UPDATE ON transcripts
BEGIN
    DELETE FROM transcripts_fts
    WHERE video_id = old.video_id;

    INSERT INTO transcripts_fts (video_id, text)
    VALUES (new.video_id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS trg_transcripts_ad
AFTER DELETE ON transcripts
BEGIN
    DELETE FROM transcripts_fts
    WHERE video_id = old.video_id;
END;

CREATE INDEX IF NOT EXISTS idx_transcripts_language ON transcripts(language);
CREATE INDEX IF NOT EXISTS idx_transcripts_updated_at ON transcripts(updated_at DESC);

CREATE TABLE IF NOT EXISTS transcription_jobs (
    job_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_transcription_jobs_status ON transcription_jobs(status);
CREATE INDEX IF NOT EXISTS idx_transcription_jobs_created_at ON transcription_jobs(created_at DESC);

CREATE TABLE IF NOT EXISTS transcription_job_items (
    job_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT,
    PRIMARY KEY(job_id, video_id)
);

CREATE INDEX IF NOT EXISTS idx_transcription_job_items_status ON transcription_job_items(status);
CREATE INDEX IF NOT EXISTS idx_transcription_job_items_job_id ON transcription_job_items(job_id);
CREATE INDEX IF NOT EXISTS idx_transcription_job_items_video_id ON transcription_job_items(video_id);
