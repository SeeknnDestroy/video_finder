from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.core.config import get_app_config
from app.db.database import DatabaseConnectionRequest, get_database_connection
from app.models.job_models import (
    CreateTranscriptionJobRequest,
    CreateTranscriptionJobServiceRequest,
    GetJobStatusServiceRequest,
    SpokenSearchRequest,
    SpokenSearchServiceRequest,
)
from app.services.transcription_service import (
    create_transcription_job,
    get_transcription_job_status,
    search_spoken_transcripts,
)
from app.services.groq_rate_limit_service import get_groq_rate_limit_snapshot

jobs_router = APIRouter()


@jobs_router.post("/jobs/transcribe")
async def create_job(*, payload: CreateTranscriptionJobRequest) -> JSONResponse:
    config = get_app_config()
    async with get_database_connection(
        request=DatabaseConnectionRequest(db_path=config.app_db_path)
    ) as db:
        result = await create_transcription_job(
            request=CreateTranscriptionJobServiceRequest(
                db=db,
                payload=payload,
                max_candidates=config.transcribe_job_max_candidates,
                api_key=config.youtube_api_key,
            )
        )

    status_code = 201 if result.job_id else 400
    return JSONResponse(status_code=status_code, content=result.model_dump())


@jobs_router.get("/jobs/{job_id}")
async def get_job_status(*, job_id: str) -> JSONResponse:
    config = get_app_config()
    async with get_database_connection(
        request=DatabaseConnectionRequest(db_path=config.app_db_path)
    ) as db:
        result = await get_transcription_job_status(
            request=GetJobStatusServiceRequest(db=db, job_id=job_id)
        )

    if result is None:
        return JSONResponse(
            status_code=404,
            content={"job_id": job_id, "error_message": "Job not found."},
        )

    return JSONResponse(status_code=200, content=result.model_dump())


@jobs_router.get("/search/spoken")
async def search_spoken(
    *,
    phrase: str = Query(...),
    title_query: str | None = Query(default=None),
    channel_query: str | None = Query(default=None),
    duration_min_seconds: int | None = Query(default=None),
    duration_max_seconds: int | None = Query(default=None),
    watched_from: date | None = Query(default=None),
    watched_to: date | None = Query(default=None),
    date_preset: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    language: str | None = Query(default=None),
) -> JSONResponse:
    try:
        request_payload = SpokenSearchRequest(
            phrase=phrase,
            title_query=title_query,
            channel_query=channel_query,
            duration_min_seconds=duration_min_seconds,
            duration_max_seconds=duration_max_seconds,
            watched_from=watched_from,
            watched_to=watched_to,
            date_preset=date_preset,
            limit=limit,
            language=language,
        )
    except ValidationError as exc:
        return JSONResponse(
            status_code=400,
            content={"error_message": f"Invalid spoken-search input: {exc.errors()}"},
        )
    config = get_app_config()
    async with get_database_connection(
        request=DatabaseConnectionRequest(db_path=config.app_db_path)
    ) as db:
        result = await search_spoken_transcripts(
            request=SpokenSearchServiceRequest(
                db=db,
                payload=request_payload,
                max_candidates=config.transcribe_job_max_candidates,
                api_key=config.youtube_api_key,
            )
        )

    status_code = 200 if result.error_message is None else 400
    return JSONResponse(status_code=status_code, content=result.model_dump())


@jobs_router.get("/jobs/groq/rate-limit")
async def get_groq_rate_limit_status() -> JSONResponse:
    config = get_app_config()
    snapshot = await get_groq_rate_limit_snapshot(
        db_path=config.app_db_path,
        model=config.groq_transcription_model,
    )
    return JSONResponse(status_code=200, content=snapshot.model_dump())
