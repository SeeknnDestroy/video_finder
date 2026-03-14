from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import ValidationError

from app.core.config import get_app_config
from app.core.web import templates
from app.db.database import DatabaseConnectionRequest, get_database_connection
from app.models.job_models import GetJobStatusServiceRequest, SpokenSearchRequest, SpokenSearchServiceRequest
from app.models.search_models import SearchVideosRequest, SearchVideosServiceRequest
from app.routers.view_context import build_page_context
from app.services.search_service import search_videos
from app.services.transcription_service import get_transcription_job_status, search_spoken_transcripts

search_router = APIRouter()

TRANSCRIPT_SEARCH_WARNING = (
    "Transcript phrase search can take longer on first run while missing transcripts are prepared in the background."
)


@search_router.get("/", response_class=HTMLResponse)
async def home(*, request: Request) -> HTMLResponse:
    context = build_page_context(request=request)
    context.update({"spoken_result": None, "job_status": None})
    return templates.TemplateResponse(request, "index.html", context)


@search_router.get("/search", response_class=HTMLResponse)
async def search(
    *,
    request: Request,
    phrase: str | None = Query(default=None),
    title_query: str | None = Query(default=None),
    channel_query: str | None = Query(default=None),
    duration_min_seconds: str | None = Query(default=None),
    duration_max_seconds: str | None = Query(default=None),
    watched_from: str | None = Query(default=None),
    watched_to: str | None = Query(default=None),
    date_preset: str | None = Query(default=None),
    limit: str | None = Query(default=None),
    language: str | None = Query(default=None),
) -> HTMLResponse:
    try:
        search_request = SearchVideosRequest(
            phrase=phrase,
            title_query=title_query,
            channel_query=channel_query,
            duration_min_seconds=duration_min_seconds,
            duration_max_seconds=duration_max_seconds,
            watched_from=watched_from,
            watched_to=watched_to,
            date_preset=date_preset,
            limit=limit,
        )
    except ValidationError as exc:
        warning_messages = [f"Invalid search input: {exc.errors()}"]
        context = build_page_context(
            request=request,
            search_request=SearchVideosRequest(),
            warnings=warning_messages,
        )
        context.update({"spoken_result": None, "job_status": None})
        return templates.TemplateResponse(request, "results.html", context)

    config = get_app_config()
    async with get_database_connection(
        request=DatabaseConnectionRequest(db_path=config.app_db_path)
    ) as db:
        if search_request.phrase:
            try:
                spoken_request = SpokenSearchRequest(
                    phrase=search_request.phrase,
                    title_query=search_request.title_query,
                    channel_query=search_request.channel_query,
                    duration_min_seconds=search_request.duration_min_seconds,
                    duration_max_seconds=search_request.duration_max_seconds,
                    watched_from=search_request.watched_from,
                    watched_to=search_request.watched_to,
                    date_preset=search_request.date_preset,
                    limit=search_request.limit or 50,
                    language=language,
                )
            except ValidationError as exc:
                warning_messages = [f"Invalid transcript-phrase input: {exc.errors()}"]
                context = build_page_context(
                    request=request,
                    search_request=search_request,
                    warnings=warning_messages,
                )
                context.update({"spoken_result": None, "job_status": None})
                return templates.TemplateResponse(request, "results.html", context)

            spoken_result = await search_spoken_transcripts(
                request=SpokenSearchServiceRequest(
                    db=db,
                    payload=spoken_request,
                    max_candidates=config.transcribe_job_max_candidates,
                    api_key=config.youtube_api_key,
                )
            )

            job_status = None
            if spoken_result.job_id:
                job_status = await get_transcription_job_status(
                    request=GetJobStatusServiceRequest(
                        db=db,
                        job_id=spoken_result.job_id,
                        include_items=False,
                    )
                )

            warnings = [TRANSCRIPT_SEARCH_WARNING, *spoken_result.warnings]
            context = build_page_context(
                request=request,
                search_request=search_request,
                warnings=warnings,
            )
            context.update(
                {
                    "spoken_result": spoken_result,
                    "search_result": None,
                    "job_status": job_status,
                }
            )
            return templates.TemplateResponse(request, "results.html", context)

        search_result = await search_videos(
            request=SearchVideosServiceRequest(
                db=db,
                search=search_request,
                api_key=config.youtube_api_key,
            )
        )

    context = build_page_context(
        request=request,
        search_request=search_request,
        search_result=search_result,
        warnings=search_result.warnings,
    )
    context.update({"spoken_result": None, "job_status": None})
    return templates.TemplateResponse(request, "results.html", context)


@search_router.get("/progress/{job_id}", response_class=HTMLResponse)
async def progress_page(*, request: Request, job_id: str) -> HTMLResponse:
    config = get_app_config()
    async with get_database_connection(
        request=DatabaseConnectionRequest(db_path=config.app_db_path)
    ) as db:
        job_status = await get_transcription_job_status(
            request=GetJobStatusServiceRequest(
                db=db,
                job_id=job_id,
                include_items=True,
            )
        )

    if job_status is None:
        context = build_page_context(request=request, warnings=["Job not found."])
        context.update({"job_status": None, "auto_refresh": False})
        return templates.TemplateResponse(request, "progress.html", context, status_code=404)

    is_running = job_status.status in {"queued", "running"}
    context = build_page_context(request=request)
    context.update(
        {
            "job_status": job_status,
            "auto_refresh": is_running,
        }
    )
    return templates.TemplateResponse(request, "progress.html", context)


@search_router.get("/spoken")
async def spoken_route_redirect() -> RedirectResponse:
    return RedirectResponse(url="/", status_code=307)
