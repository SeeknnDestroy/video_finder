from __future__ import annotations

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import HTMLResponse

from app.core.config import get_app_config
from app.core.web import templates
from app.db.database import DatabaseConnectionRequest, get_database_connection
from app.models.history_models import (
    ImportHistoryResult,
    ParseHistoryRequest,
    UpsertHistoryRequest,
)
from app.routers.view_context import build_page_context
from app.services.history_import_service import parse_watch_history, upsert_watch_history

history_router = APIRouter()


@history_router.post("/import/history", response_class=HTMLResponse)
async def import_history(*, request: Request, history_file: UploadFile = File(...)) -> HTMLResponse:
    raw_bytes = await history_file.read()
    parse_result = parse_watch_history(request=ParseHistoryRequest(raw_bytes=raw_bytes))

    if parse_result.error_message:
        import_result = ImportHistoryResult(
            is_successful=False,
            error_message=parse_result.error_message,
            total_parsed_count=0,
            skipped_count=parse_result.skipped_count,
        )
        context = build_page_context(request=request, import_result=import_result)
        return templates.TemplateResponse(request, "index.html", context)

    config = get_app_config()
    async with get_database_connection(
        request=DatabaseConnectionRequest(db_path=config.app_db_path)
    ) as db:
        upsert_result = await upsert_watch_history(
            request=UpsertHistoryRequest(db=db, items=parse_result.items)
        )

    if upsert_result.error_message:
        import_result = ImportHistoryResult(
            is_successful=False,
            error_message=upsert_result.error_message,
            inserted_count=upsert_result.inserted_count,
            deduped_count=upsert_result.deduped_count,
            skipped_count=parse_result.skipped_count,
            total_parsed_count=len(parse_result.items),
        )
        context = build_page_context(request=request, import_result=import_result)
        return templates.TemplateResponse(request, "index.html", context)

    import_result = ImportHistoryResult(
        is_successful=True,
        inserted_count=upsert_result.inserted_count,
        deduped_count=upsert_result.deduped_count,
        skipped_count=parse_result.skipped_count,
        total_parsed_count=len(parse_result.items),
    )

    context = build_page_context(request=request, import_result=import_result)
    return templates.TemplateResponse(request, "index.html", context)
