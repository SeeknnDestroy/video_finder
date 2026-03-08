from __future__ import annotations

from typing import Any

from fastapi import Request

from app.models.history_models import ImportHistoryResult
from app.models.search_models import SearchVideosRequest, SearchVideosResult

DATE_PRESET_OPTIONS: list[tuple[str, str]] = [
    ("7d", "Last 7 days"),
    ("30d", "Last 30 days"),
    ("6m", "Last 6 months"),
    ("1y", "Last year"),
    ("custom", "Custom range"),
]


def build_default_search_request() -> SearchVideosRequest:
    return SearchVideosRequest()


def build_page_context(
    *,
    request: Request,
    search_request: SearchVideosRequest | None = None,
    search_result: SearchVideosResult | None = None,
    import_result: ImportHistoryResult | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    resolved_search_request = search_request or build_default_search_request()

    return {
        "request": request,
        "search_request": resolved_search_request,
        "search_result": search_result,
        "import_result": import_result,
        "warnings": warnings or [],
        "date_preset_options": DATE_PRESET_OPTIONS,
    }
