from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.core.config import get_app_config
from app.core.web import get_static_directory
from app.db.database import InitializeDatabaseRequest, initialize_database
from app.routers.history_routes import history_router
from app.routers.jobs_routes import jobs_router
from app.routers.search_routes import search_router
from app.services.job_runner_service import WorkerRunRequest, run_transcription_worker


@asynccontextmanager
async def lifespan(_: FastAPI):
    config = get_app_config()
    logging.basicConfig(level=config.log_level)

    initialize_result = await initialize_database(
        request=InitializeDatabaseRequest(db_path=config.app_db_path)
    )
    if not initialize_result.is_successful:
        logging.error(initialize_result.error_message)

    worker_task: asyncio.Task | None = None
    if config.transcribe_worker_enabled:
        worker_task = asyncio.create_task(
            run_transcription_worker(
                request=WorkerRunRequest(
                    db_path=config.app_db_path,
                    language=config.transcribe_language,
                    max_concurrency=config.transcribe_worker_concurrency,
                    poll_seconds=config.transcribe_worker_poll_seconds,
                    run_once=False,
                )
            )
        )

    yield

    if worker_task is not None:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="YouTube Video Finder", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(get_static_directory())), name="static")

app.include_router(search_router)
app.include_router(history_router)
app.include_router(jobs_router)
