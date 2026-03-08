from __future__ import annotations

import asyncio
import logging

from app.core.config import get_app_config
from app.services.job_runner_service import WorkerRunRequest, run_transcription_worker


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = get_app_config()
    result = asyncio.run(
        run_transcription_worker(
            request=WorkerRunRequest(
                db_path=config.app_db_path,
                language=config.transcribe_language,
                max_concurrency=config.transcribe_worker_concurrency,
                poll_seconds=config.transcribe_worker_poll_seconds,
                run_once=True,
            )
        )
    )
    if not result.is_successful and result.error_message:
        logging.error(result.error_message)
