from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import aiosqlite
from pydantic import BaseModel


class InitializeDatabaseRequest(BaseModel):
    db_path: str


class InitializeDatabaseResult(BaseModel):
    is_successful: bool
    error_message: str | None = None


class DatabaseConnectionRequest(BaseModel):
    db_path: str


def resolve_schema_path() -> Path | None:
    packaged_schema_path = Path(__file__).resolve().parent / "schema.sql"
    if packaged_schema_path.exists():
        return packaged_schema_path

    local_workspace_schema_path = Path.cwd() / "app" / "db" / "schema.sql"
    if local_workspace_schema_path.exists():
        return local_workspace_schema_path

    return None


async def initialize_database(*, request: InitializeDatabaseRequest) -> InitializeDatabaseResult:
    db_path = request.db_path.strip()
    if not db_path:
        return InitializeDatabaseResult(
            is_successful=False,
            error_message="Database path is missing.",
        )

    schema_path = resolve_schema_path()
    if schema_path is None:
        return InitializeDatabaseResult(
            is_successful=False,
            error_message="Schema file not found in packaged or workspace locations.",
        )

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    schema_sql = schema_path.read_text(encoding="utf-8")

    try:
        async with aiosqlite.connect(db_path) as db:
            await db.executescript(schema_sql)
            await db.commit()
    except Exception as exc:
        return InitializeDatabaseResult(
            is_successful=False,
            error_message=f"Failed to initialize database: {exc}",
        )

    return InitializeDatabaseResult(is_successful=True)


@asynccontextmanager
async def get_database_connection(*, request: DatabaseConnectionRequest) -> AsyncIterator[aiosqlite.Connection]:
    db = await aiosqlite.connect(request.db_path)
    db.row_factory = aiosqlite.Row

    try:
        yield db
    finally:
        await db.close()
