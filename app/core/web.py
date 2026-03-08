from __future__ import annotations

from pathlib import Path

from fastapi.templating import Jinja2Templates


def resolve_app_asset_directory(*, directory_name: str) -> Path:
    packaged_directory = Path(__file__).resolve().parents[1] / directory_name
    if packaged_directory.exists():
        return packaged_directory

    local_workspace_directory = Path.cwd() / "app" / directory_name
    if local_workspace_directory.exists():
        return local_workspace_directory

    raise RuntimeError(f"Could not locate app/{directory_name} directory.")


def build_templates() -> Jinja2Templates:
    templates_directory = resolve_app_asset_directory(directory_name="templates")
    return Jinja2Templates(directory=str(templates_directory))


def get_static_directory() -> Path:
    return resolve_app_asset_directory(directory_name="static")


templates = build_templates()
