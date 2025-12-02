# # Copyright (C) KonaAI - All Rights Reserved
"""This module contains the main NiceGUI application."""
import secrets
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import RedirectResponse
from nicegui import app
from nicegui import ui
from src.admin.all_pages import create_ui_routes
from src.utils.conf import Setup
from src.utils.status import Status
from starlette.middleware.base import BaseHTTPMiddleware

unrestricted_page_routes = {"/login"}


class AuthMiddleware(BaseHTTPMiddleware):
    """This middleware restricts access to all NiceGUI pages.

    It redirects the user to the login page if they are not authenticated.
    """

    async def dispatch(self, request: Request, call_next):
        """Check if the user is authenticated and redirect to the login page if not."""
        if request.url.path.startswith("/api"):
            return await call_next(request)

        if not app.storage.user.get("authenticated", False):
            if (
                not request.url.path.startswith("/_nicegui")
                and request.url.path not in unrestricted_page_routes
            ):
                app.storage.user["referrer_path"] = request.url.path
                return RedirectResponse("/login")
        return await call_next(request)


def setup_nicegui_app(fastapi_app: FastAPI) -> app:  # type: ignore
    """
    Creates and configures a NiceGUI application instance integrated with a FastAPI app.
    This function sets up UI routes, adds authentication middleware, and runs the NiceGUI app
    with specified settings such as title, storage secret, and favicon.

    Args:
        fastapi_app (FastAPI): The FastAPI application to integrate with NiceGUI.

    Returns:
        app: The configured NiceGUI application instance.
    """
    # Setup routes and middleware
    create_ui_routes()

    # Implement the authentication middleware
    app.add_middleware(AuthMiddleware)

    app.on_exception(_handle_exception)

    favicon_path = Path(Setup().assets_path, "konaai.ico")
    ui.run_with(
        fastapi_app,
        title="KonaAI Intelligence",
        storage_secret=secrets.token_urlsafe(32),  # Set secret first
        favicon=favicon_path,
    )


def _handle_exception(exc: Optional[Exception] = None):
    """Handle uncaught exceptions in the NiceGUI app."""
    Status.FAILED("An unexpected error occurred in the NiceGUI application.", error=exc)


if __name__ in {"__main__", "__mp_main__"}:
    # Run the app with the default settings
    create_ui_routes()
    ui.run(
        title="KonaAI Intelligence",
        storage_secret=secrets.token_urlsafe(32),
        reload=True,
    )
