from typing import Any, Dict

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse

from mkgui.base import Opyrator
from mkgui.base.api.fastapi_utils import patch_fastapi


def launch_api(opyrator_path: str, port: int = 8501, host: str = "0.0.0.0") -> None:
    import uvicorn

    from mkgui.base import Opyrator
    from mkgui.base.api import create_api

    app = create_api(Opyrator(opyrator_path))
    uvicorn.run(app, host=host, port=port, log_level="info")


def create_api(opyrator: Opyrator) -> FastAPI:

    title = opyrator.name
    if "opyrator" not in opyrator.name.lower():
        title += " - Opyrator"

    # TODO what about version?
    app = FastAPI(title=title, description=opyrator.description)

    patch_fastapi(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post(
        "/call",
        operation_id="call",
        response_model=opyrator.output_type,
        # response_model_exclude_unset=True,
        summary="Execute the opyrator.",
        status_code=status.HTTP_200_OK,
    )
    def call(input: opyrator.input_type) -> Any:  # type: ignore
        """Executes this opyrator."""
        return opyrator(input)

    @app.get(
        "/info",
        operation_id="info",
        response_model=Dict,
        # response_model_exclude_unset=True,
        summary="Get info metadata.",
        status_code=status.HTTP_200_OK,
    )
    def info() -> Any:  # type: ignore
        """Returns informational metadata about this Opyrator."""
        return {}

    # Redirect to docs
    @app.get("/", include_in_schema=False)
    def root() -> Any:
        return RedirectResponse("./docs")

    return app
