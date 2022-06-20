"""Collection of utilities for FastAPI apps."""

import inspect
from typing import Any, Type

from fastapi import FastAPI, Form
from pydantic import BaseModel


def as_form(cls: Type[BaseModel]) -> Any:
    """Adds an as_form class method to decorated models.

    The as_form class method can be used with FastAPI endpoints
    """
    new_params = [
        inspect.Parameter(
            field.alias,
            inspect.Parameter.POSITIONAL_ONLY,
            default=(Form(field.default) if not field.required else Form(...)),
        )
        for field in cls.__fields__.values()
    ]

    async def _as_form(**data):  # type: ignore
        return cls(**data)

    sig = inspect.signature(_as_form)
    sig = sig.replace(parameters=new_params)
    _as_form.__signature__ = sig  # type: ignore
    setattr(cls, "as_form", _as_form)
    return cls


def patch_fastapi(app: FastAPI) -> None:
    """Patch function to allow relative url resolution.

    This patch is required to make fastapi fully functional with a relative url path.
    This code snippet can be copy-pasted to any Fastapi application.
    """
    from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
    from starlette.requests import Request
    from starlette.responses import HTMLResponse

    async def redoc_ui_html(req: Request) -> HTMLResponse:
        assert app.openapi_url is not None
        redoc_ui = get_redoc_html(
            openapi_url="./" + app.openapi_url.lstrip("/"),
            title=app.title + " - Redoc UI",
        )

        return HTMLResponse(redoc_ui.body.decode("utf-8"))

    async def swagger_ui_html(req: Request) -> HTMLResponse:
        assert app.openapi_url is not None
        swagger_ui = get_swagger_ui_html(
            openapi_url="./" + app.openapi_url.lstrip("/"),
            title=app.title + " - Swagger UI",
            oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        )

        # insert request interceptor to have all request run on relativ path
        request_interceptor = (
            "requestInterceptor: (e)  => {"
            "\n\t\t\tvar url = window.location.origin + window.location.pathname"
            '\n\t\t\turl = url.substring( 0, url.lastIndexOf( "/" ) + 1);'
            "\n\t\t\turl = e.url.replace(/http(s)?:\/\/[^/]*\//i, url);"  # noqa: W605
            "\n\t\t\te.contextUrl = url"
            "\n\t\t\te.url = url"
            "\n\t\t\treturn e;}"
        )

        return HTMLResponse(
            swagger_ui.body.decode("utf-8").replace(
                "dom_id: '#swagger-ui',",
                "dom_id: '#swagger-ui',\n\t\t" + request_interceptor + ",",
            )
        )

    # remove old docs route and add our patched route
    routes_new = []
    for app_route in app.routes:
        if app_route.path == "/docs":  # type: ignore
            continue

        if app_route.path == "/redoc":  # type: ignore
            continue

        routes_new.append(app_route)

    app.router.routes = routes_new

    assert app.docs_url is not None
    app.add_route(app.docs_url, swagger_ui_html, include_in_schema=False)
    assert app.redoc_url is not None
    app.add_route(app.redoc_url, redoc_ui_html, include_in_schema=False)

    # Make graphql realtive
    from starlette import graphql

    graphql.GRAPHIQL = graphql.GRAPHIQL.replace(
        "({{REQUEST_PATH}}", '("." + {{REQUEST_PATH}}'
    )
