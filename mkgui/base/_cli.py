"""Command line interface."""

import os
import sys

import typer
from pydantic.error_wrappers import ValidationError

cli = typer.Typer()

@cli.command()
def launch_ui(opyrator: str, port: int = typer.Option(8051, "--port", "-p")) -> None:
    """Start a graphical UI server for the opyrator.

    The UI is auto-generated from the input- and output-schema of the given function.
    """
    # Add the current working directory to the sys path
    # This is required to resolve the opyrator path
    sys.path.append(os.getcwd())

    from mkgui.base.ui.streamlit_ui import launch_ui
    launch_ui(opyrator, port)


@cli.command()
def launch_api(
    opyrator: str,
    port: int = typer.Option(8080, "--port", "-p"),
    host: str = typer.Option("0.0.0.0", "--host", "-h"),
) -> None:
    """Start a HTTP API server for the opyrator.

    This will launch a FastAPI server based on the OpenAPI standard and with an automatic interactive documentation.
    """
    # Add the current working directory to the sys path
    # This is required to resolve the opyrator path
    sys.path.append(os.getcwd())

    from mkgui.base.api.fastapi_app import launch_api  # type: ignore

    launch_api(opyrator, port, host)


@cli.command()
def call(opyrator: str, input_data: str) -> None:
    """Execute the opyrator from command line."""
    # Add the current working directory to the sys path
    # This is required to resolve the opyrator path
    sys.path.append(os.getcwd())

    try:
        from mkgui.base import Opyrator

        output = Opyrator(opyrator)(input_data)
        if output:
            typer.echo(output.json(indent=4))
        else:
            typer.echo("Nothing returned!")
    except ValidationError as ex:
        typer.secho(str(ex), fg=typer.colors.RED, err=True)


if __name__ == "__main__":
    cli()