import os
import sys
import typer

cli = typer.Typer()

@cli.command()
def launch(port: int = typer.Option(8080, "--port", "-p")) -> None:
    """Start a graphical UI server for the opyrator.

    The UI is auto-generated from the input- and output-schema of the given function.
    """
    # Add the current working directory to the sys path
    # This is required to resolve the opyrator path
    sys.path.append(os.getcwd())

    from mkgui.base.ui.streamlit_ui import launch_ui
    launch_ui(port)

if __name__ == "__main__":
    cli()