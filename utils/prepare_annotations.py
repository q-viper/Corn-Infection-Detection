"""Utility entrypoint for drawing annotation previews."""

from pathlib import Path
from typing import Annotated

import typer

from corn_vision.utils.annotations import annotate_vott_image
from corn_vision.utils.logging import setup_logger


app = typer.Typer(help="Draw annotation previews with OpenCV and Supervision.")


@app.command()
def main(
    image_path: Annotated[Path, typer.Option(help="Image file to annotate.")],
    annotation_csv: Annotated[Path, typer.Option(help="VoTT annotation CSV.")],
    output_path: Annotated[Path, typer.Option(help="Where to write the annotated preview.")],
    log_level: Annotated[str, typer.Option(help="Loguru log level.")] = "INFO",
) -> None:
    setup_logger(level=log_level)
    annotate_vott_image(
        image_path=image_path,
        annotation_csv=annotation_csv,
        output_path=output_path,
    )


if __name__ == "__main__":
    app()
