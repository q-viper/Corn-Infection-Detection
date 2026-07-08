"""Top-level entrypoint for binary corn infection classification training."""

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from corn_vision.core.configs import (
    ClassificationDataSetConfig,
    ClassificationModelConfig,
    NNTrainerConfig,
)
from corn_vision.core.defs import OptimizerType, TorchVisionBackbone
from corn_vision.data.classification_dataset import CornClassificationDataset
from corn_vision.models.classification import TorchVisionBinaryClassifier
from corn_vision.trainers.classification import ClassificationTrainer


app = typer.Typer(help="Train a local binary corn infection classifier.")


@app.command()
def main(
    image_dir: Annotated[
        Path,
        typer.Option(help="Dataset root with 'Healthy corn' and 'Infected' folders."),
    ] = Path("assets/Corn Disease detection"),
    result_dir: Annotated[
        Path,
        typer.Option(help="Directory where training outputs will be written."),
    ] = Path("results"),
    run_name: Annotated[
        str,
        typer.Option(help="Run folder name. Auto-generated when empty."),
    ] = "",
    backbone: Annotated[
        TorchVisionBackbone,
        typer.Option(help="Torchvision backbone to use."),
    ] = TorchVisionBackbone.RESNET18,
    epochs: Annotated[int, typer.Option(min=1, help="Number of training epochs.")] = 20,
    batch_size: Annotated[int, typer.Option(min=1, help="Batch size.")] = 16,
    image_height: Annotated[int, typer.Option(min=1, help="Input image height.")] = 224,
    image_width: Annotated[int, typer.Option(min=1, help="Input image width.")] = 224,
    learning_rate: Annotated[float, typer.Option(min=0.0, help="Learning rate.")] = 1e-4,
    weight_decay: Annotated[float, typer.Option(min=0.0, help="Weight decay.")] = 1e-4,
    optimizer: Annotated[
        OptimizerType,
        typer.Option(help="Optimizer to use."),
    ] = OptimizerType.ADAMW,
    train_backbone: Annotated[
        bool,
        typer.Option(help="Fine-tune the full backbone instead of only the classifier head."),
    ] = False,
    pretrained: Annotated[
        bool,
        typer.Option("--pretrained/--no-pretrained", help="Use Torchvision pretrained weights."),
    ] = True,
    max_data_per_class: Annotated[
        int,
        typer.Option(help="Limit samples per class. Use -1 for all samples."),
    ] = -1,
    num_workers: Annotated[int, typer.Option(min=0, help="DataLoader workers.")] = 0,
    log_level: Annotated[str, typer.Option(help="Loguru log level.")] = "INFO",
    progress_image_every: Annotated[
        int,
        typer.Option(
            min=0,
            help="Save a validation progress image every N epochs. Use 0 to disable.",
        ),
    ] = 0,
    progress_image_count: Annotated[
        int,
        typer.Option(min=1, help="Validation samples per progress image."),
    ] = 8,
    log_sample_images: Annotated[
        bool,
        typer.Option(
            "--log-sample-images/--no-log-sample-images",
            help="Save train and validation sample grids before training starts.",
        ),
    ] = True,
    sample_image_count: Annotated[
        int,
        typer.Option(min=1, help="Samples per split in the initial sample grids."),
    ] = 8,
) -> None:
    run_name = run_name or f"{backbone.value}_{datetime.now().strftime('%Y%m%d')}"
    data_config = ClassificationDataSetConfig(
        image_dir=image_dir,
        image_size=(image_height, image_width),
        max_data_per_class=max_data_per_class,
    )
    model_config = ClassificationModelConfig(
        backbone=backbone,
        pretrained=pretrained,
        train_backbone=train_backbone,
    )
    trainer_config = NNTrainerConfig(
        result_dir=result_dir,
        expt_name="classification",
        run_name=run_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer=optimizer,
        number_of_workers=num_workers,
        log_level=log_level,
        progress_image_every=progress_image_every,
        progress_image_count=progress_image_count,
        log_sample_images=log_sample_images,
        sample_image_count=sample_image_count,
    )
    dataset = CornClassificationDataset(config=data_config)
    train_dataset, val_dataset = dataset.get_datasets()
    model = TorchVisionBinaryClassifier(config=model_config)
    trainer = ClassificationTrainer(
        config=trainer_config,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    best_model_path = trainer.train()
    typer.echo(f"Best model saved to {best_model_path}")


if __name__ == "__main__":
    app()
