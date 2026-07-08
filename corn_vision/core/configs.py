"""Pydantic configuration models for corn vision experiments."""

from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from corn_vision.core.defs import (
    MetricType,
    OptimizerType,
    SaliencyMethod,
    TaskType,
    TorchVisionBackbone,
)


class DataSetConfig(BaseModel):
    """Base dataset configuration."""

    image_dir: Path = Field(
        default=Path("assets/Corn Disease detection"),
        description="Root directory containing corn images.",
    )
    image_size: tuple[int, int] = Field(
        default=(224, 224),
        description="Image size as (height, width).",
    )
    train_ratio: float = Field(
        default=0.8,
        ge=0.05,
        le=0.95,
        description="Ratio of samples used for training.",
    )
    max_data_per_class: int = Field(
        default=-1,
        description="Maximum samples per class. Use -1 for all samples.",
    )
    random_state: int = Field(default=42, description="Random seed for data splits.")
    image_extensions: tuple[str, ...] = Field(
        default=(".jpg", ".jpeg", ".png", ".bmp"),
        description="Image extensions to include.",
    )
    pad_value: tuple[int, int, int] = Field(
        default=(0, 0, 0),
        description="RGB padding value used after aspect-ratio-preserving resize.",
    )

    @field_validator("image_size")
    @classmethod
    def validate_image_size(cls, value: tuple[int, int]) -> tuple[int, int]:
        if value[0] <= 0 or value[1] <= 0:
            raise ValueError("image_size values must be positive.")
        return value

    @field_validator("pad_value")
    @classmethod
    def validate_pad_value(cls, value: tuple[int, int, int]) -> tuple[int, int, int]:
        if len(value) != 3:
            raise ValueError("pad_value must contain three RGB values.")
        if any(channel < 0 or channel > 255 for channel in value):
            raise ValueError("pad_value channels must be in the range [0, 255].")
        return value


class ClassificationDataSetConfig(DataSetConfig):
    """Dataset config for binary healthy/infected classification."""

    task: TaskType = TaskType.CLASSIFICATION
    healthy_dir_name: str = "Healthy corn"
    infected_dir_name: str = "Infected"
    class_names: tuple[str, str] = ("Healthy", "Infected")


class ModelConfig(BaseModel):
    """Base model configuration."""

    task: TaskType
    num_classes: int = Field(default=2, ge=2)


class ClassificationModelConfig(ModelConfig):
    """Torchvision binary classifier configuration."""

    task: TaskType = TaskType.CLASSIFICATION
    backbone: TorchVisionBackbone = TorchVisionBackbone.RESNET18
    pretrained: bool = Field(
        default=True,
        description="Use Torchvision default pretrained weights when available.",
    )
    train_backbone: bool = Field(
        default=False,
        description="Train the feature backbone. If false, only the classifier head trains.",
    )
    dropout: float = Field(default=0.2, ge=0.0, le=0.9)
    gradcam_target_layer: str | None = Field(
        default=None,
        description="Optional module path used for Grad-CAM.",
    )
    saliency_method: SaliencyMethod = SaliencyMethod.GRADCAM


class NNTrainerConfig(BaseModel):
    """Base neural network trainer configuration."""

    result_dir: Path = Field(default=Path("results"))
    expt_name: str = "classification"
    run_name: str = "resnet18_baseline"
    log_every: int = 1
    chkpt_every: int = 0
    best_model_name: str = "best_model.pth"
    best_model_metric: MetricType = MetricType.LOSS
    best_model_metric_greater: bool = False
    optimizer: OptimizerType = OptimizerType.ADAMW
    device: str = "cuda"
    epochs: int = Field(default=20, ge=1)
    batch_size: int = Field(default=16, ge=1)
    shuffle: bool = True
    number_of_workers: int = Field(default=0, ge=0)
    log_file: str = "trainer.log"
    log_level: str = "INFO"
    metric_file: str = "metrics.csv"
    progress_image_every: int = Field(
        default=0,
        ge=0,
        description="Save a validation progress image every N epochs. Use 0 to disable.",
    )
    progress_image_count: int = Field(
        default=8,
        ge=1,
        description="Number of validation samples to include in each progress image.",
    )
    progress_image_dir: str = "progress_images"
    log_sample_images: bool = Field(
        default=True,
        description="Save train and validation sample image grids before training starts.",
    )
    sample_image_count: int = Field(
        default=8,
        ge=1,
        description="Number of samples per split to include in initial sample image grids.",
    )
    sample_image_dir: str = "sample_images"
    metrics: list[MetricType] = Field(
        default_factory=lambda: [
            MetricType.ACCURACY,
            MetricType.PRECISION,
            MetricType.RECALL,
            MetricType.F1_SCORE,
        ]
    )
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    early_stopping_patience: int = Field(default=10, ge=1)
    weighted_loss: bool = True
    random_seed: int = 42

    @property
    def run_dir(self) -> Path:
        return self.result_dir / self.expt_name / self.run_name


class ClassificationExperimentConfig(BaseModel):
    """Full config bundle for a binary classification run."""

    data: ClassificationDataSetConfig = Field(default_factory=ClassificationDataSetConfig)
    model: ClassificationModelConfig = Field(default_factory=ClassificationModelConfig)
    trainer: NNTrainerConfig = Field(default_factory=NNTrainerConfig)


ClassificationDataConfig = ClassificationDataSetConfig
