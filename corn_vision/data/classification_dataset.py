"""Binary classification dataset for healthy vs. infected corn leaves."""

from collections import defaultdict
from pathlib import Path
import random

import albumentations as A
import cv2
from loguru import logger
import numpy as np
import torch

from corn_vision.core.configs import ClassificationDataSetConfig
from corn_vision.core.defs import DataType, TaskType
from corn_vision.data.base import CornDatasetBase, ImageSample


def resize_with_aspect_ratio_and_pad(
    image: np.ndarray,
    target_size: tuple[int, int],
    pad_value: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Resize an RGB image without distortion and pad to target size."""

    target_height, target_width = target_size
    image_height, image_width = image.shape[:2]
    if image_height <= 0 or image_width <= 0:
        raise ValueError("Image must have positive height and width.")

    scale = min(target_width / image_width, target_height / image_height)
    resized_width = max(1, int(round(image_width * scale)))
    resized_height = max(1, int(round(image_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=interpolation,
    )

    canvas = np.full(
        (target_height, target_width, 3),
        pad_value,
        dtype=resized.dtype,
    )
    top = (target_height - resized_height) // 2
    left = (target_width - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert an RGB uint8 image to a float tensor normalized by division by 255."""

    image = np.ascontiguousarray(image)
    return torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)


class ClassificationTransform:
    """Albumentations transform followed by aspect-ratio resize and /255 tensor conversion."""

    def __init__(self, config: ClassificationDataSetConfig, data_type: DataType) -> None:
        self.config = config
        self.data_type = data_type
        transforms = []
        if data_type == DataType.TRAIN:
            transforms.extend(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                ]
            )
        self.augment = A.Compose(transforms) if transforms else None

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        if self.augment is not None:
            image = self.augment(image=image)["image"]
        image = resize_with_aspect_ratio_and_pad(
            image=image,
            target_size=self.config.image_size,
            pad_value=self.config.pad_value,
        )
        return image_to_tensor(image)


def build_classification_transform(
    config: ClassificationDataSetConfig,
    data_type: DataType,
) -> ClassificationTransform:
    """Create Albumentations-based transforms for classification."""

    return ClassificationTransform(config=config, data_type=data_type)


class CornClassificationDataset(CornDatasetBase):
    """Dataset for binary healthy/infected corn leaf classification."""

    task_name = TaskType.CLASSIFICATION.value

    def __init__(
        self,
        config: ClassificationDataSetConfig,
        samples: list[ImageSample] | None = None,
        transform=None,
        data_type: DataType | None = None,
    ) -> None:
        super().__init__(
            config=config,
            samples=samples,
            transform=transform,
            data_type=data_type,
        )
        self.config: ClassificationDataSetConfig
        self.label_encoding = {
            config.class_names[0]: 0,
            config.class_names[1]: 1,
        }
        self.num_classes = len(self.label_encoding)

    def load_data(self) -> None:
        image_dir = Path(self.config.image_dir)
        class_dirs = {
            self.config.class_names[0]: image_dir / self.config.healthy_dir_name,
            self.config.class_names[1]: image_dir / self.config.infected_dir_name,
        }
        samples: list[ImageSample] = []
        for label_name, label_dir in class_dirs.items():
            if not label_dir.exists():
                raise FileNotFoundError(f"Expected class directory does not exist: {label_dir}")
            label = self.label_encoding[label_name]
            image_paths = [
                path
                for path in sorted(label_dir.rglob("*"))
                if path.suffix.lower() in self.config.image_extensions
            ]
            if self.config.max_data_per_class > 0:
                image_paths = image_paths[: self.config.max_data_per_class]
            samples.extend(
                ImageSample(path=path, label=label, label_name=label_name)
                for path in image_paths
            )
        if not samples:
            raise ValueError(f"No images found in {image_dir}.")
        self.samples = samples
        label_counts = {
            label_name: sum(sample.label_name == label_name for sample in self.samples)
            for label_name in self.label_encoding
        }
        logger.info(f"Loaded classification samples from {image_dir}: {label_counts}")

    def get_datasets(self) -> tuple["CornClassificationDataset", "CornClassificationDataset"]:
        self.load_data()
        rng = random.Random(self.config.random_state)
        grouped: dict[int, list[ImageSample]] = defaultdict(list)
        for sample in self.samples:
            grouped[sample.label].append(sample)

        train_samples: list[ImageSample] = []
        val_samples: list[ImageSample] = []
        for label_samples in grouped.values():
            rng.shuffle(label_samples)
            split_idx = max(1, int(len(label_samples) * self.config.train_ratio))
            if len(label_samples) > 1:
                split_idx = min(split_idx, len(label_samples) - 1)
            train_samples.extend(label_samples[:split_idx])
            val_samples.extend(label_samples[split_idx:])

        rng.shuffle(train_samples)
        rng.shuffle(val_samples)
        logger.info(
            f"Created classification split with {len(train_samples)} train and "
            f"{len(val_samples)} validation samples"
        )
        return (
            self.__class__(
                config=self.config,
                samples=train_samples,
                transform=build_classification_transform(self.config, DataType.TRAIN),
                data_type=DataType.TRAIN,
            ),
            self.__class__(
                config=self.config,
                samples=val_samples,
                transform=build_classification_transform(self.config, DataType.VALIDATION),
                data_type=DataType.VALIDATION,
            ),
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        image = cv2.imread(str(sample.path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {sample.path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image = resize_with_aspect_ratio_and_pad(
                image=image,
                target_size=self.config.image_size,
                pad_value=self.config.pad_value,
            )
            image_tensor = image_to_tensor(image)
        label = torch.tensor(sample.label, dtype=torch.long)
        return image_tensor, label
