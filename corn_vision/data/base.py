"""Base dataset abstractions."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from corn_vision.core.configs import DataSetConfig
from corn_vision.core.defs import DataType


@dataclass(frozen=True)
class ImageSample:
    """Single image sample with an encoded label."""

    path: Path
    label: int
    label_name: str


class CornDatasetBase(torch.utils.data.Dataset):
    """Common dataset parent used by task-specific datasets."""

    task_name: str

    def __init__(
        self,
        config: DataSetConfig,
        samples: list[ImageSample] | None = None,
        transform: Any | None = None,
        data_type: DataType | None = None,
    ) -> None:
        self.config = config
        self.samples = samples or []
        self.transform = transform
        self.data_type = data_type
        self.label_encoding: dict[str, int] = {}
        self.num_classes = 0

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def label_counts(self) -> Counter[int]:
        return Counter(sample.label for sample in self.samples)

    @property
    def class_weights(self) -> torch.Tensor:
        """Return inverse-frequency weights for CrossEntropyLoss."""

        counts = self.label_counts
        if not counts:
            return torch.ones(self.num_classes or 1, dtype=torch.float32)
        total = sum(counts.values())
        class_count = self.num_classes or len(counts)
        weights = [
            total / (class_count * max(counts.get(idx, 0), 1))
            for idx in range(class_count)
        ]
        return torch.tensor(weights, dtype=torch.float32)

    def load_data(self) -> None:
        raise NotImplementedError("load_data must be implemented in subclasses.")

    def get_datasets(self) -> tuple["CornDatasetBase", "CornDatasetBase"]:
        raise NotImplementedError("get_datasets must be implemented in subclasses.")
