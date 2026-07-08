"""Classification trainer."""

import math
import random

import cv2
import numpy as np
import torch

from corn_vision.core.defs import TaskType
from corn_vision.trainers.base import TrainerBase
from corn_vision.utils.image import write_image


class ClassificationTrainer(TrainerBase):
    """Trainer for binary corn infection classification."""

    task_name = TaskType.CLASSIFICATION.value

    def at_train_start(self) -> None:
        if not self.config.log_sample_images:
            return
        self.log_sample_images()

    def at_epoch_end(self) -> None:
        if self.config.progress_image_every <= 0:
            return
        if (self.epoch + 1) % self.config.progress_image_every != 0:
            return
        self.log_progress_image()

    def log_sample_images(self) -> None:
        self._log_sample_grid(
            dataset=self.train_dataset,
            split_name="train",
            output_name="train_samples.jpg",
        )
        self._log_sample_grid(
            dataset=self.val_dataset,
            split_name="validation",
            output_name="validation_samples.jpg",
        )

    def _log_sample_grid(self, dataset, split_name: str, output_name: str) -> None:
        if len(dataset) == 0:
            self.logger.warning(f"Skipping {split_name} sample image because dataset is empty")
            return
        sample_count = min(self.config.sample_image_count, len(dataset))
        rng = random.Random(self.config.random_seed)
        sample_indices = rng.sample(range(len(dataset)), sample_count)
        tiles = []
        for idx in sample_indices:
            image, label = dataset[idx]
            tiles.append(self._make_sample_tile(image=image, label=int(label.item()), dataset=dataset))
        sample_grid = self._make_grid(tiles)
        output_dir = self.config.run_dir / self.config.sample_image_dir
        output_path = output_dir / output_name
        write_image(output_path, sample_grid)
        self.logger.info(f"Saved {split_name} sample image grid to {output_path}")

    def log_progress_image(self) -> None:
        if len(self.val_dataset) == 0:
            self.logger.warning("Skipping progress image because validation dataset is empty")
            return

        sample_count = min(self.config.progress_image_count, len(self.val_dataset))
        rng = random.Random(self.config.random_seed + self.epoch)
        sample_indices = rng.sample(range(len(self.val_dataset)), sample_count)
        images = []
        labels = []
        for idx in sample_indices:
            image, label = self.val_dataset[idx]
            images.append(image)
            labels.append(label)

        image_batch = torch.stack(images).to(self.device)
        label_tensor = torch.stack(labels)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            logits = self.model(image_batch)
            probabilities = torch.softmax(logits, dim=1).detach().cpu()
            predictions = probabilities.argmax(dim=1)
            confidences = probabilities.max(dim=1).values
        if was_training:
            self.model.train()

        tiles = [
            self._make_progress_tile(
                image=images[idx],
                true_label=int(label_tensor[idx].item()),
                predicted_label=int(predictions[idx].item()),
                confidence=float(confidences[idx].item()),
            )
            for idx in range(sample_count)
        ]
        progress_image = self._make_grid(tiles)
        output_dir = self.config.run_dir / self.config.progress_image_dir
        output_path = output_dir / f"epoch_{self.epoch + 1:04d}.jpg"
        write_image(output_path, progress_image)
        self.logger.info(f"Saved classification progress image to {output_path}")

    def _make_sample_tile(self, image: torch.Tensor, label: int, dataset) -> np.ndarray:
        tile = self._tensor_to_bgr(image, dataset=dataset)
        header_height = 24
        tile = cv2.copyMakeBorder(
            tile,
            header_height,
            0,
            0,
            0,
            borderType=cv2.BORDER_CONSTANT,
            value=(20, 20, 20),
        )
        class_names = getattr(dataset.config, "class_names", ("class_0", "class_1"))
        label_name = class_names[label] if label < len(class_names) else str(label)
        cv2.putText(
            tile,
            f"label: {label_name}",
            (6, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        return tile

    def _make_progress_tile(
        self,
        image: torch.Tensor,
        true_label: int,
        predicted_label: int,
        confidence: float,
    ) -> np.ndarray:
        tile = self._tensor_to_bgr(image, dataset=self.val_dataset)
        header_height = 42
        tile = cv2.copyMakeBorder(
            tile,
            header_height,
            0,
            0,
            0,
            borderType=cv2.BORDER_CONSTANT,
            value=(20, 20, 20),
        )
        class_names = getattr(self.val_dataset.config, "class_names", ("class_0", "class_1"))
        true_name = class_names[true_label] if true_label < len(class_names) else str(true_label)
        predicted_name = (
            class_names[predicted_label]
            if predicted_label < len(class_names)
            else str(predicted_label)
        )
        color = (40, 190, 80) if true_label == predicted_label else (40, 40, 220)
        cv2.putText(
            tile,
            f"true: {true_name}",
            (6, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            tile,
            f"pred: {predicted_name} ({confidence:.2f})",
            (6, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        return tile

    def _tensor_to_bgr(self, image: torch.Tensor, dataset) -> np.ndarray:
        image = image.detach().cpu().clone()
        image = image.clamp(0, 1)
        image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    def _make_grid(self, tiles: list[np.ndarray]) -> np.ndarray:
        columns = max(1, math.ceil(math.sqrt(len(tiles))))
        rows = math.ceil(len(tiles) / columns)
        blank = np.zeros_like(tiles[0])
        grid_rows = []
        for row_idx in range(rows):
            row_tiles = tiles[row_idx * columns : (row_idx + 1) * columns]
            while len(row_tiles) < columns:
                row_tiles.append(blank.copy())
            grid_rows.append(np.concatenate(row_tiles, axis=1))
        return np.concatenate(grid_rows, axis=0)


def main() -> None:
    """Command-line entrypoint for classification training."""

    from trainers.classification_trainer import app

    app()
