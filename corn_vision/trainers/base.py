"""Base trainer abstractions."""

from collections import defaultdict
import csv
import random

import torch
from torch import nn
from tqdm import tqdm

from corn_vision.core.configs import NNTrainerConfig
from corn_vision.core.defs import MetricType, OptimizerType
from corn_vision.data.base import CornDatasetBase
from corn_vision.utils.logging import setup_logger


class TrainerBase:
    """Common neural network trainer."""

    task_name: str

    def __init__(
        self,
        config: NNTrainerConfig,
        model: nn.Module,
        train_dataset: CornDatasetBase,
        val_dataset: CornDatasetBase,
        criterion: nn.Module | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config.run_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(
            log_dir=self.config.run_dir,
            log_file=self.config.log_file,
            level=self.config.log_level,
        )
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = criterion or nn.CrossEntropyLoss()
        if config.weighted_loss and isinstance(self.criterion, nn.CrossEntropyLoss):
            self.criterion = nn.CrossEntropyLoss(
                weight=self.train_dataset.class_weights.to(self.device)
            )
        self.optimizer = self._make_optimizer()
        self.metric_history: dict[str, list[float]] = defaultdict(list)
        self.patience_counter = 0
        self.epoch = 0
        self._set_seed(config.random_seed)
        self._write_configs()
        self.make_dataloaders()
        self.logger.info(f"Trainer initialized on device: {self.device}")
        self.logger.info(f"Run directory: {self.config.run_dir}")
        self.logger.info(
            f"Training samples: {len(self.train_dataset)}, validation samples: {len(self.val_dataset)}"
        )

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _write_configs(self) -> None:
        (self.config.run_dir / "trainer_config.json").write_text(
            self.config.model_dump_json(indent=2),
            encoding="utf-8",
        )
        data_config = getattr(self.train_dataset, "config", None)
        if data_config is not None:
            (self.config.run_dir / "dataset_config.json").write_text(
                data_config.model_dump_json(indent=2),
                encoding="utf-8",
            )
        model_config = getattr(self.model, "config", None)
        if model_config is not None:
            (self.config.run_dir / "model_config.json").write_text(
                model_config.model_dump_json(indent=2),
                encoding="utf-8",
            )

    def _make_optimizer(self) -> torch.optim.Optimizer:
        parameters = [param for param in self.model.parameters() if param.requires_grad]
        if self.config.optimizer == OptimizerType.ADAM:
            return torch.optim.Adam(
                parameters,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        if self.config.optimizer == OptimizerType.ADAMW:
            return torch.optim.AdamW(
                parameters,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        if self.config.optimizer == OptimizerType.SGD:
            return torch.optim.SGD(
                parameters,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )
        if self.config.optimizer == OptimizerType.RMSPROP:
            return torch.optim.RMSprop(
                parameters,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

    def make_dataloaders(self) -> None:
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.number_of_workers,
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.number_of_workers,
        )

    def forward_step(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        logits = self.model(images)
        loss = self.criterion(logits, labels)
        predictions = logits.argmax(dim=1)
        return logits, loss, predictions

    def run_epoch(self, dataloader, is_train: bool) -> dict[str, float]:
        self.model.train(is_train)
        total_loss = 0.0
        total_samples = 0
        correct = 0
        true_positive = 0
        false_positive = 0
        false_negative = 0
        progress = tqdm(
            dataloader,
            desc=f"{'Train' if is_train else 'Validation'} {self.epoch + 1}/{self.config.epochs}",
            unit="batch",
        )
        for batch_idx, batch in enumerate(progress, start=1):
            if is_train:
                self.optimizer.zero_grad(set_to_none=True)
            _logits, loss, predictions = self.forward_step(batch)
            labels = batch[1].to(self.device)
            if is_train:
                loss.backward()
                self.at_batch_end()
                self.optimizer.step()
            total_loss += loss.item()
            correct += (predictions == labels).sum().item()
            total_samples += labels.numel()
            true_positive += ((predictions == 1) & (labels == 1)).sum().item()
            false_positive += ((predictions == 1) & (labels == 0)).sum().item()
            false_negative += ((predictions == 0) & (labels == 1)).sum().item()
            running_metrics = self.compute_metrics_from_counts(
                correct=correct,
                total=total_samples,
                true_positive=true_positive,
                false_positive=false_positive,
                false_negative=false_negative,
            )
            running_metrics[MetricType.LOSS.value] = total_loss / batch_idx
            progress.set_postfix(
                {
                    key: f"{value:.4f}"
                    for key, value in running_metrics.items()
                    if key == MetricType.LOSS.value or MetricType(key) in self.config.metrics
                }
            )
        if total_samples == 0:
            metrics = {
                metric.value: 0.0
                for metric in (
                    MetricType.ACCURACY,
                    MetricType.PRECISION,
                    MetricType.RECALL,
                    MetricType.F1_SCORE,
                )
            }
            metrics[MetricType.LOSS.value] = 0.0
            return metrics
        metrics = self.compute_metrics_from_counts(
            correct=correct,
            total=total_samples,
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
        )
        metrics[MetricType.LOSS.value] = total_loss / max(len(dataloader), 1)
        return metrics

    def compute_metrics(self, predictions: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        correct = (predictions == labels).sum().item()
        total = max(labels.numel(), 1)
        true_positive = ((predictions == 1) & (labels == 1)).sum().item()
        false_positive = ((predictions == 1) & (labels == 0)).sum().item()
        false_negative = ((predictions == 0) & (labels == 1)).sum().item()
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1_score = 2 * precision * recall / max(precision + recall, 1e-8)
        return {
            MetricType.ACCURACY.value: correct / total,
            MetricType.PRECISION.value: precision,
            MetricType.RECALL.value: recall,
            MetricType.F1_SCORE.value: f1_score,
        }

    def compute_metrics_from_counts(
        self,
        correct: int,
        total: int,
        true_positive: int,
        false_positive: int,
        false_negative: int,
    ) -> dict[str, float]:
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1_score = 2 * precision * recall / max(precision + recall, 1e-8)
        return {
            MetricType.ACCURACY.value: correct / max(total, 1),
            MetricType.PRECISION.value: precision,
            MetricType.RECALL.value: recall,
            MetricType.F1_SCORE.value: f1_score,
        }

    def at_batch_end(self) -> None:
        pass

    def at_train_start(self) -> None:
        pass

    def at_epoch_end(self) -> None:
        pass

    def train(self) -> str:
        self.logger.info("Starting training")
        self.at_train_start()
        best_value = float("-inf") if self.config.best_model_metric_greater else float("inf")
        best_model_path = self.config.run_dir / self.config.best_model_name
        metric_file = self.config.run_dir / self.config.metric_file
        for epoch in range(self.config.epochs):
            self.epoch = epoch
            self.logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")
            train_metrics = self.run_epoch(self.train_loader, is_train=True)
            with torch.no_grad():
                val_metrics = self.run_epoch(self.val_loader, is_train=False)
            self.at_epoch_end()
            self._append_metrics(metric_file, epoch, train_metrics, val_metrics)
            self.logger.info(
                "Train metrics: "
                + ", ".join(f"{key}={value:.4f}" for key, value in train_metrics.items())
            )
            self.logger.info(
                "Validation metrics: "
                + ", ".join(f"{key}={value:.4f}" for key, value in val_metrics.items())
            )
            for key, value in train_metrics.items():
                self.metric_history[key].append(value)
            for key, value in val_metrics.items():
                self.metric_history[f"val_{key}"].append(value)
            metric_name = self.config.best_model_metric.value
            current_value = val_metrics[metric_name]
            improved = (
                current_value > best_value
                if self.config.best_model_metric_greater
                else current_value < best_value
            )
            if improved:
                best_value = current_value
                torch.save(self.model.state_dict(), best_model_path)
                best_full_model_path = best_model_path.with_name(
                    best_model_path.stem + "_full" + best_model_path.suffix
                )
                torch.save(self.model, best_full_model_path)
                self.patience_counter = 0
                self.logger.info(f"Saved best model to {best_model_path}")
                self.logger.info(f"Saved best full model to {best_full_model_path}")
            else:
                self.patience_counter += 1
                self.logger.info(
                    f"No improvement in {metric_name}; patience "
                    f"{self.patience_counter}/{self.config.early_stopping_patience}"
                )
            last_state_path = self.config.run_dir / "last_model.pth"
            last_full_model_path = self.config.run_dir / "last_model_full.pth"
            torch.save(self.model.state_dict(), last_state_path)
            torch.save(self.model, last_full_model_path)
            self.logger.info(f"Saved last model state to {last_state_path}")
            self.logger.info(f"Saved last full model to {last_full_model_path}")
            if self.patience_counter >= self.config.early_stopping_patience:
                self.logger.info("Early stopping triggered")
                break
        self.logger.info("Training completed")
        return str(best_model_path)

    def _append_metrics(
        self,
        metric_file,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> None:
        fieldnames = ["epoch"]
        fieldnames.extend(f"train_{key}" for key in train_metrics)
        fieldnames.extend(f"val_{key}" for key in val_metrics)
        row = {"epoch": epoch + 1}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        write_header = not metric_file.exists()
        with metric_file.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
