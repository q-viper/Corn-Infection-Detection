"""Base model helpers."""

import torch
from torch import nn


class BaseVisionModel(nn.Module):
    """Common parent for vision models."""

    task_name: str

    def forward_logits(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward_logits.")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_logits(inputs)

    def predict_proba(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.forward_logits(inputs)
        return torch.softmax(logits, dim=1)

    def set_backbone_trainable(self, trainable: bool) -> None:
        raise NotImplementedError("Subclasses must implement set_backbone_trainable.")


class BaseClassificationModel(BaseVisionModel):
    """Base class for classification models with explainability hooks."""

    def generate_saliency_map(
        self,
        inputs: torch.Tensor,
        target_class: int | torch.Tensor | None = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement generate_saliency_map.")

    def generate_gradcam(
        self,
        inputs: torch.Tensor,
        target_class: int | torch.Tensor | None = None,
        target_layer: str | None = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement generate_gradcam.")
