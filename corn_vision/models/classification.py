"""Torchvision binary classification models."""

from collections.abc import Callable

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

from corn_vision.core.configs import ClassificationModelConfig
from corn_vision.core.defs import TaskType, TorchVisionBackbone
from corn_vision.models.base import BaseClassificationModel


_BACKBONE_BUILDERS: dict[TorchVisionBackbone, str] = {
    TorchVisionBackbone.RESNET18: "resnet18",
    TorchVisionBackbone.RESNET34: "resnet34",
    TorchVisionBackbone.RESNET50: "resnet50",
    TorchVisionBackbone.EFFICIENTNET_B0: "efficientnet_b0",
    TorchVisionBackbone.EFFICIENTNET_B2: "efficientnet_b2",
    TorchVisionBackbone.MOBILENET_V3_LARGE: "mobilenet_v3_large",
    TorchVisionBackbone.CONVNEXT_TINY: "convnext_tiny",
    TorchVisionBackbone.DENSENET121: "densenet121",
    TorchVisionBackbone.REGNET_Y_400MF: "regnet_y_400mf",
}

_WEIGHT_ENUMS: dict[TorchVisionBackbone, str] = {
    TorchVisionBackbone.RESNET18: "ResNet18_Weights",
    TorchVisionBackbone.RESNET34: "ResNet34_Weights",
    TorchVisionBackbone.RESNET50: "ResNet50_Weights",
    TorchVisionBackbone.EFFICIENTNET_B0: "EfficientNet_B0_Weights",
    TorchVisionBackbone.EFFICIENTNET_B2: "EfficientNet_B2_Weights",
    TorchVisionBackbone.MOBILENET_V3_LARGE: "MobileNet_V3_Large_Weights",
    TorchVisionBackbone.CONVNEXT_TINY: "ConvNeXt_Tiny_Weights",
    TorchVisionBackbone.DENSENET121: "DenseNet121_Weights",
    TorchVisionBackbone.REGNET_Y_400MF: "RegNet_Y_400MF_Weights",
}

_DEFAULT_GRADCAM_LAYERS: dict[TorchVisionBackbone, str] = {
    TorchVisionBackbone.RESNET18: "layer4",
    TorchVisionBackbone.RESNET34: "layer4",
    TorchVisionBackbone.RESNET50: "layer4",
    TorchVisionBackbone.EFFICIENTNET_B0: "features",
    TorchVisionBackbone.EFFICIENTNET_B2: "features",
    TorchVisionBackbone.MOBILENET_V3_LARGE: "features",
    TorchVisionBackbone.CONVNEXT_TINY: "features",
    TorchVisionBackbone.DENSENET121: "features",
    TorchVisionBackbone.REGNET_Y_400MF: "trunk_output",
}


def _make_head(in_features: int, out_features: int, dropout: float) -> nn.Module:
    if dropout <= 0:
        return nn.Linear(in_features, out_features)
    return nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, out_features))


def _get_torchvision_weights(backbone: TorchVisionBackbone, pretrained: bool):
    if not pretrained:
        return None
    weights_enum = getattr(models, _WEIGHT_ENUMS[backbone], None)
    if weights_enum is None:
        return None
    return weights_enum.DEFAULT


def _build_torchvision_model(backbone: TorchVisionBackbone, pretrained: bool) -> nn.Module:
    builder_name = _BACKBONE_BUILDERS[backbone]
    builder: Callable = getattr(models, builder_name)
    weights = _get_torchvision_weights(backbone, pretrained)
    try:
        return builder(weights=weights)
    except TypeError:
        return builder(pretrained=pretrained)


class TorchVisionBinaryClassifier(BaseClassificationModel):
    """Binary classifier with a Torchvision backbone and explainability methods."""

    task_name = TaskType.CLASSIFICATION.value

    def __init__(self, config: ClassificationModelConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone_name = config.backbone
        self.model = _build_torchvision_model(config.backbone, config.pretrained)
        self._replace_classifier_head()
        self.set_backbone_trainable(config.train_backbone)

    def _replace_classifier_head(self) -> None:
        out_features = self.config.num_classes
        dropout = self.config.dropout
        if hasattr(self.model, "fc") and isinstance(self.model.fc, nn.Linear):
            in_features = self.model.fc.in_features
            self.model.fc = _make_head(in_features, out_features, dropout)
            self.classifier_module = self.model.fc
        elif hasattr(self.model, "classifier") and isinstance(self.model.classifier, nn.Linear):
            in_features = self.model.classifier.in_features
            self.model.classifier = _make_head(in_features, out_features, dropout)
            self.classifier_module = self.model.classifier
        elif hasattr(self.model, "classifier") and isinstance(self.model.classifier, nn.Sequential):
            last_linear_idx = None
            for idx in reversed(range(len(self.model.classifier))):
                if isinstance(self.model.classifier[idx], nn.Linear):
                    last_linear_idx = idx
                    break
            if last_linear_idx is None:
                raise ValueError(f"Could not find classifier head for {self.backbone_name}.")
            in_features = self.model.classifier[last_linear_idx].in_features
            self.model.classifier[last_linear_idx] = _make_head(
                in_features,
                out_features,
                dropout,
            )
            self.classifier_module = self.model.classifier[last_linear_idx]
        elif hasattr(self.model, "heads") and hasattr(self.model.heads, "head"):
            in_features = self.model.heads.head.in_features
            self.model.heads.head = _make_head(in_features, out_features, dropout)
            self.classifier_module = self.model.heads.head
        else:
            raise ValueError(f"Unsupported classifier head for {self.backbone_name}.")

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = trainable
        for parameter in self.classifier_module.parameters():
            parameter.requires_grad = True

    def forward_logits(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _target_scores(
        self,
        logits: torch.Tensor,
        target_class: int | torch.Tensor | None,
    ) -> torch.Tensor:
        if target_class is None:
            target = logits.argmax(dim=1)
        elif isinstance(target_class, int):
            target = torch.full(
                (logits.shape[0],),
                target_class,
                dtype=torch.long,
                device=logits.device,
            )
        else:
            target = target_class.to(logits.device).long()
        return logits.gather(1, target.view(-1, 1)).sum()

    @staticmethod
    def _normalize_maps(values: torch.Tensor) -> torch.Tensor:
        flattened = values.flatten(start_dim=1)
        mins = flattened.min(dim=1).values.view(-1, 1, 1)
        maxs = flattened.max(dim=1).values.view(-1, 1, 1)
        return (values - mins) / (maxs - mins + 1e-8)

    def generate_saliency_map(
        self,
        inputs: torch.Tensor,
        target_class: int | torch.Tensor | None = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        saliency_input = inputs.detach().clone().requires_grad_(True)
        self.zero_grad(set_to_none=True)
        logits = self.forward_logits(saliency_input)
        score = self._target_scores(logits, target_class)
        score.backward()
        saliency = saliency_input.grad.detach().abs().amax(dim=1)
        if normalize:
            saliency = self._normalize_maps(saliency)
        if was_training:
            self.train()
        return saliency

    def generate_gradcam(
        self,
        inputs: torch.Tensor,
        target_class: int | torch.Tensor | None = None,
        target_layer: str | None = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        layer_name = (
            target_layer
            or self.config.gradcam_target_layer
            or _DEFAULT_GRADCAM_LAYERS[self.backbone_name]
        )
        layer = self.model.get_submodule(layer_name)
        activations: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []

        def forward_hook(_module, _inputs, output):
            activations.append(output)

        def backward_hook(_module, _grad_input, grad_output):
            gradients.append(grad_output[0])

        was_training = self.training
        self.eval()
        forward_handle = layer.register_forward_hook(forward_hook)
        backward_handle = layer.register_full_backward_hook(backward_hook)
        try:
            gradcam_input = inputs.detach().clone().requires_grad_(True)
            self.zero_grad(set_to_none=True)
            logits = self.forward_logits(gradcam_input)
            score = self._target_scores(logits, target_class)
            score.backward()
            if not activations or not gradients:
                raise RuntimeError(f"Grad-CAM hooks did not capture layer: {layer_name}")
            activation = activations[-1]
            gradient = gradients[-1]
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * activation).sum(dim=1))
            cam = F.interpolate(
                cam.unsqueeze(1),
                size=gradcam_input.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            if normalize:
                cam = self._normalize_maps(cam)
            return cam.detach()
        finally:
            forward_handle.remove()
            backward_handle.remove()
            if was_training:
                self.train()
