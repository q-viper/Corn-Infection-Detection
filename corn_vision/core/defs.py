"""Shared enums for corn vision experiments."""

from enum import Enum


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    OBJECT_DETECTION = "object_detection"


class DataType(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

    def __str__(self) -> str:
        return self.value


class OptimizerType(str, Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"


class MetricType(str, Enum):
    LOSS = "loss"
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"


class TorchVisionBackbone(str, Enum):
    RESNET18 = "resnet18"
    RESNET34 = "resnet34"
    RESNET50 = "resnet50"
    EFFICIENTNET_B0 = "efficientnet_b0"
    EFFICIENTNET_B2 = "efficientnet_b2"
    MOBILENET_V3_LARGE = "mobilenet_v3_large"
    CONVNEXT_TINY = "convnext_tiny"
    DENSENET121 = "densenet121"
    REGNET_Y_400MF = "regnet_y_400mf"


class SaliencyMethod(str, Enum):
    SALIENCY = "saliency"
    GRADCAM = "gradcam"


CLASSIFICATION_TASK = TaskType.CLASSIFICATION.value
OBJECT_DETECTION_TASK = TaskType.OBJECT_DETECTION.value
