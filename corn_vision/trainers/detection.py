"""Object detection trainer scaffold."""

from corn_vision.core.defs import OBJECT_DETECTION_TASK
from corn_vision.trainers.base import TrainerBase


class DetectionTrainer(TrainerBase):
    """Trainer for future object detection experiments."""

    task_name = OBJECT_DETECTION_TASK


def main() -> None:
    """Command-line entrypoint for object detection training."""

    raise NotImplementedError("Choose an object detection model before running this trainer.")
