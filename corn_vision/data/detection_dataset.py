"""Object detection dataset scaffold."""

from corn_vision.core.defs import OBJECT_DETECTION_TASK
from corn_vision.data.base import CornDatasetBase


class CornDetectionDataset(CornDatasetBase):
    """Dataset for corn leaf infection bounding box detection."""

    task_name = OBJECT_DETECTION_TASK
