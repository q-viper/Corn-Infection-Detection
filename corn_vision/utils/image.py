"""Image loading and writing helpers backed by OpenCV."""

from pathlib import Path

import cv2
import numpy as np


def read_image_bgr(image_path: str | Path) -> np.ndarray:
    """Read an image as a BGR OpenCV array."""

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def write_image(image_path: str | Path, image: np.ndarray) -> None:
    """Write an image array to disk, creating parent directories if needed."""

    output_path = Path(image_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Could not write image: {output_path}")


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR image to RGB."""

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to OpenCV BGR."""

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
