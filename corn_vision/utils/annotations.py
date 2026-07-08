"""Annotation conversion and drawing helpers using OpenCV and Supervision."""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
from loguru import logger
import numpy as np
import supervision as sv

from corn_vision.utils.image import read_image_bgr, write_image


@dataclass(frozen=True)
class BoundingBoxAnnotation:
    """Single xyxy bounding box annotation."""

    image_name: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    class_name: str = "infected"
    confidence: float | None = None

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        x1, x2 = sorted((self.x_min, self.x_max))
        y1, y2 = sorted((self.y_min, self.y_max))
        return x1, y1, x2, y2

    @property
    def is_valid(self) -> bool:
        x1, y1, x2, y2 = self.xyxy
        return x2 > x1 and y2 > y1


def read_vott_csv(annotation_csv: str | Path) -> list[BoundingBoxAnnotation]:
    """Read the VoTT-style CSV exported with this dataset."""

    annotations: list[BoundingBoxAnnotation] = []
    with Path(annotation_csv).open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if not row:
                continue
            if row[0].lower() in {"id", "index"}:
                continue
            if len(row) < 7:
                logger.warning(f"Skipping malformed annotation row: {row}")
                continue
            try:
                annotations.append(
                    BoundingBoxAnnotation(
                        image_name=row[1],
                        x_min=float(row[2]),
                        y_min=float(row[3]),
                        x_max=float(row[4]),
                        y_max=float(row[5]),
                        class_name=row[6],
                    )
                )
            except ValueError:
                logger.warning(f"Skipping annotation row with invalid coordinates: {row}")
    logger.info(f"Loaded {len(annotations)} annotations from {annotation_csv}")
    return annotations


def group_annotations_by_image(
    annotations: list[BoundingBoxAnnotation],
) -> dict[str, list[BoundingBoxAnnotation]]:
    """Group annotations by image filename."""

    grouped: dict[str, list[BoundingBoxAnnotation]] = defaultdict(list)
    for annotation in annotations:
        grouped[annotation.image_name].append(annotation)
    return dict(grouped)


def annotations_to_detections(
    annotations: list[BoundingBoxAnnotation],
    class_name_to_id: dict[str, int] | None = None,
) -> tuple[sv.Detections, list[str]]:
    """Convert local annotations into a Supervision Detections object and labels."""

    if class_name_to_id is None:
        class_names = sorted({annotation.class_name for annotation in annotations})
        class_name_to_id = {class_name: idx for idx, class_name in enumerate(class_names)}

    xyxy = np.array([annotation.xyxy for annotation in annotations], dtype=np.float32)
    class_id = np.array(
        [class_name_to_id[annotation.class_name] for annotation in annotations],
        dtype=int,
    )
    confidence_values = [
        1.0 if annotation.confidence is None else annotation.confidence
        for annotation in annotations
    ]
    confidence = np.array(confidence_values, dtype=np.float32)
    labels = [
        f"{annotation.class_name} {score:.2f}"
        for annotation, score in zip(annotations, confidence_values)
    ]
    detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
    return detections, labels


def annotate_image(
    image: np.ndarray,
    annotations: list[BoundingBoxAnnotation],
    class_name_to_id: dict[str, int] | None = None,
    box_thickness: int = 2,
    text_scale: float = 0.5,
) -> np.ndarray:
    """Draw annotations on an image using Supervision annotators."""

    valid_annotations = [annotation for annotation in annotations if annotation.is_valid]
    skipped_count = len(annotations) - len(valid_annotations)
    if skipped_count:
        logger.warning(f"Skipping {skipped_count} invalid zero-area annotations")
    if not valid_annotations:
        return image.copy()
    detections, labels = annotations_to_detections(valid_annotations, class_name_to_id)
    annotated = image.copy()

    box_annotator_cls = getattr(sv, "BoundingBoxAnnotator", None) or getattr(
        sv,
        "BoxAnnotator",
    )
    box_annotator = box_annotator_cls(thickness=box_thickness)
    annotated = box_annotator.annotate(scene=annotated, detections=detections)

    label_annotator_cls = getattr(sv, "LabelAnnotator", None)
    if label_annotator_cls is not None:
        label_annotator = label_annotator_cls(text_scale=text_scale)
        annotated = label_annotator.annotate(
            scene=annotated,
            detections=detections,
            labels=labels,
        )
    else:
        annotated = box_annotator.annotate(
            scene=annotated,
            detections=detections,
            labels=labels,
        )
    return annotated


def annotate_image_file(
    image_path: str | Path,
    annotations: list[BoundingBoxAnnotation],
    output_path: str | Path | None = None,
    class_name_to_id: dict[str, int] | None = None,
) -> np.ndarray:
    """Read an image with OpenCV, draw annotations, and optionally write it."""

    image = read_image_bgr(image_path)
    annotated = annotate_image(image, annotations, class_name_to_id)
    if output_path is not None:
        write_image(output_path, annotated)
        logger.info(f"Wrote annotated image to {output_path}")
    return annotated


def annotate_vott_image(
    image_path: str | Path,
    annotation_csv: str | Path,
    output_path: str | Path | None = None,
) -> np.ndarray:
    """Annotate one dataset image from a VoTT CSV file."""

    annotations = read_vott_csv(annotation_csv)
    image_name = Path(image_path).name
    grouped = group_annotations_by_image(annotations)
    return annotate_image_file(
        image_path=image_path,
        annotations=grouped.get(image_name, []),
        output_path=output_path,
    )


def draw_cv2_rectangle(
    image: np.ndarray,
    annotation: BoundingBoxAnnotation,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a single rectangle with OpenCV for lightweight debugging."""

    x1, y1, x2, y2 = [int(value) for value in annotation.xyxy]
    annotated = image.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color=color, thickness=thickness)
    return annotated
