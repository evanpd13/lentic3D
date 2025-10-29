#!/usr/bin/env python3
"""Command-line interface for generating lenticular-ready interlaced images."""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_IMAGE = SCRIPT_DIR / "examplePhoto.JPG"
DEFAULT_MIDAS_MODEL = SCRIPT_DIR / "midas_v21_small_256.onnx"
DEFAULT_DEPTH_ANYTHING_MODEL = SCRIPT_DIR / "depth_anything_v2_small.pth"


def _as_existing(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Expected {description} at {path}")
    return path


@dataclass
class PipelineMetadata:
    """Metadata saved with each pipeline run."""

    source_image: str
    bounding_box: Tuple[int, int, int, int]
    radius: float
    knob_degrees: float
    lenticule_width: int
    models: Dict[str, str] = field(default_factory=dict)

    def to_json(self) -> Dict[str, object]:
        data = asdict(self)
        data["bounding_box"] = list(self.bounding_box)
        return data

    @classmethod
    def from_json(cls, data: Dict[str, object]) -> "PipelineMetadata":
        return cls(
            source_image=data.get("source_image", ""),
            bounding_box=tuple(int(v) for v in data.get("bounding_box", (0, 0, 0, 0))),
            radius=float(data.get("radius", 0.0)),
            knob_degrees=float(data.get("knob_degrees", 0.0)),
            lenticule_width=int(data.get("lenticule_width", 1)),
            models={k: str(v) for k, v in data.get("models", {}).items()},
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a lenticular interlaced print from a single source image. "
            "The pipeline estimates a subject bounding box to derive a reusable depth radius, "
            "creates rotated views, and interlaces them into a printable composite."
        )
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=str(DEFAULT_SOURCE_IMAGE),
        help=(
            "Path to the source image (RGB/BGR). If omitted the bundled example photo "
            "located next to this script will be used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "outputs"),
        help="Directory where generated images and metadata will be stored.",
    )
    parser.add_argument(
        "--knob",
        type=float,
        default=5.0,
        help="Angular knob in degrees that controls the left/right rotation magnitude.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Optional override for the depth radius. When omitted the pipeline derives it from the subject bounding box.",
    )
    parser.add_argument(
        "--lenticule-width",
        type=int,
        default=4,
        help="Width in pixels of each lenticule strip in the interlaced composite.",
    )
    parser.add_argument(
        "--metadata-json",
        default=None,
        help="Optional path to write metadata JSON. Defaults to <output-dir>/metadata.json.",
    )
    parser.add_argument(
        "--midas-model",
        default=str(DEFAULT_MIDAS_MODEL),
        help=(
            "Path to the MiDaS model file. Defaults to the bundled ONNX model in the "
            "same directory as this script."
        ),
    )
    parser.add_argument(
        "--depth-anything-model",
        default=str(DEFAULT_DEPTH_ANYTHING_MODEL),
        help=(
            "Path to the Depth Anything v2 small checkpoint. Defaults to the bundled "
            "PyTorch model next to this script."
        ),
    )
    return parser.parse_args()


def ensure_output_dir(path: str) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_source_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path

    fallback = SCRIPT_DIR / path_str
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Unable to locate source image at {path_str}")


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image at {path}")
    return image


def find_subject_bounding_box(image: np.ndarray) -> Tuple[int, int, int, int]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        height, width = gray.shape
        return 0, 0, width, height

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return x, y, w, h


def derive_radius(bounding_box: Tuple[int, int, int, int]) -> float:
    _, _, w, h = bounding_box
    major_axis = max(w, h)
    return max(major_axis / 2.0, 1.0)


def load_metadata(metadata_path: Path) -> Optional[PipelineMetadata]:
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        try:
            return PipelineMetadata.from_json(data)
        except Exception:
            return None
    return None


def save_metadata(metadata_path: Path, metadata: PipelineMetadata) -> None:
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata.to_json(), fh, indent=2)


def rotation_matrix_y(angle_degrees: float) -> np.ndarray:
    angle_radians = math.radians(angle_degrees)
    cos_a = math.cos(angle_radians)
    sin_a = math.sin(angle_radians)
    return np.array(
        [
            [cos_a, 0.0, sin_a],
            [0.0, 1.0, 0.0],
            [-sin_a, 0.0, cos_a],
        ],
        dtype=np.float32,
    )


def warp_view(image: np.ndarray, angle_degrees: float, radius: float) -> np.ndarray:
    height, width = image.shape[:2]
    if radius <= 0:
        raise ValueError("Radius must be positive for perspective warping.")

    cx, cy = width / 2.0, height / 2.0
    focal = radius

    k = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float32)
    k_inv = np.linalg.inv(k)
    rotation = rotation_matrix_y(angle_degrees)

    transform = k @ rotation @ k_inv
    warped = cv2.warpPerspective(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped


def interlace_views(views: Tuple[np.ndarray, np.ndarray, np.ndarray], lenticule_width: int) -> np.ndarray:
    if lenticule_width <= 0:
        raise ValueError("Lenticule width must be a positive integer.")

    center = views[1]
    height, width = center.shape[:2]
    canvas = np.zeros_like(center)
    strip_sources = (views[0], views[1], views[2])

    for i, start in enumerate(range(0, width, lenticule_width)):
        end = min(start + lenticule_width, width)
        source = strip_sources[i % 3]
        canvas[:, start:end] = source[:, start:end]

    return canvas


def main() -> None:
    args = parse_args()
    midas_model = _as_existing(Path(args.midas_model), "MiDaS model")
    depth_anything_model = _as_existing(Path(args.depth_anything_model), "Depth Anything model")

    source_path = resolve_source_path(args.source)
    image = load_image(source_path)

    output_dir = ensure_output_dir(args.output_dir)
    metadata_path = Path(args.metadata_json) if args.metadata_json else output_dir / "metadata.json"

    existing_metadata = load_metadata(metadata_path)

    bounding_box = find_subject_bounding_box(image)
    derived_radius = derive_radius(bounding_box)

    if args.radius is not None:
        radius = float(args.radius)
    elif existing_metadata and existing_metadata.radius > 0:
        radius = existing_metadata.radius
    else:
        radius = derived_radius

    center_view = image
    left_view = warp_view(center_view, -abs(args.knob), radius)
    right_view = warp_view(center_view, abs(args.knob), radius)

    composite = interlace_views((left_view, center_view, right_view), args.lenticule_width)

    output_dir.mkdir(parents=True, exist_ok=True)

    source_name = source_path.stem
    cv2.imwrite(str(output_dir / f"{source_name}_left.png"), left_view)
    cv2.imwrite(str(output_dir / f"{source_name}_center.png"), center_view)
    cv2.imwrite(str(output_dir / f"{source_name}_right.png"), right_view)
    cv2.imwrite(str(output_dir / f"{source_name}_lenticular.png"), composite)

    metadata = PipelineMetadata(
        source_image=os.path.abspath(str(source_path)),
        bounding_box=bounding_box,
        radius=radius,
        knob_degrees=args.knob,
        lenticule_width=args.lenticule_width,
        models={
            "midas": str(midas_model),
            "depth_anything_v2_small": str(depth_anything_model),
        },
    )
    save_metadata(metadata_path, metadata)


if __name__ == "__main__":
    main()
