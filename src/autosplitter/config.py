"""Configuration loading and validation.

All tunables live in config.yaml (AGENTS.md non-negotiable). Validation must
fail with precise, path-style errors such as:

    config.yaml: splits[2].threshold must be between 0.0 and 1.0
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard, cast

import yaml


class ConfigError(Exception):
    """Raised when the configuration file is missing, malformed, or invalid."""


@dataclass(frozen=True)
class Roi:
    """Region of interest as normalized coordinates in [0.0, 1.0]."""

    x: float
    y: float
    width: float
    height: float


DEFAULT_ROI = Roi(x=0.25, y=0.25, width=0.50, height=0.50)


@dataclass(frozen=True)
class Capture:
    """Frame acquisition settings for the MPEG-TS sidecar stream.

    width/height describe the OBS sidecar output resolution (pinned at
    1920x1080 regardless of canvas size); the pipe reader needs them to
    assemble raw BGR frames.
    """

    fps: float
    stream_url: str
    width: int
    height: int


@dataclass(frozen=True)
class Color:
    name: str
    rgb: tuple[int, int, int]
    tolerance: float


@dataclass(frozen=True)
class Detection:
    required_frames: int
    reset_threshold: float


@dataclass(frozen=True)
class Split:
    id: int
    name: str
    color: str
    threshold: float


@dataclass(frozen=True)
class LiveSplit:
    host: str
    port: int


@dataclass(frozen=True)
class Config:
    capture: Capture
    roi: Roi
    detection: Detection
    colors: dict[str, Color]
    splits: tuple[Split, ...]
    livesplit: LiveSplit

    def color(self, name: str) -> Color:
        return self.colors[name]


def _is_number(value: object) -> TypeGuard[float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def load(path: str | Path = "config.yaml") -> Config:
    """Load and validate a YAML configuration file.

    Raises ConfigError with a precise message on any problem.
    """
    path = Path(path)
    label = str(path)

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"{label}: cannot read file ({exc.strerror})") from exc

    try:
        # PyYAML is untyped; cast pins the boundary so Unknown cannot spread.
        root = cast(object, yaml.safe_load(raw))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{label}: invalid YAML ({exc})") from exc

    if root is None:
        raise ConfigError(f"{label}: file is empty")
    if not isinstance(root, dict):
        raise ConfigError(f"{label}: top level must be a mapping")

    # Trust boundary: shape validated above; keys are only ever looked up by
    # the fixed section names below, so a str key assumption is safe.
    data = cast(dict[str, object], root)

    capture = _parse_capture(data.get("capture"), label)
    roi = _parse_roi(data.get("roi"), label)
    detection = _parse_detection(data.get("detection"), label)
    colors = _parse_colors(data.get("colors"), label)
    splits = _parse_splits(data.get("splits"), colors, label)
    livesplit = _parse_livesplit(data.get("livesplit"), label)

    return Config(
        capture=capture,
        roi=roi,
        detection=detection,
        colors=colors,
        splits=splits,
        livesplit=livesplit,
    )


def _require_section(value: object, key: str, label: str) -> dict[str, object]:
    if value is None:
        raise ConfigError(f"{label}: missing required section '{key}'")
    if not isinstance(value, dict):
        raise ConfigError(f"{label}: {key} must be a mapping")
    return cast(dict[str, object], value)


def _parse_capture(value: object, label: str) -> Capture:
    data = _require_section(value, "capture", label)

    fps = data.get("fps")
    if not _is_number(fps):
        raise ConfigError(f"{label}: capture.fps must be a number")
    if fps <= 0:
        raise ConfigError(f"{label}: capture.fps must be greater than 0")

    stream_url = data.get("stream_url")
    if not isinstance(stream_url, str) or not stream_url.strip():
        raise ConfigError(f"{label}: capture.stream_url must be a non-empty string")
    if not stream_url.startswith(("udp://", "tcp://")):
        raise ConfigError(
            f"{label}: capture.stream_url must start with udp:// or tcp://"
        )

    width = data.get("width")
    if isinstance(width, bool) or not isinstance(width, int):
        raise ConfigError(f"{label}: capture.width must be an integer")
    if width <= 0:
        raise ConfigError(f"{label}: capture.width must be a positive integer")

    height = data.get("height")
    if isinstance(height, bool) or not isinstance(height, int):
        raise ConfigError(f"{label}: capture.height must be an integer")
    if height <= 0:
        raise ConfigError(f"{label}: capture.height must be a positive integer")

    return Capture(fps=float(fps), stream_url=stream_url, width=width, height=height)


def _parse_roi(value: object, label: str) -> Roi:
    """Parse the roi section.

    Two equivalent forms are accepted (both use normalized 0.0-1.0 values):

        x, y, width, height          origin + size
        x1, y1, x2, y2               corner endpoints (top-left, bottom-right)

    Mixing forms is an error; an absent or empty section defaults to the
    middle 50% of the frame.
    """
    if value is None:
        return DEFAULT_ROI
    if not isinstance(value, dict):
        raise ConfigError(f"{label}: roi must be a mapping")
    data = cast(dict[str, object], value)

    uses_size = any(key in data for key in ("x", "y", "width", "height"))
    uses_endpoints = any(key in data for key in ("x1", "y1", "x2", "y2"))

    if uses_size and uses_endpoints:
        raise ConfigError(f"{label}: roi must use either x/y/width/height or the endpoints " + "x1/y1/x2/y2, not both")
    if not uses_size and not uses_endpoints:
        return DEFAULT_ROI

    fields: dict[str, float] = {}
    if uses_endpoints:
        points: dict[str, float] = {}
        for key in ("x1", "y1", "x2", "y2"):
            if key not in data:
                raise ConfigError(
                    f"{label}: roi.{key} is required when using endpoint form"
                )
            raw = data[key]
            if not _is_number(raw):
                raise ConfigError(f"{label}: roi.{key} must be a number")
            if not 0.0 <= raw <= 1.0:
                raise ConfigError(f"{label}: roi.{key} must be between 0.0 and 1.0")
            points[key] = float(raw)
        if points["x2"] < points["x1"]:
            raise ConfigError(f"{label}: roi.x2 must be greater than or equal to roi.x1")
        if points["y2"] < points["y1"]:
            raise ConfigError(f"{label}: roi.y2 must be greater than or equal to roi.y1")
        fields = {
            "x": points["x1"],
            "y": points["y1"],
            "width": points["x2"] - points["x1"],
            "height": points["y2"] - points["y1"],
        }
    else:
        for key in ("x", "y", "width", "height"):
            raw = data.get(key)
            if not _is_number(raw):
                raise ConfigError(f"{label}: roi.{key} must be a number")
            if not 0.0 <= raw <= 1.0:
                raise ConfigError(f"{label}: roi.{key} must be between 0.0 and 1.0")
            fields[key] = float(raw)

    if fields["width"] <= 0:
        raise ConfigError(f"{label}: roi.width must be greater than 0")
    if fields["height"] <= 0:
        raise ConfigError(f"{label}: roi.height must be greater than 0")
    if fields["x"] + fields["width"] > 1.0:
        raise ConfigError(f"{label}: roi.x + roi.width must not exceed 1.0")
    if fields["y"] + fields["height"] > 1.0:
        raise ConfigError(f"{label}: roi.y + roi.height must not exceed 1.0")

    return Roi(**fields)


def _parse_detection(value: object, label: str) -> Detection:
    data = _require_section(value, "detection", label)

    required_frames = data.get("required_frames")
    if isinstance(required_frames, bool) or not isinstance(required_frames, int):
        raise ConfigError(f"{label}: detection.required_frames must be an integer")
    if required_frames <= 0:
        raise ConfigError(
            f"{label}: detection.required_frames must be a positive integer"
        )

    reset_threshold = data.get("reset_threshold")
    if not _is_number(reset_threshold):
        raise ConfigError(f"{label}: detection.reset_threshold must be a number")
    if not 0.0 <= reset_threshold <= 1.0:
        raise ConfigError(
            f"{label}: detection.reset_threshold must be between 0.0 and 1.0"
        )

    return Detection(required_frames=required_frames, reset_threshold=float(reset_threshold))


def _coerce_rgb_list(value: object) -> list[int] | None:
    """Return the values if value is a list of exactly three ints, else None."""
    if not isinstance(value, list):
        return None
    items = cast(list[object], value)
    if len(items) != 3:
        return None
    result: list[int] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, int):
            return None
        result.append(item)
    return result


def _parse_colors(value: object, label: str) -> dict[str, Color]:
    data = _require_section(value, "colors", label)
    if not data:
        raise ConfigError(f"{label}: colors must define at least one color")

    colors: dict[str, Color] = {}
    for name, entry in data.items():
        where = f"colors.{name}"
        if not isinstance(entry, dict):
            raise ConfigError(f"{label}: {where} must be a mapping")
        spec = cast(dict[str, object], entry)

        rgb = _coerce_rgb_list(spec.get("rgb"))
        if rgb is None:
            raise ConfigError(
                f"{label}: {where}.rgb must be a list of three integers"
            )
        if any(not 0 <= c <= 255 for c in rgb):
            raise ConfigError(
                f"{label}: {where}.rgb values must be between 0 and 255"
            )

        tolerance = spec.get("tolerance")
        if not _is_number(tolerance):
            raise ConfigError(f"{label}: {where}.tolerance must be a number")
        if tolerance < 0:
            raise ConfigError(
                f"{label}: {where}.tolerance must be a non-negative number"
            )

        colors[str(name)] = Color(
            name=str(name), rgb=(rgb[0], rgb[1], rgb[2]), tolerance=float(tolerance)
        )
    return colors


def _parse_splits(
    value: object, colors: dict[str, Color], label: str
) -> tuple[Split, ...]:
    if value is None:
        raise ConfigError(f"{label}: missing required section 'splits'")
    if not isinstance(value, list):
        raise ConfigError(f"{label}: splits must be a list")
    entries = cast(list[object], value)
    if not entries:
        raise ConfigError(f"{label}: splits must contain at least one split")

    seen_ids: set[int] = set()
    splits: list[Split] = []
    for index, entry in enumerate(entries):
        where = f"splits[{index}]"
        if not isinstance(entry, dict):
            raise ConfigError(f"{label}: {where} must be a mapping")
        spec = cast(dict[str, object], entry)

        split_id = spec.get("id")
        if isinstance(split_id, bool) or not isinstance(split_id, int):
            raise ConfigError(f"{label}: {where}.id must be an integer")
        if split_id in seen_ids:
            raise ConfigError(
                f"{label}: {where}.id must be unique ({split_id} already used)"
            )
        seen_ids.add(split_id)

        name = spec.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"{label}: {where}.name must be a non-empty string")

        color_name = spec.get("color")
        if not isinstance(color_name, str) or color_name not in colors:
            known = ", ".join(sorted(colors)) or "<none defined>"
            raise ConfigError(
                f"{label}: {where}.color must be one of the defined colors: {known}"
            )

        threshold = spec.get("threshold")
        if not _is_number(threshold):
            raise ConfigError(f"{label}: {where}.threshold must be a number")
        if not 0.0 <= threshold <= 1.0:
            raise ConfigError(f"{label}: {where}.threshold must be between 0.0 and 1.0")

        splits.append(
            Split(
                id=split_id,
                name=name,
                color=color_name,
                threshold=float(threshold),
            )
        )
    return tuple(splits)


def _parse_livesplit(value: object, label: str) -> LiveSplit:
    data = _require_section(value, "livesplit", label)

    host = data.get("host")
    if not isinstance(host, str) or not host.strip():
        raise ConfigError(f"{label}: livesplit.host must be a non-empty string")

    port = data.get("port")
    if isinstance(port, bool) or not isinstance(port, int):
        raise ConfigError(f"{label}: livesplit.port must be an integer")
    if not 1 <= port <= 65535:
        raise ConfigError(f"{label}: livesplit.port must be between 1 and 65535")

    return LiveSplit(host=host, port=port)
