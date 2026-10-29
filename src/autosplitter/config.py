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


@dataclass(frozen=True)
class StartTransition:
    """Optional start-of-run transition (menu → gameplay).

    If present, the state machine waits for this color before entering
    the normal split sequence. Uses the main ROI unless roi_override is set.
    """

    name: str
    color: str
    threshold: float
    roi_override: Roi | None = None
    # If true, send `start` command to LiveSplit Server when detected.
    auto_start_timer: bool = False


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
    # --loop: quiet period (seconds) on the next run's first color before
    # the machine rearms after DONE.
    restart_quiet_seconds: float = 3.0


@dataclass(frozen=True)
class Split:
    id: int
    name: str
    color: str
    threshold: float


@dataclass(frozen=True)
class LiveSplit:
    """Connection target for LiveSplit Server's TCP command interface.

    ``timeout_seconds`` bounds any single socket operation (a hung server
    must never stall the detection loop); ``reconnect_seconds`` rate-limits
    reconnect attempts while the server is unreachable.
    """

    host: str
    port: int
    timeout_seconds: float = 0.25
    reconnect_seconds: float = 5.0
    resync_on_unsplit: bool = False


@dataclass(frozen=True)
class Record:
    """Recording settings: path template and optional post-run conversion.

    ``path_template`` supports variables: {game}, {date}, {time}, {datetime},
    {attempt}.  {attempt} is auto-incremented from existing files.
    ``convert_to`` if set (e.g. "mp4") triggers a codec-copy conversion
    after the run; the original .mkv is never deleted.
    ``post_run_seconds`` keeps recording for this many seconds after the
    last split fires (captures victory screen / final time display).
    """

    path_template: str
    convert_to: str | None = None
    post_run_seconds: float = 3.0


@dataclass(frozen=True)
class OcrEngine:
    """Tesseract engine settings for OCR-based split detection."""

    language: str = "eng"
    psm: int = 7
    min_confidence: float = 60.0


@dataclass(frozen=True)
class OcrPattern:
    """A regex pattern for matching OCR text to a split index.

    If split_index is None, the first digit group in the match is used
    as a 1-based split id (e.g. "Level 3" → split index 2).
    """

    regex: str
    split_index: int | None = None


@dataclass(frozen=True)
class OcrConfig:
    """OCR-based split number detection (augments color detection).

    Reads text from a separate ROI and re-syncs the split index when
    the recognized number diverges from the color-driven index.
    """

    roi: Roi
    engine: OcrEngine = OcrEngine()
    patterns: tuple[OcrPattern, ...] = ()
    poll_interval_frames: int = 3


@dataclass(frozen=True)
class Config:
    capture: Capture
    roi: Roi
    detection: Detection
    colors: dict[str, Color]
    splits: tuple[Split, ...]
    livesplit: LiveSplit
    start_transition: StartTransition | None = None
    record: Record | None = None
    ocr: OcrConfig | None = None

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
    start_transition = _parse_start_transition(data.get("start_transition"), colors, label)
    record = _parse_record(data.get("record"), label)
    ocr = _parse_ocr(data.get("ocr"), label)

    return Config(
        capture=capture,
        roi=roi,
        detection=detection,
        colors=colors,
        splits=splits,
        livesplit=livesplit,
        start_transition=start_transition,
        record=record,
        ocr=ocr,
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
    if not stream_url.startswith(("udp://", "tcp://", "srt://")):
        raise ConfigError(
            f"{label}: capture.stream_url must start with udp://, tcp:// or srt://"
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

    restart_quiet = data.get("restart_quiet_seconds", 3.0)
    if not _is_number(restart_quiet):
        raise ConfigError(f"{label}: detection.restart_quiet_seconds must be a number")
    if restart_quiet < 0.0:
        raise ConfigError(
            f"{label}: detection.restart_quiet_seconds must not be negative"
        )

    return Detection(
        required_frames=required_frames,
        reset_threshold=float(reset_threshold),
        restart_quiet_seconds=float(restart_quiet),
    )


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

    # Socket-op budget and reconnect cadence: optional, mirroring the ROI
    # section's fallback behavior for minimal configs.
    timeout_value = data.get("timeout_seconds")
    if timeout_value is not None:
        if not _is_number(timeout_value):
            raise ConfigError(f"{label}: livesplit.timeout_seconds must be a number")
        if timeout_value <= 0:
            raise ConfigError(
                f"{label}: livesplit.timeout_seconds must be greater than 0"
            )

    reconnect_value = data.get("reconnect_seconds")
    if reconnect_value is not None:
        if not _is_number(reconnect_value):
            raise ConfigError(f"{label}: livesplit.reconnect_seconds must be a number")
        if reconnect_value <= 0:
            raise ConfigError(
                f"{label}: livesplit.reconnect_seconds must be greater than 0"
            )

    resync_value = data.get("resync_on_unsplit")
    if resync_value is not None and not isinstance(resync_value, bool):
        raise ConfigError(f"{label}: livesplit.resync_on_unsplit must be a boolean")

    return LiveSplit(
        host=host,
        port=port,
        timeout_seconds=float(timeout_value) if timeout_value is not None else 0.25,
        reconnect_seconds=(
            float(reconnect_value) if reconnect_value is not None else 5.0
        ),
        resync_on_unsplit=resync_value is True,
    )


def _parse_start_transition(
    value: object, colors: dict[str, Color], label: str
) -> StartTransition | None:
    """Parse optional start_transition section.

    Returns None if the section is absent or empty (start transition disabled).
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ConfigError(f"{label}: start_transition must be a mapping")
    data = cast(dict[str, object], value)
    if not data:
        return None

    color_name = data.get("color")
    if not isinstance(color_name, str) or not color_name.strip():
        raise ConfigError(f"{label}: start_transition.color must be a non-empty string")
    if color_name not in colors:
        raise ConfigError(
            f"{label}: start_transition.color '{color_name}' not defined in colors"
        )

    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(f"{label}: start_transition.name must be a non-empty string")

    threshold = data.get("threshold")
    if not _is_number(threshold):
        raise ConfigError(f"{label}: start_transition.threshold must be a number")
    if not 0.0 <= threshold <= 1.0:
        raise ConfigError(
            f"{label}: start_transition.threshold must be between 0.0 and 1.0"
        )

    roi_override = None
    roi_data = data.get("roi")
    if roi_data is not None:
        if not isinstance(roi_data, dict):
            raise ConfigError(f"{label}: start_transition.roi must be a mapping")
        roi_override = _parse_roi(roi_data, f"{label}: start_transition")

    auto_start = data.get("auto_start_timer", False)
    if not isinstance(auto_start, bool):
        raise ConfigError(
            f"{label}: start_transition.auto_start_timer must be a boolean"
        )

    return StartTransition(
        name=name,
        color=color_name,
        threshold=float(threshold),
        roi_override=roi_override,
        auto_start_timer=auto_start,
    )


def _parse_record(value: object, label: str) -> Record | None:
    """Parse optional record section.

    Returns None if the section is absent or empty (recording disabled).
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ConfigError(f"{label}: record must be a mapping")
    data = cast(dict[str, object], value)
    if not data:
        return None

    path_template = data.get("path_template")
    if not isinstance(path_template, str) or not path_template.strip():
        raise ConfigError(f"{label}: record.path_template must be a non-empty string")

    convert_to = data.get("convert_to")
    if convert_to is not None and (not isinstance(convert_to, str) or not convert_to.strip()):
        raise ConfigError(f"{label}: record.convert_to must be a non-empty string or null")

    post_run = data.get("post_run_seconds", 3.0)
    if not _is_number(post_run):
        raise ConfigError(f"{label}: record.post_run_seconds must be a number")
    if post_run < 0:
        raise ConfigError(f"{label}: record.post_run_seconds must not be negative")

    return Record(
        path_template=path_template,
        convert_to=convert_to if isinstance(convert_to, str) else None,
        post_run_seconds=float(post_run),
    )


def _parse_ocr(value: object, label: str) -> OcrConfig | None:
    """Parse optional ocr section.

    Returns None if the section is absent or empty (OCR disabled).
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ConfigError(f"{label}: ocr must be a mapping")
    data = cast(dict[str, object], value)
    if not data:
        return None

    roi = _parse_roi(data.get("roi"), f"{label}: ocr")

    # Engine settings
    engine_data = data.get("engine", {})
    if engine_data is None:
        engine_data = {}
    if not isinstance(engine_data, dict):
        raise ConfigError(f"{label}: ocr.engine must be a mapping")
    engine_raw = cast(dict[str, object], engine_data)

    language = engine_raw.get("language", "eng")
    if not isinstance(language, str) or not language.strip():
        raise ConfigError(f"{label}: ocr.engine.language must be a non-empty string")

    psm = engine_raw.get("psm", 7)
    if not isinstance(psm, int) or isinstance(psm, bool):
        raise ConfigError(f"{label}: ocr.engine.psm must be an integer")
    if not 0 <= psm <= 13:
        raise ConfigError(f"{label}: ocr.engine.psm must be between 0 and 13")

    min_conf = engine_raw.get("min_confidence", 60.0)
    if not _is_number(min_conf):
        raise ConfigError(f"{label}: ocr.engine.min_confidence must be a number")
    if not 0.0 <= min_conf <= 100.0:
        raise ConfigError(
            f"{label}: ocr.engine.min_confidence must be between 0.0 and 100.0"
        )

    engine = OcrEngine(
        language=language,
        psm=int(psm),
        min_confidence=float(min_conf),
    )

    # Patterns
    raw_patterns = data.get("patterns")
    if raw_patterns is None:
        raw_patterns = {}
    if not isinstance(raw_patterns, dict):
        raise ConfigError(f"{label}: ocr.patterns must be a mapping")
    patterns: list[OcrPattern] = []
    for key, val in cast(dict[str, object], raw_patterns).items():
        if not isinstance(key, str):
            raise ConfigError(f"{label}: ocr.patterns keys must be strings")
        # Validate regex compiles
        import re

        try:
            re.compile(key)
        except re.error as exc:
            raise ConfigError(
                f"{label}: ocr.patterns key '{key}' is not a valid regex: {exc}"
            ) from exc
        if val is not None and not isinstance(val, int):
            raise ConfigError(
                f"{label}: ocr.patterns['{key}'] must be an integer or null"
            )
        patterns.append(
            OcrPattern(regex=key, split_index=val if isinstance(val, int) else None)
        )

    poll_interval = data.get("poll_interval_frames", 3)
    if not isinstance(poll_interval, int) or isinstance(poll_interval, bool):
        raise ConfigError(
            f"{label}: ocr.poll_interval_frames must be an integer"
        )
    if poll_interval < 1:
        raise ConfigError(
            f"{label}: ocr.poll_interval_frames must be at least 1"
        )

    return OcrConfig(
        roi=roi,
        engine=engine,
        patterns=tuple(patterns),
        poll_interval_frames=int(poll_interval),
    )
