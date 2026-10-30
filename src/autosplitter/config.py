"""Configuration loading and validation.

All tunables live in config.yaml (AGENTS.md non-negotiable). Validation must
fail with precise, path-style errors such as:

    config.yaml: splits[2].threshold must be between 0.0 and 1.0
"""

from __future__ import annotations

import math
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
    follow_timer: bool = False
    follow_pause: bool = False
    poll_seconds: float = 2.0


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
    backend: str = "auto"
    whitelist: str | None = None
    timeout_seconds: float = 2.0


@dataclass(frozen=True)
class OcrPreprocess:
    """Image preprocessing applied to the OCR crop before the engine reads it."""

    scale: int = 4
    blur: int = 3
    threshold: str = "otsu"
    pad: int = 10
    # threshold == "color": RGB text colour (converted to BGR bounds once at
    # worker construction) and inclusive per-channel tolerance.
    text_color: tuple[int, int, int] | None = None
    text_tolerance: int = 40


@dataclass(frozen=True)
class OcrFilter:
    """Trust rules applied to OCR readings before they count as evidence."""

    consensus_frames: int = 3
    jump_consensus_frames: int = 6
    max_age_seconds: float = 0.5
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
class OcrUndo:
    """"Go then undo" judge settings (``ocr.undo``).

    The colour detector still splits on the transition; the judge decides
    whether the HUD number failed to advance (a death) afterwards. With
    ``ocr.mode: verify_only`` it only LOGS what it would do (dry run); with
    ``ocr.mode: undo`` (and ``--livesplit``) the executor sends ``unsplit``.
    """

    settle_seconds: float = 3.0
    confirm_reads: int = 3
    confirm_seconds: float = 1.0
    window_seconds: float = 10.0
    min_confidence: float = 80.0
    require_baseline: bool = True
    baseline_seconds: float = 2.0
    # A D-1 read counts only this long after the advancing CLEAR (settle is
    # measured from the FIRE; a long-lit transition could otherwise count
    # while the old HUD number still lingers).
    post_clear_seconds: float = 1.0
    # Opt-in: also judge (and, in mode undo, undo) a death on the LAST level.
    # After the last split the HUD level number should VANISH on a genuine
    # finish; it keeps showing the level after a death. A false undo of a
    # genuine finish has no automatic recovery, so last-split windows wait
    # ``last_split_extra_settle_seconds`` longer than ``settle_seconds``.
    apply_to_last_split: bool = False
    last_split_extra_settle_seconds: float = 2.0
    # ``ocr.mode: undo`` only: after a verified undo of split k, if the HUD
    # shows k+1 within ``recover_seconds`` (the number was merely late), send
    # ``split`` again. The re-split is recorded late by the HUD lag.
    recover_late_hud: bool = True
    recover_seconds: float = 10.0


@dataclass(frozen=True)
class OcrConfig:
    """OCR-based split number detection (augments color detection).

    Reads text from a separate ROI and compares the recognized split number
    with the color-driven index (``mode: verify_only`` is read-only: it logs
    and shows the comparison but never changes detector or LiveSplit state;
    ``mode: undo`` is the opt-in executor that may send ``unsplit``).
    """

    roi: Roi
    engine: OcrEngine = OcrEngine()
    patterns: tuple[OcrPattern, ...] = ()
    poll_interval_frames: int = 3
    enabled: bool = True
    mode: str = "verify_only"
    max_reads_per_second: float = 4.0
    preprocess: OcrPreprocess = OcrPreprocess()
    filter: OcrFilter = OcrFilter()
    undo: OcrUndo | None = None


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
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, int):
        try:
            _ = float(value)  # ints beyond float range are not usable numbers
        except OverflowError:
            return False
    return True


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
    # Cross-section rule (after every per-section error): the OCR undo executor
    # relies on follow_timer's rollback/skip rules to repair what it cannot
    # confirm. resync_on_unsplit alone is not enough.
    if ocr is not None and ocr.mode == "undo" and not livesplit.follow_timer:
        raise ConfigError(f'{label}: ocr.mode "undo" requires livesplit.follow_timer: true')

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

    follow_value = data.get("follow_timer")
    if follow_value is not None and not isinstance(follow_value, bool):
        raise ConfigError(f"{label}: livesplit.follow_timer must be a boolean")

    pause_value = data.get("follow_pause")
    if pause_value is not None and not isinstance(pause_value, bool):
        raise ConfigError(f"{label}: livesplit.follow_pause must be a boolean")
    if pause_value is True and follow_value is not True:
        raise ConfigError(f"{label}: livesplit.follow_pause requires livesplit.follow_timer")

    poll_value = data.get("poll_seconds")
    if poll_value is not None:
        if not _is_number(poll_value):
            raise ConfigError(f"{label}: livesplit.poll_seconds must be a number")
        if not math.isfinite(poll_value) or poll_value <= 0:
            raise ConfigError(
                f"{label}: livesplit.poll_seconds must be greater than 0 and finite"
            )

    return LiveSplit(
        host=host,
        port=port,
        timeout_seconds=float(timeout_value) if timeout_value is not None else 0.25,
        reconnect_seconds=(
            float(reconnect_value) if reconnect_value is not None else 5.0
        ),
        resync_on_unsplit=resync_value is True,
        follow_timer=follow_value is True,
        follow_pause=pause_value is True,
        poll_seconds=float(poll_value) if poll_value is not None else 2.0,
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

    backend = engine_raw.get("backend", "auto")
    if backend not in ("auto", "tesseract_cli", "tesserocr"):
        raise ConfigError(
            f'{label}: ocr.engine.backend must be "auto", "tesseract_cli" or "tesserocr"'
        )
    whitelist = engine_raw.get("whitelist")
    if whitelist is not None and (not isinstance(whitelist, str) or not whitelist):
        raise ConfigError(f"{label}: ocr.engine.whitelist must be a non-empty string or null")
    timeout = engine_raw.get("timeout_seconds", 2.0)
    if not _is_number(timeout) or isinstance(timeout, bool):
        raise ConfigError(f"{label}: ocr.engine.timeout_seconds must be a number")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ConfigError(f"{label}: ocr.engine.timeout_seconds must be greater than 0")

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

    enabled = data.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ConfigError(f"{label}: ocr.enabled must be true or false")

    mode = data.get("mode", "verify_only")
    if mode not in ("verify_only", "undo"):
        raise ConfigError(f'{label}: ocr.mode must be "verify_only" or "undo"')
    if mode == "undo" and not patterns:
        raise ConfigError(f'{label}: ocr.patterns must not be empty when ocr.mode is "undo"')

    max_rate = data.get("max_reads_per_second", 4.0)
    if not _is_number(max_rate) or isinstance(max_rate, bool):
        raise ConfigError(f"{label}: ocr.max_reads_per_second must be a number")
    if not math.isfinite(max_rate) or not 0 < max_rate <= 30:
        raise ConfigError(
            f"{label}: ocr.max_reads_per_second must be greater than 0 and at most 30"
        )

    engine = OcrEngine(
        language=language,
        psm=int(psm),
        min_confidence=float(min_conf),
        backend=backend,
        whitelist=whitelist,
        timeout_seconds=float(timeout),
    )
    preprocess = _parse_ocr_preprocess(data.get("preprocess"), label)
    flt = _parse_ocr_filter(data.get("filter"), label, float(min_conf))
    undo = _parse_ocr_undo(data.get("undo"), label, bool(patterns))
    if mode == "undo" and undo is None:
        undo = OcrUndo()  # mode: undo without an undo: section = all defaults

    return OcrConfig(
        roi=roi,
        engine=engine,
        patterns=tuple(patterns),
        poll_interval_frames=int(poll_interval),
        enabled=enabled,
        mode=mode,
        max_reads_per_second=float(max_rate),
        preprocess=preprocess,
        filter=flt,
        undo=undo,
    )


def _ocr_subsection(value: object, label: str, name: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{label}: ocr.{name} must be a mapping")
    return cast(dict[str, object], value)


def _ocr_int(raw: dict[str, object], key: str, default: int, label: str, path: str) -> int:
    val = raw.get(key, default)
    if not isinstance(val, int) or isinstance(val, bool):
        raise ConfigError(f"{label}: ocr.{path}.{key} must be an integer")
    return val


def _parse_ocr_preprocess(value: object, label: str) -> OcrPreprocess:
    raw = _ocr_subsection(value, label, "preprocess")
    scale = _ocr_int(raw, "scale", 4, label, "preprocess")
    if not 1 <= scale <= 8:
        raise ConfigError(f"{label}: ocr.preprocess.scale must be between 1 and 8")
    blur = _ocr_int(raw, "blur", 3, label, "preprocess")
    if blur < 0 or (blur != 0 and blur % 2 == 0):
        raise ConfigError(f"{label}: ocr.preprocess.blur must be 0 or an odd integer")
    threshold = raw.get("threshold", "otsu")
    if threshold not in ("otsu", "none", "color"):
        raise ConfigError(
            f'{label}: ocr.preprocess.threshold must be "otsu", "none" or "color"'
        )
    pad = _ocr_int(raw, "pad", 10, label, "preprocess")
    if not 0 <= pad <= 50:
        raise ConfigError(f"{label}: ocr.preprocess.pad must be between 0 and 50")
    text_color: tuple[int, int, int] | None = None
    raw_color = raw.get("text_color")
    if raw_color is not None:
        rgb = _coerce_rgb_list(raw_color)
        if rgb is None or not all(0 <= c <= 255 for c in rgb):
            raise ConfigError(
                f"{label}: ocr.preprocess.text_color must be [r, g, b] with values 0-255"
            )
        text_color = (rgb[0], rgb[1], rgb[2])
    elif threshold == "color":
        raise ConfigError(
            f'{label}: ocr.preprocess.text_color is required when'
            + ' ocr.preprocess.threshold is "color"'
        )
    tol = _ocr_int(raw, "text_tolerance", 40, label, "preprocess")
    if not 0 <= tol <= 255:
        raise ConfigError(f"{label}: ocr.preprocess.text_tolerance must be between 0 and 255")
    return OcrPreprocess(
        scale=scale, blur=blur, threshold=threshold, pad=pad,
        text_color=text_color, text_tolerance=tol,
    )


def _undo_float(raw: dict[str, object], key: str, default: float, label: str) -> float:
    val = raw.get(key, default)
    if not _is_number(val) or isinstance(val, bool):
        raise ConfigError(f"{label}: ocr.undo.{key} must be a number")
    if not math.isfinite(val):
        raise ConfigError(f"{label}: ocr.undo.{key} must be a finite number")
    return float(val)


def _undo_bool(raw: dict[str, object], key: str, default: bool, label: str) -> bool:
    val = raw.get(key, default)
    if not isinstance(val, bool):
        raise ConfigError(f"{label}: ocr.undo.{key} must be true or false")
    return val


def _parse_ocr_undo(value: object, label: str, has_patterns: bool) -> OcrUndo | None:
    """Parse the optional ``ocr.undo`` section (None when absent)."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ConfigError(f"{label}: ocr.undo must be a mapping")
    raw = cast(dict[str, object], value)
    if not has_patterns:
        raise ConfigError(f"{label}: ocr.patterns must not be empty when ocr.undo is set")
    settle = _undo_float(raw, "settle_seconds", 3.0, label)
    if settle <= 0:
        raise ConfigError(f"{label}: ocr.undo.settle_seconds must be greater than 0")
    reads = raw.get("confirm_reads", 3)
    if not isinstance(reads, int) or isinstance(reads, bool):
        raise ConfigError(f"{label}: ocr.undo.confirm_reads must be an integer")
    if reads < 1:
        raise ConfigError(f"{label}: ocr.undo.confirm_reads must be at least 1")
    confirm = _undo_float(raw, "confirm_seconds", 1.0, label)
    if confirm < 0:
        raise ConfigError(f"{label}: ocr.undo.confirm_seconds must be at least 0")
    if reads == 1 and confirm == 0:
        raise ConfigError(
            f"{label}: ocr.undo.confirm_reads of 1 requires"
            + " ocr.undo.confirm_seconds greater than 0"
        )
    post_clear = _undo_float(raw, "post_clear_seconds", 1.0, label)
    if post_clear < 0:
        raise ConfigError(f"{label}: ocr.undo.post_clear_seconds must be at least 0")
    window = _undo_float(raw, "window_seconds", 10.0, label)
    last = _undo_bool(raw, "apply_to_last_split", False, label)
    extra = _undo_float(raw, "last_split_extra_settle_seconds", 2.0, label)
    if extra < 0:
        raise ConfigError(
            f"{label}: ocr.undo.last_split_extra_settle_seconds must be at least 0"
        )
    if window <= settle + confirm:
        raise ConfigError(
            f"{label}: ocr.undo.window_seconds must be greater than"
            + " settle_seconds + confirm_seconds"
        )
    if last and window <= settle + extra + confirm:
        raise ConfigError(
            f"{label}: ocr.undo.window_seconds must be greater than settle_seconds"
            + " + last_split_extra_settle_seconds + confirm_seconds"
            + " when apply_to_last_split is true"
        )
    min_conf = _undo_float(raw, "min_confidence", 80.0, label)
    if not 0.0 <= min_conf <= 100.0:
        raise ConfigError(f"{label}: ocr.undo.min_confidence must be between 0.0 and 100.0")
    require_baseline = _undo_bool(raw, "require_baseline", True, label)
    baseline = _undo_float(raw, "baseline_seconds", 2.0, label)
    if baseline <= 0:
        raise ConfigError(f"{label}: ocr.undo.baseline_seconds must be greater than 0")
    recover = _undo_bool(raw, "recover_late_hud", True, label)
    recover_secs = _undo_float(raw, "recover_seconds", 10.0, label)
    if recover_secs <= 0:
        raise ConfigError(f"{label}: ocr.undo.recover_seconds must be greater than 0")
    return OcrUndo(
        settle_seconds=settle,
        confirm_reads=reads,
        confirm_seconds=confirm,
        window_seconds=window,
        min_confidence=min_conf,
        require_baseline=require_baseline,
        baseline_seconds=baseline,
        post_clear_seconds=post_clear,
        apply_to_last_split=last,
        last_split_extra_settle_seconds=extra,
        recover_late_hud=recover,
        recover_seconds=recover_secs,
    )


def _parse_ocr_filter(value: object, label: str, default_conf: float) -> OcrFilter:
    raw = _ocr_subsection(value, label, "filter")
    consensus = _ocr_int(raw, "consensus_frames", 3, label, "filter")
    if consensus < 1:
        raise ConfigError(f"{label}: ocr.filter.consensus_frames must be at least 1")
    jump = _ocr_int(raw, "jump_consensus_frames", 6, label, "filter")
    if jump < 1:
        raise ConfigError(f"{label}: ocr.filter.jump_consensus_frames must be at least 1")
    max_age = raw.get("max_age_seconds", 0.5)
    if not _is_number(max_age) or isinstance(max_age, bool):
        raise ConfigError(f"{label}: ocr.filter.max_age_seconds must be a number")
    if not math.isfinite(max_age) or max_age <= 0:
        raise ConfigError(f"{label}: ocr.filter.max_age_seconds must be greater than 0")
    min_conf = raw.get("min_confidence", default_conf)
    if not _is_number(min_conf) or isinstance(min_conf, bool):
        raise ConfigError(f"{label}: ocr.filter.min_confidence must be a number")
    if not 0.0 <= min_conf <= 100.0:
        raise ConfigError(
            f"{label}: ocr.filter.min_confidence must be between 0.0 and 100.0"
        )
    return OcrFilter(
        consensus_frames=consensus,
        jump_consensus_frames=jump,
        max_age_seconds=float(max_age),
        min_confidence=float(min_conf),
    )
