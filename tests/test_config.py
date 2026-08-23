"""Tests for autosplitter.config.

Each invalid case asserts both the ConfigError type and the exact message,
because precise errors are a project requirement (spec section 17).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autosplitter.config import (
    DEFAULT_ROI,
    Capture,
    Color,
    Config,
    ConfigError,
    Detection,
    LiveSplit,
    Roi,
    Split,
    load,
)

FIXTURES = Path(__file__).parent / "fixtures"


def write_config(tmp_path: Path, text: str) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def load_fixture() -> Config:
    return load(FIXTURES / "valid_config.yaml")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestValidConfig:
    def test_reference_config_loads(self) -> None:
        cfg = load_fixture()

        assert cfg.capture == Capture(
            fps=60.0,
            stream_url="udp://127.0.0.1:9733",
            width=1920,
            height=1080,
        )
        assert cfg.roi == Roi(x=0.25, y=0.25, width=0.50, height=0.50)
        assert cfg.detection == Detection(required_frames=3, reset_threshold=0.80)
        assert cfg.colors["black"] == Color(name="black", rgb=(0, 0, 0), tolerance=15.0)
        assert cfg.colors["white"].rgb == (255, 255, 255)
        assert len(cfg.splits) == 3
        assert cfg.splits[0] == Split(id=1, name="Transition 1", color="black", threshold=0.95)
        assert cfg.livesplit == LiveSplit(host="127.0.0.1", port=16834)

    def test_color_lookup_resolves_defined_color(self) -> None:
        cfg = load_fixture()
        assert cfg.color("white").rgb == (255, 255, 255)

    def test_missing_roi_defaults_to_middle_50_percent(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            """
            capture:
              fps: 60
              stream_url: "udp://127.0.0.1:9733"
              width: 1920
              height: 1080
            detection:
              required_frames: 3
              reset_threshold: 0.80
            colors:
              black:
                rgb: [0, 0, 0]
                tolerance: 15
            splits:
              - id: 1
                name: "Only"
                color: black
                threshold: 0.95
            livesplit:
              host: "127.0.0.1"
              port: 16834
            """,
        )

        cfg = load(path)
        assert cfg.roi == DEFAULT_ROI


# ---------------------------------------------------------------------------
# File-level errors
# ---------------------------------------------------------------------------


class TestFileErrors:
    def test_missing_file_raises_precise_error(self, tmp_path: Path) -> None:
        missing = str(tmp_path / "nope.yaml")

        with pytest.raises(ConfigError, match=r"nope\.yaml: cannot read file"):
            load(missing)

    def test_invalid_yaml_reports_clean_error(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, "capture: [unclosed")

        with pytest.raises(ConfigError, match=r"config\.yaml: invalid YAML"):
            load(path)

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, "")

        with pytest.raises(ConfigError, match="file is empty"):
            load(path)

    def test_top_level_list_rejected(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, "- just\n- a\n- list\n")

        with pytest.raises(ConfigError, match="top level must be a mapping"):
            load(path)


# ---------------------------------------------------------------------------
# ROI validation
# ---------------------------------------------------------------------------


class TestRoi:
    @pytest.mark.parametrize("key", ["x", "y", "width", "height"])
    def test_value_above_one_rejected(self, tmp_path: Path, key: str) -> None:
        base = {
            "x": 0.25,
            "y": 0.25,
            "width": 0.50,
            "height": 0.50,
            key: 1.5,
        }
        roi_block = "\n".join(f"  {k}: {v}" for k, v in base.items())
        text = (
            "capture:\n"
            "  fps: 60\n"
            '  stream_url: "udp://127.0.0.1:9733"\n'
            "  width: 1920\n"
            "  height: 1080\n"
            "roi:\n"
            f"{roi_block}\n"
            "detection:\n"
            "  required_frames: 3\n"
            "  reset_threshold: 0.80\n"
            "colors:\n"
            "  black:\n"
            "    rgb: [0, 0, 0]\n"
            "    tolerance: 15\n"
            "splits:\n"
            "  - id: 1\n"
            '    name: "S"\n'
            "    color: black\n"
            "    threshold: 0.95\n"
            "livesplit:\n"
            '  host: "127.0.0.1"\n'
            "  port: 16834\n"
        )
        path = write_config(tmp_path, text)

        with pytest.raises(ConfigError, match=f"roi.{key} must be between 0.0 and 1.0"):
            load(path)

    def test_negative_roi_rejected(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            """
            capture:
              fps: 60
              stream_url: "udp://127.0.0.1:9733"
              width: 1920
              height: 1080
            roi:
              x: -0.1
              y: 0.25
              width: 0.5
              height: 0.5
            detection:
              required_frames: 3
              reset_threshold: 0.80
            colors:
              black:
                rgb: [0, 0, 0]
                tolerance: 15
            splits:
              - id: 1
                name: "S"
                color: black
                threshold: 0.95
            livesplit:
              host: "127.0.0.1"
              port: 16834
            """,
        )

        with pytest.raises(ConfigError, match="roi.x must be between 0.0 and 1.0"):
            load(path)

    def test_roi_overflowing_frame_rejected(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            """
            capture:
              fps: 60
              stream_url: "udp://127.0.0.1:9733"
              width: 1920
              height: 1080
            roi:
              x: 0.8
              y: 0.25
              width: 0.5
              height: 0.5
            detection:
              required_frames: 3
              reset_threshold: 0.80
            colors:
              black:
                rgb: [0, 0, 0]
                tolerance: 15
            splits:
              - id: 1
                name: "S"
                color: black
                threshold: 0.95
            livesplit:
              host: "127.0.0.1"
              port: 16834
            """,
        )

        with pytest.raises(ConfigError, match=r"roi\.x \+ roi\.width must not exceed 1\.0"):
            load(path)


# ---------------------------------------------------------------------------
# ROI endpoint form (x1/y1/x2/y2)
# ---------------------------------------------------------------------------


def full_config(roi_block: str) -> str:
    return f"""capture:
  fps: 60
  stream_url: "udp://127.0.0.1:9733"
  width: 1920
  height: 1080
roi:
{roi_block}
detection:
  required_frames: 3
  reset_threshold: 0.80
colors:
  black:
    rgb: [0, 0, 0]
    tolerance: 15
splits:
  - id: 1
    name: "S"
    color: black
    threshold: 0.95
livesplit:
  host: "127.0.0.1"
  port: 16834
"""


class TestRoiEndpoints:
    def test_middle_50_percent_endpoints_load_equivalently(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            full_config("  x1: 0.25\n  y1: 0.25\n  x2: 0.75\n  y2: 0.75"),
        )

        cfg = load(path)
        assert cfg.roi == Roi(x=0.25, y=0.25, width=0.50, height=0.50)

    def test_arbitrary_rectangle(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            full_config("  x1: 0.10\n  y1: 0.20\n  x2: 0.90\n  y2: 0.60"),
        )

        cfg = load(path)
        assert cfg.roi.x == pytest.approx(0.10)
        assert cfg.roi.y == pytest.approx(0.20)
        assert cfg.roi.width == pytest.approx(0.80)
        assert cfg.roi.height == pytest.approx(0.40)

    def test_mixing_size_and_endpoint_forms_rejected(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            full_config("  x: 0.25\n  y: 0.25\n  width: 0.5\n  height: 0.5\n  x2: 0.75"),
        )

        with pytest.raises(
            ConfigError,
            match="roi must use either x/y/width/height or the endpoints",
        ):
            load(path)

    def test_incomplete_endpoint_form_rejected(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            full_config("  x1: 0.25\n  y1: 0.25\n  x2: 0.75"),
        )

        with pytest.raises(
            ConfigError,
            match=r"roi\.y2 is required when using endpoint form",
        ):
            load(path)

    def test_reversed_x_endpoints_rejected(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            full_config("  x1: 0.75\n  y1: 0.25\n  x2: 0.25\n  y2: 0.75"),
        )

        with pytest.raises(
            ConfigError,
            match=r"roi\.x2 must be greater than or equal to roi\.x1",
        ):
            load(path)

    def test_endpoint_out_of_range_rejected(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            full_config("  x1: 0.25\n  y1: 0.25\n  x2: 1.50\n  y2: 0.75"),
        )

        with pytest.raises(ConfigError, match="roi.x2 must be between 0.0 and 1.0"):
            load(path)

    def test_zero_width_via_endpoints_rejected(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            full_config("  x1: 0.50\n  y1: 0.25\n  x2: 0.50\n  y2: 0.75"),
        )

        with pytest.raises(ConfigError, match="roi.width must be greater than 0"):
            load(path)

    def test_zero_width_via_size_form_rejected(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            full_config("  x: 0.25\n  y: 0.25\n  width: 0.0\n  height: 0.5"),
        )

        with pytest.raises(ConfigError, match="roi.width must be greater than 0"):
            load(path)


# ---------------------------------------------------------------------------
# Color validation
# ---------------------------------------------------------------------------


class TestColors:
    def test_rgb_out_of_range_rejected(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            """
            capture:
              fps: 60
              stream_url: "udp://127.0.0.1:9733"
              width: 1920
              height: 1080
            detection:
              required_frames: 3
              reset_threshold: 0.80
            colors:
              pink:
                rgb: [300, 0, 0]
                tolerance: 15
            splits:
              - id: 1
                name: "S"
                color: pink
                threshold: 0.95
            livesplit:
              host: "127.0.0.1"
              port: 16834
            """,
        )

        with pytest.raises(ConfigError, match="colors.pink.rgb values must be between 0 and 255"):
            load(path)

    def test_rgb_wrong_length_rejected(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            """
            capture:
              fps: 60
              stream_url: "udp://127.0.0.1:9733"
              width: 1920
              height: 1080
            detection:
              required_frames: 3
              reset_threshold: 0.80
            colors:
              teal:
                rgb: [0, 128]
                tolerance: 15
            splits:
              - id: 1
                name: "S"
                color: teal
                threshold: 0.95
            livesplit:
              host: "127.0.0.1"
              port: 16834
            """,
        )

        with pytest.raises(ConfigError, match=r"colors\.teal\.rgb must be a list of three integers"):
            load(path)

    def test_negative_tolerance_rejected(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            """
            capture:
              fps: 60
              stream_url: "udp://127.0.0.1:9733"
              width: 1920
              height: 1080
            detection:
              required_frames: 3
              reset_threshold: 0.80
            colors:
              black:
                rgb: [0, 0, 0]
                tolerance: -1
            splits:
              - id: 1
                name: "S"
                color: black
                threshold: 0.95
            livesplit:
              host: "127.0.0.1"
              port: 16834
            """,
        )

        with pytest.raises(ConfigError, match="colors.black.tolerance must be a non-negative number"):
            load(path)


# ---------------------------------------------------------------------------
# Detection validation
# ---------------------------------------------------------------------------


class TestDetection:
    def test_reset_threshold_above_one_rejected(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            """
            capture:
              fps: 60
              stream_url: "udp://127.0.0.1:9733"
              width: 1920
              height: 1080
            detection:
              required_frames: 3
              reset_threshold: 1.2
            colors:
              black:
                rgb: [0, 0, 0]
                tolerance: 15
            splits:
              - id: 1
                name: "S"
                color: black
                threshold: 0.95
            livesplit:
              host: "127.0.0.1"
              port: 16834
            """,
        )

        with pytest.raises(
            ConfigError,
            match=r"detection\.reset_threshold must be between 0\.0 and 1\.0",
        ):
            load(path)

    def test_zero_required_frames_rejected(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            """
            capture:
              fps: 60
              stream_url: "udp://127.0.0.1:9733"
              width: 1920
              height: 1080
            detection:
              required_frames: 0
              reset_threshold: 0.80
            colors:
              black:
                rgb: [0, 0, 0]
                tolerance: 15
            splits:
              - id: 1
                name: "S"
                color: black
                threshold: 0.95
            livesplit:
              host: "127.0.0.1"
              port: 16834
            """,
        )

        with pytest.raises(ConfigError, match="required_frames must be a positive integer"):
            load(path)

    def test_non_integer_required_frames_rejected(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            """
            capture:
              fps: 60
              stream_url: "udp://127.0.0.1:9733"
              width: 1920
              height: 1080
            detection:
              required_frames: true
              reset_threshold: 0.80
            colors:
              black:
                rgb: [0, 0, 0]
                tolerance: 15
            splits:
              - id: 1
                name: "S"
                color: black
                threshold: 0.95
            livesplit:
              host: "127.0.0.1"
              port: 16834
            """,
        )

        with pytest.raises(ConfigError, match="required_frames must be an integer"):
            load(path)


# ---------------------------------------------------------------------------
# Split validation
# ---------------------------------------------------------------------------

THREE_SPLITS_TEMPLATE = """
capture:
  fps: 60
  stream_url: "udp://127.0.0.1:9733"
  width: 1920
  height: 1080
detection:
  required_frames: 3
  reset_threshold: 0.80
colors:
  black:
    rgb: [0, 0, 0]
    tolerance: 15
  white:
    rgb: [255, 255, 255]
    tolerance: 15
splits:
  - id: 1
    name: "First"
    color: black
    threshold: {t1}
  - id: 2
    name: "Second"
    color: white
    threshold: {t2}
  - id: 3
    name: "Third"
    color: {c3}
    threshold: {t3}
livesplit:
  host: "127.0.0.1"
  port: 16834
"""


class TestSplits:
    def test_spec_example_exact_message(self, tmp_path: Path) -> None:
        """Spec section 17 requires exactly this error for this config shape."""
        path = write_config(tmp_path, THREE_SPLITS_TEMPLATE.format(t1=0.95, t2=0.95, t3=1.5, c3="black"))

        with pytest.raises(ConfigError) as excinfo:
            load(path)
        # Label is the path as given; the spec-mandated suffix must match exactly.
        assert str(excinfo.value).endswith(
            "config.yaml: splits[2].threshold must be between 0.0 and 1.0"
        )

    def test_duplicate_ids_rejected(self, tmp_path: Path) -> None:
        text = THREE_SPLITS_TEMPLATE.format(t1=0.95, t2=0.95, t3=0.95, c3="black").replace(
            "- id: 2", "- id: 1"
        )
        path = write_config(tmp_path, text)

        with pytest.raises(ConfigError, match=r"splits\[1\]\.id must be unique \(1 already used\)"):
            load(path)

    def test_undefined_color_reference_rejected(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, THREE_SPLITS_TEMPLATE.format(t1=0.95, t2=0.95, t3=0.90, c3="blue"))

        with pytest.raises(
            ConfigError,
            match=r"splits\[2\]\.color must be one of the defined colors: black, white",
        ):
            load(path)


# ---------------------------------------------------------------------------
# Capture / LiveSplit validation
# ---------------------------------------------------------------------------


CAPTURE_TEMPLATE = """capture:
{capture_block}
detection:
  required_frames: 3
  reset_threshold: 0.80
colors:
  black:
    rgb: [0, 0, 0]
    tolerance: 15
splits:
  - id: 1
    name: "S"
    color: black
    threshold: 0.95
livesplit:
  host: "127.0.0.1"
  port: 16834
"""

VALID_CAPTURE_BLOCK = (
    '  fps: 60\n'
    '  stream_url: "udp://127.0.0.1:9733"\n'
    "  width: 1920\n"
    "  height: 1080"
)


def capture_config(tmp_path: Path, capture_block: str) -> str:
    return write_config(tmp_path, CAPTURE_TEMPLATE.format(capture_block=capture_block))


class TestCaptureAndLivesplit:
    def test_zero_fps_rejected(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            """
            capture:
              fps: 0
              stream_url: "udp://127.0.0.1:9733"
              width: 1920
              height: 1080
            detection:
              required_frames: 3
              reset_threshold: 0.80
            colors:
              black:
                rgb: [0, 0, 0]
                tolerance: 15
            splits:
              - id: 1
                name: "S"
                color: black
                threshold: 0.95
            livesplit:
              host: "127.0.0.1"
              port: 16834
            """,
        )

        with pytest.raises(ConfigError, match="capture.fps must be greater than 0"):
            load(path)

    def test_valid_stream_settings_load(self, tmp_path: Path) -> None:
        path = capture_config(tmp_path, VALID_CAPTURE_BLOCK)

        cfg = load(path)
        assert cfg.capture == Capture(
            fps=60.0,
            stream_url="udp://127.0.0.1:9733",
            width=1920,
            height=1080,
        )

    def test_bad_scheme_rejected(self, tmp_path: Path) -> None:
        block = VALID_CAPTURE_BLOCK.replace(
            'udp://127.0.0.1:9733', 'http://example.com'
        )
        path = capture_config(tmp_path, block)

        with pytest.raises(
            ConfigError,
            match=r"capture\.stream_url must start with udp:// or tcp://",
        ):
            load(path)

    def test_empty_stream_url_rejected(self, tmp_path: Path) -> None:
        block = VALID_CAPTURE_BLOCK.replace(
            '"udp://127.0.0.1:9733"', '""'
        )
        path = capture_config(tmp_path, block)

        with pytest.raises(
            ConfigError, match=r"capture\.stream_url must be a non-empty string"
        ):
            load(path)

    def test_non_string_stream_url_rejected(self, tmp_path: Path) -> None:
        block = VALID_CAPTURE_BLOCK.replace(
            '"udp://127.0.0.1:9733"', "123"
        )
        path = capture_config(tmp_path, block)

        with pytest.raises(
            ConfigError, match=r"capture\.stream_url must be a non-empty string"
        ):
            load(path)

    def test_missing_stream_url_rejected(self, tmp_path: Path) -> None:
        block = "\n".join(
            line for line in VALID_CAPTURE_BLOCK.splitlines() if "stream_url" not in line
        )
        path = capture_config(tmp_path, block)

        with pytest.raises(
            ConfigError, match=r"capture\.stream_url must be a non-empty string"
        ):
            load(path)

    def test_boolean_width_rejected(self, tmp_path: Path) -> None:
        block = VALID_CAPTURE_BLOCK.replace("width: 1920", "width: true")
        path = capture_config(tmp_path, block)

        with pytest.raises(ConfigError, match=r"capture\.width must be an integer"):
            load(path)

    def test_zero_width_rejected(self, tmp_path: Path) -> None:
        block = VALID_CAPTURE_BLOCK.replace("width: 1920", "width: 0")
        path = capture_config(tmp_path, block)

        with pytest.raises(
            ConfigError, match=r"capture\.width must be a positive integer"
        ):
            load(path)

    def test_negative_height_rejected(self, tmp_path: Path) -> None:
        block = VALID_CAPTURE_BLOCK.replace("height: 1080", "height: -1")
        path = capture_config(tmp_path, block)

        with pytest.raises(
            ConfigError, match=r"capture\.height must be a positive integer"
        ):
            load(path)

    def test_port_out_of_range_rejected(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            """
            capture:
              fps: 60
              stream_url: "udp://127.0.0.1:9733"
              width: 1920
              height: 1080
            detection:
              required_frames: 3
              reset_threshold: 0.80
            colors:
              black:
                rgb: [0, 0, 0]
                tolerance: 15
            splits:
              - id: 1
                name: "S"
                color: black
                threshold: 0.95
            livesplit:
              host: "127.0.0.1"
              port: 99999
            """,
        )

        with pytest.raises(ConfigError, match=r"livesplit\.port must be between 1 and 65535"):
            load(path)

    def test_missing_required_section_names_the_section(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            """
            capture:
              fps: 60
              stream_url: "udp://127.0.0.1:9733"
              width: 1920
              height: 1080
            colors:
              black:
                rgb: [0, 0, 0]
                tolerance: 15
            splits:
              - id: 1
                name: "S"
                color: black
                threshold: 0.95
            livesplit:
              host: "127.0.0.1"
              port: 16834
            """,
        )

        with pytest.raises(ConfigError, match="missing required section 'detection'"):
            load(path)
