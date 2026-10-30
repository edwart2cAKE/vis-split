"""config.ocr parsing: legacy keys unchanged, new keys defaulted and validated."""


from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from autosplitter.config import Config, ConfigError, load

BASE = """
capture: {fps: 60, stream_url: "udp://127.0.0.1:9733", width: 1920, height: 1080}
roi: {x: 0.25, y: 0.25, width: 0.5, height: 0.5}
detection: {required_frames: 3, reset_threshold: 0.8}
colors: {black: {rgb: [0, 0, 0], tolerance: 15}}
splits:
  - {id: 1, name: S, color: black, threshold: 0.95}
livesplit: {host: "127.0.0.1", port: 16834}
"""
ROI = "  roi: {x: 0.0, y: 0.0, width: 0.1, height: 0.1}\n"


def _which_none(_name: str) -> None:
    return None


def cfg_with(tmp_path: Path, ocr_body: str, follow: bool | None = None) -> Config:
    """``mode: undo`` bodies get ``livesplit.follow_timer: true`` (now required)
    unless ``follow`` says otherwise."""
    if follow is None:
        follow = "mode: undo" in ocr_body
    base = BASE.replace('port: 16834}', 'port: 16834, follow_timer: true}') if follow else BASE
    p = tmp_path / "cfg.yaml"
    p.write_text(base + "ocr:\n" + ROI + ocr_body, encoding="utf-8")
    return load(p)


def err(tmp_path: Path, ocr_body: str, follow: bool | None = None) -> str:
    with pytest.raises(ConfigError) as ei:
        cfg_with(tmp_path, ocr_body, follow)
    return str(ei.value)


def test_defaults(tmp_path: Path) -> None:
    c = cfg_with(tmp_path, "")
    o = c.ocr
    assert o is not None
    assert o.enabled and o.mode == "verify_only"
    assert (o.engine.backend, o.engine.whitelist, o.engine.timeout_seconds) == ("auto", None, 2.0)
    assert (o.engine.language, o.engine.psm, o.engine.min_confidence) == ("eng", 7, 60.0)
    assert (o.preprocess.scale, o.preprocess.blur, o.preprocess.threshold, o.preprocess.pad) == (4, 3, "otsu", 10)
    f = o.filter
    assert (f.consensus_frames, f.jump_consensus_frames, f.max_age_seconds) == (3, 6, 0.5)
    assert f.min_confidence == 60.0
    assert o.poll_interval_frames == 3
    assert o.max_reads_per_second == 4.0


def test_filter_min_confidence_defaults_to_engine(tmp_path: Path) -> None:
    c = cfg_with(tmp_path, "  engine: {min_confidence: 77.5}\n")
    assert c.ocr is not None and c.ocr.filter.min_confidence == 77.5
    c = cfg_with(tmp_path, "  engine: {min_confidence: 77.5}\n  filter: {min_confidence: 40}\n")
    assert c.ocr is not None and c.ocr.filter.min_confidence == 40.0


def test_all_new_keys_parse(tmp_path: Path) -> None:
    c = cfg_with(
        tmp_path,
        "  enabled: true\n  mode: verify_only\n"
        + '  engine: {backend: tesseract_cli, whitelist: "0123456789", timeout_seconds: 1.5, psm: 10}\n'
        + "  preprocess: {scale: 2, blur: 0, threshold: none, pad: 0}\n"
        + "  filter: {consensus_frames: 1, jump_consensus_frames: 2, max_age_seconds: 0.25}\n",
    )
    o = c.ocr
    assert o is not None
    assert o.engine.whitelist == "0123456789" and o.engine.timeout_seconds == 1.5
    assert o.preprocess.scale == 2 and o.preprocess.threshold == "none"
    assert o.filter.consensus_frames == 1 and o.filter.max_age_seconds == 0.25


def test_start_powerless_ocr_block_parses() -> None:
    """Follows the shipped configs/start-powerless.yaml (re-tuned: tight ROI, colour threshold)."""
    c = load(Path(__file__).parent.parent / "configs" / "start-powerless.yaml")
    o = c.ocr
    assert o is not None
    assert (o.engine.language, o.engine.psm, o.engine.min_confidence) == ("eng", 7, 60.0)
    assert o.poll_interval_frames == 3
    # configs/start-powerless.yaml was re-tuned (tight ROI, colour threshold);
    # these pins follow the file's current values.
    assert [(p.regex, p.split_index) for p in o.patterns] == [("^\\s*(\\d+)\\s*$", None)]
    assert (o.roi.x, o.roi.y, o.roi.width, o.roi.height) == (0.1469, 0.1472, 0.0177, 0.0231)
    assert o.enabled


def test_enabled_false_still_parses(tmp_path: Path) -> None:
    c = cfg_with(tmp_path, "  enabled: false\n")
    assert c.ocr is not None and c.ocr.enabled is False


@pytest.mark.parametrize(
    ("body", "msg"),
    [
        ("  max_reads_per_second: 0\n", "ocr.max_reads_per_second must be greater than 0 and at most 30"),
        ("  max_reads_per_second: -1\n", "ocr.max_reads_per_second must be greater than 0"),
        ("  max_reads_per_second: 30.5\n", "ocr.max_reads_per_second must be greater than 0 and at most 30"),
        ("  max_reads_per_second: .inf\n", "ocr.max_reads_per_second must be greater than 0"),
        ("  max_reads_per_second: .nan\n", "ocr.max_reads_per_second must be greater than 0"),
        ("  max_reads_per_second: true\n", "ocr.max_reads_per_second must be a number"),
        ("  max_reads_per_second: fast\n", "ocr.max_reads_per_second must be a number"),
        ("  enabled: 1\n", "ocr.enabled must be true or false"),
        ("  mode: resync\n", 'ocr.mode must be "verify_only"'),
        ("  mode: 3\n", 'ocr.mode must be "verify_only"'),
        ("  engine: {backend: foo}\n", "ocr.engine.backend must be"),
        ("  engine: {whitelist: ''}\n", "ocr.engine.whitelist"),
        ("  engine: {whitelist: 5}\n", "ocr.engine.whitelist"),
        ("  engine: {timeout_seconds: 0}\n", "ocr.engine.timeout_seconds must be greater than 0"),
        ("  engine: {timeout_seconds: .inf}\n", "ocr.engine.timeout_seconds must be greater than 0"),
        ("  engine: {timeout_seconds: .nan}\n", "ocr.engine.timeout_seconds must be greater than 0"),
        ("  engine: {timeout_seconds: x}\n", "ocr.engine.timeout_seconds must be a number"),
        ("  engine: {timeout_seconds: true}\n", "ocr.engine.timeout_seconds must be a number"),
        ("  preprocess: {scale: 0}\n", "ocr.preprocess.scale must be between 1 and 8"),
        ("  preprocess: {scale: 9}\n", "ocr.preprocess.scale must be between 1 and 8"),
        ("  preprocess: {scale: true}\n", "ocr.preprocess.scale must be an integer"),
        ("  preprocess: {scale: 2.5}\n", "ocr.preprocess.scale must be an integer"),
        ("  preprocess: {blur: 2}\n", "ocr.preprocess.blur must be 0 or an odd integer"),
        ("  preprocess: {blur: -1}\n", "ocr.preprocess.blur must be 0 or an odd integer"),
        ("  preprocess: {threshold: adaptive}\n", "ocr.preprocess.threshold must be"),
        ("  preprocess: {pad: 51}\n", "ocr.preprocess.pad must be between 0 and 50"),
        ("  preprocess: {pad: -1}\n", "ocr.preprocess.pad must be between 0 and 50"),
        ("  preprocess: 5\n", "ocr.preprocess must be a mapping"),
        ("  filter: {consensus_frames: 0}\n", "ocr.filter.consensus_frames must be at least 1"),
        ("  filter: {consensus_frames: true}\n", "ocr.filter.consensus_frames must be an integer"),
        ("  filter: {jump_consensus_frames: 0}\n", "ocr.filter.jump_consensus_frames must be at least 1"),
        ("  filter: {max_age_seconds: 0}\n", "ocr.filter.max_age_seconds must be greater than 0"),
        ("  filter: {max_age_seconds: .inf}\n", "ocr.filter.max_age_seconds must be greater than 0"),
        ("  filter: {max_age_seconds: x}\n", "ocr.filter.max_age_seconds must be a number"),
        ("  filter: {min_confidence: 101}\n", "ocr.filter.min_confidence must be between"),
        ("  filter: 3\n", "ocr.filter must be a mapping"),
    ],
)
def test_validation_errors(tmp_path: Path, body: str, msg: str) -> None:
    assert msg in err(tmp_path, body)
    assert "cfg.yaml: " in err(tmp_path, body)


def test_check_prints_backend_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from autosplitter import main as m

    p = tmp_path / "cfg.yaml"
    p.write_text(BASE + "ocr:\n" + ROI, encoding="utf-8")
    monkeypatch.setattr(shutil, "which", _which_none)
    assert m.main(["--config", str(p), "--check"]) == 0
    out = capsys.readouterr().out
    assert "ocr backend: UNAVAILABLE (tesseract binary not found on PATH)" in out

    monkeypatch.undo()
    if shutil.which("tesseract") is not None:
        assert m.main(["--config", str(p), "--check"]) == 0
        assert "ocr backend: tesseract_cli OK (" in capsys.readouterr().out


def test_check_invalid_ocr_exit_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from autosplitter import main as m

    p = tmp_path / "cfg.yaml"
    p.write_text(BASE + "ocr:\n" + ROI + "  mode: resync\n", encoding="utf-8")
    assert m.main(["--config", str(p), "--check"]) == 1
    assert 'ocr.mode must be "verify_only"' in capsys.readouterr().err


def test_max_reads_per_second_bounds_ok(tmp_path: Path) -> None:
    for v, want in (("30", 30.0), ("0.5", 0.5), ("8", 8.0)):
        c = cfg_with(tmp_path, f"  max_reads_per_second: {v}\n")
        assert c.ocr is not None and c.ocr.max_reads_per_second == want


# --------------------------------------------------------------------------
# preprocess.threshold: color
# --------------------------------------------------------------------------


def test_threshold_color_parses_and_defaults(tmp_path: Path) -> None:
    c = cfg_with(tmp_path, "  preprocess: {threshold: color, text_color: [255, 250, 0]}\n")
    assert c.ocr is not None
    p = c.ocr.preprocess
    assert (p.threshold, p.text_color, p.text_tolerance) == ("color", (255, 250, 0), 40)
    c = cfg_with(tmp_path, "  preprocess: {threshold: color, text_color: [1, 2, 3], text_tolerance: 0}\n")
    assert c.ocr is not None and c.ocr.preprocess.text_tolerance == 0
    c = cfg_with(tmp_path, "  preprocess: {threshold: otsu}\n")
    assert c.ocr is not None and c.ocr.preprocess.text_color is None


@pytest.mark.parametrize(
    ("body", "msg"),
    [
        ("  preprocess: {threshold: color}\n",
         'ocr.preprocess.text_color is required when ocr.preprocess.threshold is "color"'),
        ("  preprocess: {threshold: color, text_color: [255, 255]}\n",
         "ocr.preprocess.text_color must be [r, g, b] with values 0-255"),
        ("  preprocess: {threshold: color, text_color: [256, 0, 0]}\n",
         "ocr.preprocess.text_color must be [r, g, b] with values 0-255"),
        ("  preprocess: {threshold: color, text_color: [-1, 0, 0]}\n",
         "ocr.preprocess.text_color must be [r, g, b] with values 0-255"),
        ("  preprocess: {threshold: color, text_color: [1.5, 0, 0]}\n",
         "ocr.preprocess.text_color must be [r, g, b] with values 0-255"),
        ("  preprocess: {threshold: color, text_color: [true, 0, 0]}\n",
         "ocr.preprocess.text_color must be [r, g, b] with values 0-255"),
        ("  preprocess: {threshold: color, text_color: white}\n",
         "ocr.preprocess.text_color must be [r, g, b] with values 0-255"),
        ("  preprocess: {threshold: otsu, text_color: [1, 2]}\n",
         "ocr.preprocess.text_color must be [r, g, b] with values 0-255"),
        ("  preprocess: {threshold: color, text_color: [1, 2, 3], text_tolerance: 256}\n",
         "ocr.preprocess.text_tolerance must be between 0 and 255"),
        ("  preprocess: {threshold: color, text_color: [1, 2, 3], text_tolerance: -1}\n",
         "ocr.preprocess.text_tolerance must be between 0 and 255"),
        ("  preprocess: {threshold: color, text_color: [1, 2, 3], text_tolerance: true}\n",
         "ocr.preprocess.text_tolerance must be an integer"),
    ],
)
def test_threshold_color_errors(tmp_path: Path, body: str, msg: str) -> None:
    e = err(tmp_path, body)
    assert e.startswith("cfg.yaml: ") or "cfg.yaml: " in e
    assert msg in e


# --------------------------------------------------------------------------
# ocr.undo
# --------------------------------------------------------------------------

PAT = '  patterns: {"^\\\\s*(\\\\d+)\\\\s*$": null}\n'


def test_undo_absent_means_none(tmp_path: Path) -> None:
    c = cfg_with(tmp_path, PAT)
    assert c.ocr is not None and c.ocr.undo is None
    c = cfg_with(tmp_path, "")
    assert c.ocr is not None and c.ocr.undo is None


def test_undo_defaults(tmp_path: Path) -> None:
    c = cfg_with(tmp_path, PAT + "  undo: {}\n")
    assert c.ocr is not None and c.ocr.undo is not None
    u = c.ocr.undo
    assert (u.settle_seconds, u.confirm_reads, u.confirm_seconds, u.window_seconds) == (3.0, 3, 1.0, 10.0)
    assert (u.min_confidence, u.require_baseline, u.baseline_seconds) == (80.0, True, 2.0)
    assert u.apply_to_last_split is False
    assert c.ocr.mode == "verify_only"


def test_undo_values_parse(tmp_path: Path) -> None:
    c = cfg_with(
        tmp_path,
        PAT + "  undo: {settle_seconds: 2, confirm_reads: 1, confirm_seconds: 0.25, window_seconds: 5,"
        + " min_confidence: 0, require_baseline: false, baseline_seconds: 0.5, apply_to_last_split: true}\n",
    )
    assert c.ocr is not None and c.ocr.undo is not None
    u = c.ocr.undo
    assert (u.settle_seconds, u.confirm_reads, u.confirm_seconds, u.window_seconds) == (2.0, 1, 0.25, 5.0)
    assert (u.min_confidence, u.require_baseline, u.baseline_seconds, u.apply_to_last_split) == (
        0.0, False, 0.5, True,
    )


@pytest.mark.parametrize(
    ("undo", "msg"),
    [
        ("{settle_seconds: 0}", "ocr.undo.settle_seconds must be greater than 0"),
        ("{settle_seconds: -1}", "ocr.undo.settle_seconds must be greater than 0"),
        ("{settle_seconds: .inf}", "ocr.undo.settle_seconds must be a finite number"),
        ("{settle_seconds: .nan}", "ocr.undo.settle_seconds must be a finite number"),
        ("{settle_seconds: x}", "ocr.undo.settle_seconds must be a number"),
        ("{settle_seconds: true}", "ocr.undo.settle_seconds must be a number"),
        ("{confirm_reads: 0}", "ocr.undo.confirm_reads must be at least 1"),
        ("{confirm_reads: true}", "ocr.undo.confirm_reads must be an integer"),
        ("{confirm_reads: 2.5}", "ocr.undo.confirm_reads must be an integer"),
        ("{confirm_seconds: -0.1}", "ocr.undo.confirm_seconds must be at least 0"),
        ("{confirm_seconds: .inf}", "ocr.undo.confirm_seconds must be a finite number"),
        ("{window_seconds: 4}", "ocr.undo.window_seconds must be greater than settle_seconds + confirm_seconds"),
        ("{window_seconds: .inf}", "ocr.undo.window_seconds must be a finite number"),
        ("{settle_seconds: 5, confirm_seconds: 5, window_seconds: 10}",
         "ocr.undo.window_seconds must be greater than settle_seconds + confirm_seconds"),
        ("{min_confidence: 101}", "ocr.undo.min_confidence must be between 0.0 and 100.0"),
        ("{min_confidence: -1}", "ocr.undo.min_confidence must be between 0.0 and 100.0"),
        ("{min_confidence: true}", "ocr.undo.min_confidence must be a number"),
        ("{require_baseline: 1}", "ocr.undo.require_baseline must be true or false"),
        ("{require_baseline: yes_}", "ocr.undo.require_baseline must be true or false"),
        ("{baseline_seconds: 0}", "ocr.undo.baseline_seconds must be greater than 0"),
        ("{baseline_seconds: .nan}", "ocr.undo.baseline_seconds must be a finite number"),
        ("{apply_to_last_split: 0}", "ocr.undo.apply_to_last_split must be true or false"),
        ("5", "ocr.undo must be a mapping"),
    ],
)
def test_undo_errors(tmp_path: Path, undo: str, msg: str) -> None:
    e = err(tmp_path, PAT + f"  undo: {undo}\n")
    assert msg in e and "cfg.yaml: " in e


def test_undo_requires_patterns(tmp_path: Path) -> None:
    assert "cfg.yaml: ocr.patterns must not be empty when ocr.undo is set" in err(
        tmp_path, "  undo: {}\n"
    )


def test_mode_undo_is_now_accepted_with_undo_section(tmp_path: Path) -> None:
    # Superseded by increment 4 (executor): `mode: undo` used to be rejected.
    c = cfg_with(tmp_path, PAT + "  mode: undo\n  undo: {}\n")
    assert c.ocr is not None and c.ocr.mode == "undo" and c.ocr.undo is not None


def test_mode_undo_without_undo_section_uses_defaults(tmp_path: Path) -> None:
    c = cfg_with(tmp_path, PAT + "  mode: undo\n")
    assert c.ocr is not None and c.ocr.mode == "undo"
    u = c.ocr.undo
    assert u is not None and u.recover_late_hud is True and u.recover_seconds == 10.0
    assert u.settle_seconds == 3.0


def test_verify_only_with_undo_stays_dry_run_mode(tmp_path: Path) -> None:
    c = cfg_with(tmp_path, PAT + "  undo: {}\n")
    assert c.ocr is not None and c.ocr.mode == "verify_only" and c.ocr.undo is not None


def test_verify_only_without_undo_has_no_undo(tmp_path: Path) -> None:
    c = cfg_with(tmp_path, PAT)
    assert c.ocr is not None and c.ocr.undo is None


@pytest.mark.parametrize("mode", ["resync", "3", "Undo", "[a]"])
def test_bad_mode_message(tmp_path: Path, mode: str) -> None:
    assert 'ocr.mode must be "verify_only" or "undo"' in err(tmp_path, PAT + f"  mode: {mode}\n")


def test_mode_undo_requires_patterns(tmp_path: Path) -> None:
    assert 'ocr.patterns must not be empty when ocr.mode is "undo"' in err(
        tmp_path, "  mode: undo\n"
    )


def test_recover_keys_parse(tmp_path: Path) -> None:
    c = cfg_with(tmp_path, PAT + "  mode: undo\n  undo: {recover_late_hud: false, recover_seconds: 4}\n")
    assert c.ocr is not None and c.ocr.undo is not None
    assert c.ocr.undo.recover_late_hud is False and c.ocr.undo.recover_seconds == 4.0


@pytest.mark.parametrize(
    ("undo", "msg"),
    [
        ("{recover_seconds: 0}", "ocr.undo.recover_seconds must be greater than 0"),
        ("{recover_seconds: -2}", "ocr.undo.recover_seconds must be greater than 0"),
        ("{recover_seconds: .inf}", "ocr.undo.recover_seconds must be a finite number"),
        ("{recover_seconds: .nan}", "ocr.undo.recover_seconds must be a finite number"),
        ("{recover_seconds: x}", "ocr.undo.recover_seconds must be a number"),
        ("{recover_seconds: true}", "ocr.undo.recover_seconds must be a number"),
        ("{recover_seconds: " + "9" * 400 + "}", "ocr.undo.recover_seconds must be a number"),
        ("{recover_late_hud: 1}", "ocr.undo.recover_late_hud must be true or false"),
    ],
)
def test_recover_errors(tmp_path: Path, undo: str, msg: str) -> None:
    e = err(tmp_path, PAT + f"  mode: undo\n  undo: {undo}\n")
    assert msg in e and "cfg.yaml: " in e


def test_mode_undo_requires_follow_timer(tmp_path: Path) -> None:
    e = err(tmp_path, PAT + "  mode: undo\n", follow=False)
    assert 'cfg.yaml: ocr.mode "undo" requires livesplit.follow_timer: true' in e


def test_mode_undo_with_only_resync_on_unsplit_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    base = BASE.replace("port: 16834}", "port: 16834, resync_on_unsplit: true}")
    p.write_text(base + "ocr:\n" + ROI + PAT + "  mode: undo\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="requires livesplit.follow_timer"):
        _ = load(p)


def test_mode_undo_with_follow_timer_is_accepted(tmp_path: Path) -> None:
    c = cfg_with(tmp_path, PAT + "  mode: undo\n", follow=True)
    assert c.ocr is not None and c.ocr.mode == "undo" and c.livesplit.follow_timer


def test_verify_only_dry_run_does_not_need_follow_timer(tmp_path: Path) -> None:
    c = cfg_with(tmp_path, PAT + "  undo: {}\n", follow=False)
    assert c.ocr is not None and c.ocr.mode == "verify_only" and c.ocr.undo is not None
    assert not c.livesplit.follow_timer


def test_other_errors_win_over_the_follow_timer_rule(tmp_path: Path) -> None:
    e = err(tmp_path, "  mode: undo\n", follow=False)  # no patterns AND no follow_timer
    assert "ocr.patterns must not be empty" in e
    e = err(tmp_path, PAT + "  mode: undo\n  undo: {settle_seconds: 0}\n", follow=False)
    assert "ocr.undo.settle_seconds" in e
