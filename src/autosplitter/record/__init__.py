"""Recording path resolution.

Resolves a path template (from the ``record:`` config section) into an
actual file path, filling variables like {game}, {datetime}, and {attempt}.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path


def resolve_path(template: str, config_path: str) -> str:
    """Resolve a recording path template to an actual file path.

    Supported variables:

    - ``{game}``       — config file stem (e.g. ``start-powerless``)
    - ``{date}``       — ``YYYYMMDD``
    - ``{time}``       — ``HHMMSS``
    - ``{datetime}``   — ``YYYYMMDD_HHMMSS``
    - ``{attempt}``    — auto-incremented from existing files
    - ``{attempt:Nd}`` — same, zero-padded to N digits (e.g. ``{attempt:04d}``)

    The attempt counter counts ``.mkv`` and ``.mp4`` files in the
    recordings directory that match the resolved pattern (with {attempt}
    replaced by a numeric wildcard).  It never resets.
    """
    now = datetime.now(tz=UTC)
    game = Path(config_path).stem

    # Resolve all variables except {attempt} first.
    partial = template
    partial = partial.replace("{game}", game)
    partial = partial.replace("{date}", now.strftime("%Y%m%d"))
    partial = partial.replace("{time}", now.strftime("%H%M%S"))
    partial = partial.replace("{datetime}", now.strftime("%Y%m%d_%H%M%S"))

    # Resolve {attempt}: count existing files in the recordings directory
    # that match the pattern (with {attempt} or {attempt:Nd} still literal).
    attempt = _next_attempt(partial)
    # Handle both {attempt} and {attempt:Nd} format specs.
    partial = re.sub(r"\{attempt(?::(\d+)d)?\}", lambda m: f"{attempt:0{int(m.group(1)) if m.group(1) else 4}d}", partial)

    return partial


def _next_attempt(resolved_with_attempt_literal: str) -> int:
    """Return the next attempt number (1-based) for the given path.

    ``resolved_with_attempt_literal`` has all variables resolved except
    {attempt} (or {attempt:Nd}), which is still literal.

    Counts existing ``.mkv`` and ``.mp4`` files in the parent directory
    whose stem matches the pattern with {attempt} replaced by digits.
    """
    path = Path(resolved_with_attempt_literal)
    directory = path.parent
    if not directory.exists():
        return 1

    # Check if {attempt} or {attempt:Nd} is in the path.
    attempt_match = re.search(r"\{attempt(?::\d+d)?\}", resolved_with_attempt_literal)
    if attempt_match is None:
        return 1

    # Build a regex: escape the stem, then replace the literal {attempt...}
    # with a digit-capturing group.
    stem = path.stem
    escaped = re.escape(stem)
    regex_stem = escaped.replace(
        re.escape(attempt_match.group()), r"(\d+)"
    )

    count = 0
    for entry in directory.iterdir():
        if entry.suffix in (".mkv", ".mp4") and re.fullmatch(
            regex_stem, entry.stem
        ):
            count += 1
    return count + 1
