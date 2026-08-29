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

    If the template contains {datetime}, the attempt counter is global
    per game (matches any datetime in the filename).
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
    attempt = _next_attempt(partial, template)
    # Handle both {attempt} and {attempt:Nd} format specs.
    partial = re.sub(r"\{attempt(?::(\d+)d)?\}", lambda m: f"{attempt:0{int(m.group(1)) if m.group(1) else 4}d}", partial)

    return partial


def _next_attempt(resolved_with_attempt_literal: str, template: str) -> int:
    """Return the next attempt number (1-based) for the given path.

    ``resolved_with_attempt_literal`` has all variables resolved except
    {attempt} (or {attempt:Nd}), which is still literal.

    ``template`` is the original template (with {datetime} still literal)
    used to determine if the attempt counter should be global per game.

    Counts existing ``.mkv`` and ``.mp4`` files in the parent directory
    whose stem matches the pattern with {attempt} replaced by digits.

    If the template contains {datetime}, the attempt counter is global
    per game (matches any datetime in the filename).
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

    # If the template contains {datetime}, the datetime portion varies per run.
    # Make the regex match any datetime pattern (YYYYMMDD_HHMMSS) so the
    # attempt counter increments globally per game, not per datetime.
    if "{datetime}" in template:
        # The datetime in the resolved path is already expanded, so we need to
        # find where it is in the escaped stem and replace it with a wildcard.
        actual_datetime = re.search(r"\d{8}_\d{6}", resolved_with_attempt_literal)
        if actual_datetime:
            escaped_datetime = re.escape(actual_datetime.group())
            regex_stem = regex_stem.replace(escaped_datetime, r"\d{8}_\d{6}")

    count = 0
    for entry in directory.iterdir():
        if entry.suffix in (".mkv", ".mp4") and re.fullmatch(
            regex_stem, entry.stem
        ):
            count += 1
    return count + 1
