#!/usr/bin/env python3
"""Copy segment names from a LiveSplit .lss file into a config's ``splits:``.

stdlib only. The config is edited line by line, so comments, ordering and
every other key stay exactly as written; only the ``name:`` value of each
split changes.

    uv run python scripts/sync_split_names.py configs/mr_rich.yaml run.lss
    uv run python scripts/sync_split_names.py configs/mr_rich.yaml run.lss --dry-run

The number of segments in the .lss must equal the number of splits in the
config (nothing is written otherwise). Segment names are copied verbatim,
including subsplit markers such as ``-Name`` or ``{Group}Name``, because the
desync check compares against the name LiveSplit reports.
Exit code 0 = ok, 1 = error.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_TOP_KEY = re.compile(r"^[A-Za-z_][\w-]*\s*:")
_ITEM = re.compile(r"^(\s*)-\s+(\w+)\s*:")
_NAME = re.compile(r"^(?P<head>\s*(?:-\s+)?name\s*:\s*)(?P<value>.*)$")


def read_lss_names(path: Path) -> list[str]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise SystemExit(f"{path}: cannot read .lss: {exc}")
    names = [(seg.findtext("Name") or "").strip() for seg in root.findall("./Segments/Segment")]
    if not names:
        raise SystemExit(f"{path}: no <Segments>/<Segment> entries found")
    return names


def _value_end(value: str) -> int:
    """Index where the scalar ends and any trailing ``  # comment`` begins."""
    if value[:1] in ('"', "'"):
        quote = value[0]
        i = 1
        while i < len(value):
            if quote == '"' and value[i] == "\\":
                i += 2
                continue
            if value[i] == quote:
                if quote == "'" and value[i + 1 : i + 2] == "'":
                    i += 2
                    continue
                return i + 1
            i += 1
        return len(value)
    m = re.search(r"\s+#", value)
    return m.start() if m else len(value.rstrip())


def update_config_text(text: str, names: list[str]) -> tuple[str, list[tuple[str, str]]]:
    """Return (new text, [(old, new), ...]) for every split's name."""
    lines = text.splitlines(keepends=True)
    start = next((i for i, ln in enumerate(lines) if re.match(r"^splits\s*:", ln)), None)
    if start is None:
        raise SystemExit("config has no top-level 'splits:' section")
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if _TOP_KEY.match(lines[i]):
            end = i
            break

    # Locate the line index of each list item and of its name: line.
    item_indent: int | None = None
    items: list[int] = []
    for i in range(start + 1, end):
        m = _ITEM.match(lines[i])
        if m and (item_indent is None or len(m.group(1)) == item_indent):
            item_indent = len(m.group(1))
            items.append(i)
    name_lines: list[int | None] = []
    for n, first in enumerate(items):
        stop = items[n + 1] if n + 1 < len(items) else end
        name_lines.append(next((j for j in range(first, stop) if _NAME.match(lines[j])), None))

    if len(items) != len(names):
        raise SystemExit(
            f"config has {len(items)} splits but the .lss has {len(names)} segments; "
            "nothing written"
        )
    missing = [n + 1 for n, j in enumerate(name_lines) if j is None]
    if missing:
        raise SystemExit(f"splits {missing} have no 'name:' line; nothing written")

    changes: list[tuple[str, str]] = []
    for j, new in zip(name_lines, names):
        assert j is not None
        line = lines[j]
        eol = line[len(line.rstrip("\r\n")) :]
        m = _NAME.match(line.rstrip("\r\n"))
        assert m is not None
        value = m.group("value")
        end_idx = _value_end(value)
        old_raw = value[:end_idx]
        try:
            old = json.loads(old_raw) if old_raw[:1] == '"' else old_raw.strip("'")
        except ValueError:
            old = old_raw
        changes.append((str(old), new))
        lines[j] = m.group("head") + json.dumps(new, ensure_ascii=False) + value[end_idx:] + eol
    return "".join(lines), changes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path, help="per-game YAML config to update")
    ap.add_argument("lss", type=Path, help="LiveSplit .lss splits file")
    ap.add_argument("--dry-run", action="store_true", help="show the changes without writing")
    args = ap.parse_args()

    names = read_lss_names(args.lss)
    try:
        text = args.config.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"{args.config}: {exc}")
    new_text, changes = update_config_text(text, names)

    for n, (old, new) in enumerate(changes, 1):
        print(f"  {n:>2}  {old!r} (unchanged)" if old == new else f"  {n:>2}  {old!r} -> {new!r}")
    changed = sum(old != new for old, new in changes)
    if args.dry_run or changed == 0:
        print(f"{changed} name(s) would change" if args.dry_run else "already in sync")
        return 0
    args.config.write_text(new_text, encoding="utf-8")
    print(f"{args.config}: updated {changed} name(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
