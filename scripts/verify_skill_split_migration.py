#!/usr/bin/env python3
"""Verify that a SKILL.md split preserved every section of the original file.

A split moves the bulk of an oversized SKILL.md into ``references/`` files and leaves
the SKILL.md as an index. The split is only safe if EVERY section of the original is
still present, somewhere, verbatim - a section that is dropped during the move is a
lost lesson, and nothing in the tool reports it.

This script checks that invariant against a directory of pre-migration originals:

    scripts/verify_skill_split_migration.py \\
        --originals <dir> --skills-root <dir> [--manifest <file>]

``--originals`` holds one ``<skill-name>.SKILL.md`` per migrated skill (the
pre-migration body). For each one, the skill's CURRENT directory is located under
``--skills-root`` (a profile's skills tree is searched too), and every markdown
section of the original must appear in the SKILL.md index or in one of its
``references/`` files.

Exit code 0 when every section of every skill is found, 1 otherwise. With
``--verbose`` the missing sections are listed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# A markdown heading: '#'..'######', with a space and a title.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def _sections(text: str) -> list[tuple[str, str]]:
    """``(heading, body)`` for every markdown section, in file order.

    The preamble before the first heading is section ``""`` (the frontmatter block).
    """
    out: list[tuple[str, str]] = []
    marks = [(m.start(), m.group(1), m.group(2)) for m in _HEADING_RE.finditer(text)]
    if not marks:
        return [("", text)]
    if marks[0][0] > 0:
        out.append(("", text[: marks[0][0]]))
    for i, (start, _level, title) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        out.append((title, text[start:end]))
    return out


def _normalize(text: str) -> str:
    """The comparison form: whitespace-collapsed, so a reflowed line still matches."""
    return re.sub(r"\s+", " ", text).strip()


def _skill_dirs(skills_root: Path) -> dict[str, list[Path]]:
    """``skill name -> dirs`` found under ``skills_root`` (a profile tree included)."""
    found: dict[str, list[Path]] = {}
    for skill_md in skills_root.rglob("SKILL.md"):
        found.setdefault(skill_md.parent.name, []).append(skill_md.parent)
    return found


def _bodies(skill_dir: Path) -> str:
    """The skill's current content: the SKILL.md index plus every reference file."""
    parts: list[str] = []
    index = skill_dir / "SKILL.md"
    if index.exists():
        parts.append(index.read_text(encoding="utf-8-sig", errors="replace"))
    refs = skill_dir / "references"
    if refs.is_dir():
        for ref in sorted(refs.glob("*.md")):
            parts.append(ref.read_text(encoding="utf-8-sig", errors="replace"))
    return "\n".join(parts)


def verify(originals: Path, skills_root: Path) -> tuple[int, int, list[str]]:
    """-> ``(skills checked, skills whose sections are all present, problems)``."""
    dirs = _skill_dirs(skills_root)
    checked = 0
    ok = 0
    problems: list[str] = []
    for original in sorted(originals.glob("*.SKILL.md")):
        name = original.name[: -len(".SKILL.md")]
        checked += 1
        candidates = dirs.get(name)
        if not candidates:
            problems.append(f"{name}: no SKILL.md found under {skills_root}")
            continue
        # A name can exist in more than one profile; a skill passes when ANY of them
        # carries every section (the split happened in exactly one of them).
        best: list[str] = []
        for skill_dir in candidates:
            current = _normalize(_bodies(skill_dir))
            missing = [
                title for title, body in _sections(original.read_text(encoding="utf-8-sig"))
                if _normalize(body) and _normalize(body) not in current
            ]
            if not missing:
                best = []
                break
            if len(missing) < len(best) or not best:
                best = missing
        if best:
            problems.append(f"{name}: {len(best)} section(s) missing: {', '.join(best[:5])}")
        else:
            ok += 1
    return checked, ok, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--originals", required=True, type=Path,
                        help="directory of pre-migration <skill-name>.SKILL.md files")
    parser.add_argument("--skills-root", required=True, type=Path,
                        help="skills tree to search for the migrated skills")
    parser.add_argument("--manifest", type=Path,
                        help="optional MANIFEST.json to print the backup provenance")
    parser.add_argument("--verbose", action="store_true", help="list the missing sections")
    args = parser.parse_args(argv)

    if not args.originals.is_dir():
        print(f"not a directory: {args.originals}", file=sys.stderr)
        return 2
    if not args.skills_root.is_dir():
        print(f"not a directory: {args.skills_root}", file=sys.stderr)
        return 2

    if args.manifest and args.manifest.is_file():
        for entry in json.loads(args.manifest.read_text(encoding="utf-8-sig")):
            print(f"  {entry['chars']:>9,} chars  {entry['name']}  sha256={entry['sha256'][:16]}")

    checked, ok, problems = verify(args.originals, args.skills_root)
    print(f"\n{ok}/{checked} skills: every original section is present")
    for problem in problems:
        print(f"  FAIL {problem}")
    return 0 if ok == checked and checked else 1


if __name__ == "__main__":
    raise SystemExit(main())
