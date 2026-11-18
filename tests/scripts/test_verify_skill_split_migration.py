"""scripts/verify_skill_split_migration.py: a split must not lose a section.

A split moves the bulk of an oversized SKILL.md into ``references/`` files and leaves
the SKILL.md as an index. A section dropped during the move is a lost lesson, and
nothing in the tool reports it: the split is applied as a sequence of writes, each of
which succeeds. The verifier is the only thing that notices a missing section, so it
has to notice a missing section - and it has to PASS a split that preserved everything.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "verify_skill_split_migration.py"
_SPEC = importlib.util.spec_from_file_location("_verify_skill_split_migration", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

ORIGINAL = """---
name: demo
description: A demo skill
---
# Demo Skill

The intro stays in the index.

## Section A

Body A, with the lesson that must survive the split.

## Section B

Body B, also a lesson.
"""


def _skill_tree(root: Path, index: str, refs: dict[str, str]) -> None:
    skill_dir = root / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(index, encoding="utf-8")
    if refs:
        (skill_dir / "references").mkdir()
        for name, body in refs.items():
            (skill_dir / "references" / name).write_text(body, encoding="utf-8")


def _originals(root: Path, text: str = ORIGINAL) -> Path:
    originals = root / "originals"
    originals.mkdir()
    (originals / "demo.SKILL.md").write_text(text, encoding="utf-8")
    return originals


def _run(originals: Path, skills_root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--originals", str(originals),
         "--skills-root", str(skills_root)],
        capture_output=True, text=True, check=False,
    )


def test_a_split_that_preserved_every_section_passes(tmp_path):
    """The whole point: a good split is green."""
    originals = _originals(tmp_path)
    _skill_tree(
        tmp_path,
        index="---\nname: demo\ndescription: A demo skill\n---\n# Demo Skill\n\n"
              "The intro stays in the index.\n\n## Section A\n\n"
              "See references/section-a.md.\n",
        refs={
            "section-a.md": "## Section A\n\nBody A, with the lesson that must survive the split.\n",
            "section-b.md": "## Section B\n\nBody B, also a lesson.\n",
        },
    )

    result = _run(originals, tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1/1 skills" in result.stdout


def test_a_split_that_dropped_a_section_fails(tmp_path):
    """A dropped section is a lost lesson: the verifier must FAIL it."""
    originals = _originals(tmp_path)
    _skill_tree(
        tmp_path,
        index="---\nname: demo\ndescription: A demo skill\n---\n# Demo Skill\n\n"
              "The intro stays in the index.\n",
        refs={},
    )

    result = _run(originals, tmp_path)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "FAIL demo" in result.stdout
    assert "Section A" in result.stdout


def test_a_section_preserved_across_a_reflowed_line_still_matches(tmp_path):
    """The comparison is whitespace-insensitive: a rewrapped body is the same body."""
    originals = _originals(tmp_path)
    _skill_tree(
        tmp_path,
        index="---\nname: demo\ndescription: A demo skill\n---\n# Demo Skill\n\n"
              "The intro stays in the index.\n\n## Section A\n\n"
              "Body A, with the lesson\nthat must survive the split.\n\n"
              "## Section B\n\nBody B, also a lesson.\n",
        refs={},
    )

    result = _run(originals, tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1/1 skills" in result.stdout


def test_a_skill_that_is_gone_from_the_tree_is_reported(tmp_path):
    """The skill was deleted: that is a failure, not a silent pass."""
    originals = _originals(tmp_path)

    result = _run(originals, tmp_path)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "no SKILL.md found" in result.stdout
