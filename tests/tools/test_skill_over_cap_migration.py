"""A SKILL.md already over the content ceiling must still be writable SMALLER.

The ceiling (MAX_SKILL_CONTENT_CHARS) refuses GROWTH past the limit. A file that drifted
PAST the limit — hand-placed, hub-installed, or grown by an older build — used to be frozen:
every write to it was refused, including the shrink that brings it back under, so the only
escape was one whole-file rewrite that had to land under the cap in a single shot. The
migration that remedy demands is a SEQUENCE of writes, and a sequence cannot start.

These tests pin the two halves of the fix:

1. an over-cap file may be shrunk (patch/edit/write_file), which un-freezes the migration;
2. an over-cap file may NOT be grown — the ceiling still means what it says.

Plus the batch pre-flight: a write that would breach the ceiling fails inside the write path,
by which time the batch has applied the ops before it and must roll them back, taking the
lesson they carried with them. The pre-flight rejects the batch BEFORE anything is written.
"""

import json

import pytest

from tools.skill_manager_tool import (
    MAX_SKILL_CONTENT_CHARS,
    _validate_content_size,
    skill_manage,
)


@pytest.fixture(autouse=True)
def isolate_skills(tmp_path, monkeypatch):
    """Redirect the skills dir to a temp directory.

    ``tools.skills_tool.SKILLS_DIR`` is the shared handle every tree has. The live-lineage
    tree ALSO carries a module-level ``tools.skill_manager_tool.SKILLS_DIR``, which it reads
    ahead of the shared one, so it is patched when present (``raising=False``) — the upstream
    tree dropped that module-level copy in favour of the shared handle.
    """
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", skills_dir)
    monkeypatch.setattr("tools.skill_manager_tool.SKILLS_DIR", skills_dir, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return skills_dir


def _skill_content(name: str, body_chars: int) -> str:
    """Valid SKILL.md text with a body of exactly ``body_chars`` characters (total length)."""
    frontmatter = f"---\nname: {name}\ndescription: A test skill\n---\n"
    body = "# Test Skill\n\n" + ("x" * max(0, body_chars - len(frontmatter) - len("# Test Skill\n\n")))
    return frontmatter + body


def _place_over_cap_skill(skills_dir, name: str) -> int:
    """Hand-place a SKILL.md over the ceiling (the tool cannot create one) -> its length.

    The body is one long run of 'x' so a patch can target it, but the run is BROKEN by a
    unique marker: a bare run of 'x' matches thousands of times and the fuzzy matcher
    refuses a non-unique old_string.
    """
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True)
    over_cap = _skill_content(name, MAX_SKILL_CONTENT_CHARS + 500)
    # Break the run so "x"*N is unique: one marker line, then the rest.
    marker = "OVERCAPMARKER\n"
    head, _, tail = over_cap.partition("x" * 50)
    over_cap = head + marker + "x" * 50 + tail
    (skill_dir / "SKILL.md").write_text(over_cap, encoding="utf-8")
    return len(over_cap)


class TestOverCapFileCanShrink:
    """The migration path: an over-cap file is writable SMALLER."""

    def test_patch_may_shrink_an_over_cap_file(self, isolate_skills):
        over_cap = _place_over_cap_skill(isolate_skills, "big-skill")
        assert over_cap > MAX_SKILL_CONTENT_CHARS

        # Replace the marker region with a shorter one: the result is still over the cap
        # (so the shrink is not a "land under the cap" rewrite) but it is SMALLER.
        result = json.loads(skill_manage(
            action="patch",
            name="big-skill",
            old_string="OVERCAPMARKER",
            new_string="M",
        ))
        assert result["success"] is True, result
        content = (isolate_skills / "big-skill" / "SKILL.md").read_text(encoding="utf-8")
        assert len(content) < over_cap

    def test_patch_may_shrink_an_over_cap_file_under_the_cap(self, isolate_skills):
        _place_over_cap_skill(isolate_skills, "big-skill")
        result = json.loads(skill_manage(
            action="patch",
            name="big-skill",
            old_string="OVERCAPMARKER",
            new_string="",
        ))
        assert result["success"] is True, result

    def test_full_rewrite_may_shrink_an_over_cap_file(self, isolate_skills):
        _place_over_cap_skill(isolate_skills, "big-skill")
        result = json.loads(skill_manage(
            action="edit",
            name="big-skill",
            content=_skill_content("big-skill", 500),
        ))
        assert result["success"] is True, result
        content = (isolate_skills / "big-skill" / "SKILL.md").read_text(encoding="utf-8")
        assert len(content) < MAX_SKILL_CONTENT_CHARS

    def test_supporting_file_may_shrink_an_over_cap_file(self, isolate_skills):
        skill_dir = isolate_skills / "big-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(_skill_content("big-skill", 500), encoding="utf-8")
        refs = skill_dir / "references"
        refs.mkdir()
        (refs / "bulk.md").write_text("z" * (MAX_SKILL_CONTENT_CHARS + 500), encoding="utf-8")

        result = json.loads(skill_manage(
            action="write_file",
            name="big-skill",
            file_path="references/bulk.md",
            file_content="z" * 100,
        ))
        assert result["success"] is True, result


class TestOverCapFileMayNotGrow:
    """The ceiling still binds: an over-cap file may not be grown."""

    def test_patch_may_not_grow_an_over_cap_file(self, isolate_skills):
        _place_over_cap_skill(isolate_skills, "big-skill")
        result = json.loads(skill_manage(
            action="patch",
            name="big-skill",
            old_string="OVERCAPMARKER",
            new_string="OVERCAPMARKER" + ("y" * 5000),
        ))
        assert result["success"] is False
        assert "already over the limit" in result["error"]

    def test_a_file_at_the_cap_still_may_not_grow(self, isolate_skills):
        """The ratchet is for files ALREADY over the cap; one at the cap is unchanged."""
        content = _skill_content("edge-skill", MAX_SKILL_CONTENT_CHARS)
        assert len(content) == MAX_SKILL_CONTENT_CHARS
        result = json.loads(skill_manage(action="create", name="edge-skill", content=content))
        assert result["success"] is True

        result = json.loads(skill_manage(
            action="patch", name="edge-skill",
            old_string="# Test Skill", new_string="# Test Skill\n" + ("y" * 200),
        ))
        assert result["success"] is False
        assert "characters (limit" in result["error"]


class TestBatchSizePreflight:
    """A batch that would breach the ceiling is rejected BEFORE anything is written."""

    def test_oversized_op_rejects_the_batch_before_siblings_apply(self, isolate_skills):
        small = _skill_content("probe", 500)
        assert json.loads(skill_manage(action="create", name="probe", content=small))["success"]

        result = json.loads(skill_manage(action="", name="", operations=[
            # op[0] would succeed on its own and carries the lesson.
            {"name": "probe", "action": "patch",
             "old_string": "# Test Skill", "new_string": "# Test Skill\n- lesson; why.\n"},
            # op[1] would breach the ceiling.
            {"name": "probe", "action": "write_file",
             "file_path": "references/huge.md",
             "file_content": "x" * (MAX_SKILL_CONTENT_CHARS + 1)},
        ]))
        assert result["success"] is False
        # The pre-flight names the offending op, so the caller knows which one to fix.
        assert "operations[1]" in result["error"]
        # The batch was rejected BEFORE anything was written, so no sibling op was applied and
        # then rolled back — the lesson in op[0] was never at risk.
        assert "rolled back" not in result["error"]
        content = (isolate_skills / "probe" / "SKILL.md").read_text(encoding="utf-8")
        assert "lesson; why" not in content
        assert not (isolate_skills / "probe" / "references" / "huge.md").exists()

    def test_a_batch_within_the_ceiling_still_applies(self, isolate_skills):
        small = _skill_content("probe", 500)
        assert json.loads(skill_manage(action="create", name="probe", content=small))["success"]

        result = json.loads(skill_manage(action="", name="", operations=[
            {"name": "probe", "action": "patch",
             "old_string": "# Test Skill", "new_string": "# Test Skill\n- lesson; why.\n"},
            {"name": "probe", "action": "write_file",
             "file_path": "references/a.md", "file_content": "# A\n\ntext"},
        ]))
        assert result["success"] is True, result
        assert result["operations_applied"] == 2


class TestValidateContentSizeCurrentLen:
    """The unit behind the ratchet."""

    def test_within_limit_passes(self):
        assert _validate_content_size("a" * 1000) is None

    def test_over_limit_without_a_current_length_is_refused(self):
        err = _validate_content_size("a" * (MAX_SKILL_CONTENT_CHARS + 1))
        assert err is not None
        assert "characters (limit" in err

    def test_over_limit_that_shrinks_an_over_cap_file_passes(self):
        current = MAX_SKILL_CONTENT_CHARS + 500
        assert _validate_content_size("a" * current, current_len=current) is None

    def test_over_limit_that_grows_an_over_cap_file_is_refused(self):
        current = MAX_SKILL_CONTENT_CHARS + 500
        err = _validate_content_size("a" * (current + 1), current_len=current)
        assert err is not None
        assert "already over the limit" in err

    def test_a_shrunk_write_under_the_cap_passes(self):
        assert _validate_content_size("a" * 10, current_len=MAX_SKILL_CONTENT_CHARS + 500) is None
