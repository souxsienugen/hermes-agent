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


class TestBatchPreflightSimulatesTheBatch:
    """The pre-flight measures each op against the content the batch leaves behind.

    An op whose resulting size depends on an earlier op must be projected against THAT
    op's result, not against the file's pre-batch on-disk size: measuring every op
    against the pre-batch size both misses a breach that only the earlier op's growth
    creates (the breach then surfaces inside the write path, after the sibling lesson was
    applied and rolled back) and refuses a legal shrink-then-grow sequence.
    """

    def test_grow_then_breach_is_caught_before_the_lesson_is_rolled_back(self, isolate_skills):
        """op[0] grows the file; op[1] breaches ONLY because of that growth.

        Measured against the pre-batch size, op[1] looks legal (the file was under the
        cap before the batch ran), so the breach surfaces inside the write path — after
        op[0]'s lesson was applied and then rolled back, and the lesson is lost.
        """
        # Start under the cap, with two unique anchors so a patch can target either one.
        front = "---\nname: t\ndescription: A test skill\n---\n# Test Skill\n\n"
        start = front + "LESSONTARGET\n" + ("x" * 40000) + "GROWTARGET\n" + ("w" * 40000)
        start += "z" * (MAX_SKILL_CONTENT_CHARS - 100 - len(start))
        assert len(start) == MAX_SKILL_CONTENT_CHARS - 100
        assert json.loads(skill_manage(action="create", name="t", content=start))["success"]

        result = json.loads(skill_manage(action="", name="", operations=[
            # op[0] grows the file by 66 chars (still under the cap) and carries the lesson.
            {"name": "t", "action": "patch", "old_string": "LESSONTARGET",
             "new_string": "LESSONTARGET\n- lesson; why.\n" + ("y" * 50)},
            # op[1] adds 80 more: 99,980 against the PRE-batch size (under the cap, so
            # the old pre-flight missed it), 100,046 against op[0]'s result (over it).
            {"name": "t", "action": "patch", "old_string": "GROWTARGET",
             "new_string": "GROWTARGET" + ("z" * 80)},
        ]))
        assert result["success"] is False, result
        # The pre-flight named the op that actually breaches, and caught it BEFORE
        # anything was written: no sibling op was applied and then rolled back.
        assert "operations[1]" in result["error"]
        assert "rolled back" not in result["error"]
        content = (isolate_skills / "t" / "SKILL.md").read_text(encoding="utf-8")
        assert "lesson; why" not in content

    def test_shrink_then_grow_stays_legal(self, isolate_skills):
        """A legal shrink-then-grow sequence must NOT be refused.

        op[0] shrinks an over-cap file to ~40,000 chars; op[1] grows that result to
        ~40,500, still under the cap. Measured against the pre-batch (over-cap) size,
        op[1] looks like a growth of an over-cap file and is refused — breaking the
        migration sequence this ceiling exists to enable.
        """
        skill_dir = isolate_skills / "big-skill"
        skill_dir.mkdir(parents=True)
        front = "---\nname: big-skill\ndescription: A test skill\n---\n# Test Skill\n\n"
        over_cap = front + "SHRINKTARGET\n" + ("x" * 60000) + "GROWTARGET\n" + ("y" * 40000)
        (skill_dir / "SKILL.md").write_text(over_cap, encoding="utf-8")
        assert len(over_cap) > MAX_SKILL_CONTENT_CHARS

        result = json.loads(skill_manage(action="", name="", operations=[
            # op[0] shrinks it: the ceiling allows an over-cap file to be written smaller.
            {"name": "big-skill", "action": "patch",
             "old_string": "SHRINKTARGET\n" + ("x" * 60000), "new_string": "SHRUNK"},
            # op[1] grows the SHRUNK file: ~40,500 chars, well under the cap.
            {"name": "big-skill", "action": "patch", "old_string": "GROWTARGET",
             "new_string": "GROWTARGET" + ("q" * 500)},
        ]))
        assert result["success"] is True, result
        content = (isolate_skills / "big-skill" / "SKILL.md").read_text(encoding="utf-8")
        assert len(content) < MAX_SKILL_CONTENT_CHARS
        assert len(content) < len(over_cap)

    def test_replace_all_counts_every_match(self, isolate_skills):
        """A replace_all op grows the file by (matches) x len(new_string), not once.

        The old cheap bound ("the result only grows by the new text") under-counted a
        replace_all, so the pre-flight returned no error for an op that WOULD breach.
        """
        # 400 characters, each 'x' is a match: 400 x 400 = 160,000 chars.
        body = "# Test Skill\n\n" + ("x" * 400)
        content = f"---\nname: r\ndescription: A test skill\n---\n{body}"
        assert json.loads(skill_manage(action="create", name="r", content=content))["success"]

        result = json.loads(skill_manage(action="", name="", operations=[
            {"name": "r", "action": "patch", "old_string": "x", "new_string": "y" * 400,
             "replace_all": True},
        ]))
        assert result["success"] is False, result
        assert "operations[0]" in result["error"]
        # The PRE-FLIGHT caught it, before any op ran: the write path's own refusal
        # would name the op as FAILED and roll the batch back.
        assert "rolled back" not in result["error"]
        # 400 matches x 400 chars of new text: the result is 160,056 chars, not 800.
        assert "160,056 characters" in result["error"]

    def test_an_out_of_tree_path_is_refused_before_it_is_read(self, isolate_skills):
        """A path outside the skill tree must be refused WITHOUT reading the target.

        Reading it first turned the size pre-flight into a side channel: the refusal
        message carried the out-of-tree file's exact character count.
        """
        skill_dir = isolate_skills / "probe"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(_skill_content("probe", 500), encoding="utf-8")
        # A file OUTSIDE the skill tree, with a length the tool must never disclose.
        victim = isolate_skills.parent / "victim.txt"
        victim.write_text("v" * 123456, encoding="utf-8")

        result = json.loads(skill_manage(action="", name="", operations=[
            # The write is over the cap AND over the victim's length, so a pre-flight
            # that reads the victim first reports it as "already over the limit
            # (123,456 characters)" — disclosing a file outside the skill tree.
            {"name": "probe", "action": "write_file",
             "file_path": str(victim),
             "file_content": "x" * (MAX_SKILL_CONTENT_CHARS + 100000)},
        ]))
        assert result["success"] is False, result
        # The refusal names the path problem, and discloses NOTHING about the target's size.
        assert "123,456" not in result["error"]
        assert "characters" not in result["error"]
        assert victim.read_text(encoding="utf-8") == "v" * 123456

    def test_a_binary_path_returns_a_json_error_instead_of_raising(self, isolate_skills):
        """A binary file_path must yield a JSON error, not an uncaught UnicodeDecodeError."""
        skill_dir = isolate_skills / "probe"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(_skill_content("probe", 500), encoding="utf-8")
        blob = skill_dir / "references"
        blob.mkdir()
        (blob / "b.bin").write_bytes(b"\xff\xfe\x00binary")

        result = json.loads(skill_manage(action="", name="", operations=[
            {"name": "probe", "action": "patch", "file_path": "references/b.bin",
             "old_string": "binary", "new_string": "text"},
        ]))
        assert result["success"] is False, result
        assert "not valid UTF-8" in result["error"]


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
