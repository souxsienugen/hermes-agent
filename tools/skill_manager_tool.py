#!/usr/bin/env python3
"""Skill Manager Tool — agent-managed skill creation & editing.

Skills are the agent's procedural memory (narrow "how to do X"; MEMORY.md/USER.md are
broad, declarative). New skills land in ~/.hermes/skills/ (or ``skills.create_dir``);
existing skills (bundled, hub, user) are modified in place. Layout:
``<skills>/[category/]<skill>/SKILL.md`` + optional ``references/ templates/ scripts/ assets/``.
"""

import contextvars as _ctxvars
import hashlib
import json
from contextlib import ExitStack, suppress
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hermes_yaml as yaml

from hermes_constants import get_hermes_home
from utils import atomic_write_text, is_truthy_value
from hermes_cli.config import cfg_get
from agent.skill_utils import (
    extract_skill_description,
    is_skill_description_truncated_for_prompt,
    parse_frontmatter as _parse_frontmatter,
    SKILL_PROMPT_DESC_LIMIT)
from tools.skill_manager_guards import (
    _background_review_delete_guard, _background_review_preflight, _background_review_read_before_write_guard,
    _containing_skills_root, _curator_consolidation_delete_guard, _is_path_redirect, _pinned_guard,
    _validate_delete_target, _is_background_review, _refusal as _err)
from tools.skill_manager_batch import (
    _PATCH_EITHER_OR, _PATCH_NEEDS_NEW_STRING, _PATCH_NEEDS_OLD_STRING, _batch_target_key,
    _op_shape_error, _skill_manage_batch)
from tools.skills_guard import scan_skill, should_allow_install, format_scan_report

logger = logging.getLogger(__name__)


def _guard_agent_created_enabled() -> bool:
    """skills.guard_agent_created (default False): opt-in — terminal() runs the same code ungated."""
    try:
        from hermes_cli.config import load_config
        return is_truthy_value(cfg_get(load_config(), "skills", "guard_agent_created"), default=False)
    except Exception:
        return False


def _security_scan_skill(skill_dir: Path) -> Optional[str]:
    """Post-write scan (opt-in); error string if blocked, else None. An "ask" verdict
    (dangerous findings) is surfaced as an error so the agent can retry without them."""
    if not _guard_agent_created_enabled():
        return None
    try:
        result = scan_skill(skill_dir, source="agent-created")
        allowed, reason = should_allow_install(result)
        if allowed is None:
            logger.warning("Agent-created skill blocked (dangerous findings): %s", reason)
        if allowed is not True:
            return f"Security scan blocked this skill ({reason}):\n{format_scan_report(result)}"
    except Exception as e:
        logger.warning("Security scan failed for %s: %s", skill_dir, e, exc_info=True)
    return None


# All skills live in ~/.hermes/skills/ (single source of truth)
HERMES_HOME = get_hermes_home()
SKILLS_DIR = HERMES_HOME / "skills"
_SKILLS_DIR_AT_IMPORT = SKILLS_DIR


def _skills_dir() -> Path:
    """Active profile's skills dir at call time (multi-profile runtimes rebind per session).
    An explicitly patched module-level ``SKILLS_DIR`` (tests) wins over the live HERMES_HOME.

    Long-lived multi-profile runtimes (Dashboard/TUI/Desktop backend, cron, kanban workers) import this
    module once under the launch HERMES_HOME and later bind a different profile per session (#40677).
    """
    configured = Path(SKILLS_DIR)
    return configured if configured != _SKILLS_DIR_AT_IMPORT else get_hermes_home() / "skills"


def _skill_lock_path(name: str) -> Path:
    """Per-skill lock file under ``<skills>/.locks/`` (same idiom as the usage ledger's
    ``.usage.json.lock``), never inside the skill dir so delete/recreate cannot unlink it under a
    waiting writer. Keyed by a digest of the basename so ``foo`` and ``category/foo`` share one
    lock and no name can hit a filesystem limit (callers validate the basename first)."""
    digest = hashlib.sha256(Path(name).name.encode("utf-8", "surrogatepass")).hexdigest()
    return _skills_dir() / ".locks" / f"{digest}.lock"


def _skill_mutation_lock(name: str):
    """Exclusive lock held across one skill's whole read-modify-write; thread-re-entrant."""
    from tools.skill_usage import skill_file_lock
    return skill_file_lock(_skill_lock_path(name))


def _skill_mutation_locks(names):
    """Every lock of an atomic batch, acquired in one stable path order (deadlock-free across batches)."""
    from tools.skill_usage import skill_file_lock
    stack = ExitStack()
    for lock_path in sorted({_skill_lock_path(n) for n in names}):
        stack.enter_context(skill_file_lock(lock_path))
    return stack


MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
MAX_SKILL_CONTENT_CHARS = 100_000   # ~36k tokens at 2.75 chars/token
MAX_SKILL_FILE_BYTES = 1_048_576    # 1 MiB per supporting file
VALID_NAME_RE = re.compile(r'^[a-z0-9][a-z0-9._-]*$')  # filesystem-safe, URL-friendly
ALLOWED_SUBDIRS = {"references", "templates", "scripts", "assets"}  # for write_file/remove_file
_FRONTMATTER_END_RE = re.compile(r'\n---\s*\n')
_NAME_RULE = "Use lowercase letters, numbers, hyphens, dots, and underscores."


# --- Validation helpers -------------------------------------------------------

def _check_identifier(value: str, label: str, invalid: str) -> Optional[str]:
    if len(value) > MAX_NAME_LENGTH:
        return f"{label} exceeds {MAX_NAME_LENGTH} characters."
    return None if VALID_NAME_RE.match(value) else invalid


def _validate_name(name: str) -> Optional[str]:
    if not name:
        return "Skill name is required."
    return _check_identifier(
        name, "Skill name", f"Invalid skill name '{name}'. {_NAME_RULE} Must start with a letter or digit.")


def _validate_category(category: Optional[str]) -> Optional[str]:
    if category is None or (isinstance(category, str) and not category.strip()):
        return None
    if not isinstance(category, str):
        return "Category must be a string."
    category = category.strip()
    invalid = (f"Invalid category '{category}'. {_NAME_RULE} "
               "Categories must be a single directory name.")
    if "/" in category or "\\" in category:
        return invalid
    return _check_identifier(category, "Category", invalid)


def _validate_frontmatter(content: str) -> Optional[str]:
    """Validate frontmatter (name + description) and a non-empty body."""
    if not content.strip():
        return "Content cannot be empty."
    content = content.lstrip("\ufeff")  # tolerate a Windows UTF-8 BOM
    if not content.startswith("---"):
        return "SKILL.md must start with YAML frontmatter (---). See existing skills for format."
    end_match = _FRONTMATTER_END_RE.search(content[3:])
    if not end_match:
        return "SKILL.md frontmatter is not closed. Ensure you have a closing '---' line."
    try:
        parsed = yaml.safe_load(content[3:end_match.start() + 3])
    except yaml.YAMLError as e:
        return f"YAML frontmatter parse error: {e}"
    if not isinstance(parsed, dict):
        return "Frontmatter must be a YAML mapping (key: value pairs)."
    for field in ("name", "description"):
        if field not in parsed:
            return f"Frontmatter must include '{field}' field."
    desc = str(parsed["description"])
    if len(desc) > MAX_DESCRIPTION_LENGTH:
        return f"Description exceeds {MAX_DESCRIPTION_LENGTH} characters."
    if not content[end_match.end() + 3:].strip():
        return "SKILL.md must have content after the frontmatter (instructions, procedures, etc.)."
    return None


def _description_budget_error(content: str, before: Optional[str] = None) -> Optional[str]:
    """A description written by this call (a create, or an edit/patch that changes it from the one in
    ``before``) must fit SKILL_PROMPT_DESC_LIMIT, the width the skill index shows. An unchanged legacy
    over-limit description stays editable: a body patch must not be blocked by an old frontmatter."""
    desc = _description_of(content) or ""
    if len(desc) <= SKILL_PROMPT_DESC_LIMIT or (before is not None and desc == _description_of(before)):
        return None
    return (
        f"Description is {len(desc)} chars; a skill description must fit the "
        f"{SKILL_PROMPT_DESC_LIMIT}-char skill index (one sentence, trigger first, ends with a "
        f"period, without repeating the category name). The index clips anything longer. "
        f"Move detail into the skill body.")


def _current_file_len(path: Path) -> Optional[int]:
    """Character length of an existing file, or None when it does not exist or cannot be read."""
    try:
        return len(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError):
        return None


def _contained_file_len(skill_dir: Path, file_path: str | None = None) -> Optional[int]:
    """Character length of the file inside ``skill_dir`` a write would replace, or None.

    A target that does not resolve inside the skill dir is NEVER read: its length is not ours
    to disclose (the size pre-flight must not become a side channel for the size of a file
    outside the tree), and the write path refuses such a path anyway.
    """
    from tools.path_security import validate_within_dir
    target = skill_dir / (file_path or "SKILL.md")
    if validate_within_dir(target, skill_dir):
        return None
    return _current_file_len(target)


def _existing_content_len(name: str, file_path: str | None = None) -> Optional[int]:
    """Character length of the file a write to ``name`` would replace, or None when there is none.

    The ceiling refuses GROWTH past the limit, so a file that already sits over it may still be
    written SMALLER; deciding that needs the length of what is being replaced, not the length
    of the write. None means the write creates the file, so the ceiling applies at full strength.
    """
    existing = _find_skill(name)
    if not existing:
        return None
    return _contained_file_len(existing["path"], file_path)


def _content_size_error(content: str, label: str, name: str,
                         file_path: str | None = None) -> Optional[str]:
    """Size check for a write that REPLACES ``name``'s ``file_path`` (SKILL.md when None).

    The current length is read only when the write is actually over the limit, so a write
    within the limit never pays for the skill-tree lookup.
    """
    if len(content) <= MAX_SKILL_CONTENT_CHARS:
        return None
    return _validate_content_size(content, label=label,
                                   current_len=_existing_content_len(name, file_path))


def _file_path_refusal(skill_dir: Path, file_path: str) -> Optional[str]:
    """The refusal for a path that may not be written inside ``skill_dir``, or None.

    One definition of the path rules, shared by ``_resolve_supporting_file`` (which writes)
    and the batch size pre-flight (which only measures): a path is validated BEFORE the
    file behind it is read, because the length of a file outside the skill tree is not
    ours to disclose.
    """
    from tools.path_security import validate_within_dir
    target = skill_dir / (file_path or "")
    return _validate_file_path(file_path) or validate_within_dir(target, skill_dir)


def _not_text_error(label: str) -> str:
    """The refusal for a target that is not UTF-8 text (a binary file)."""
    return f"File is not valid UTF-8 text: {label}. skill_manage edits text files only."


class _BatchSizeProjector:
    """Projects the size of every file a batch writes, op by op.

    The pre-flight must SIMULATE the batch: an op's result depends on the ops before it,
    so every op is measured against the content the batch would have at that point, not
    against the file's on-disk size before the batch started. Measuring every op against
    the pre-batch size mis-projects every op whose result depends on an earlier one: a
    batch that grows a file and then breaches it is NOT caught (the breach surfaces inside
    the write path, after the sibling op was applied and then rolled back, so the lesson
    it carried is gone with it), and a legal shrink-then-grow batch is refused even though
    the shrink left room for the growth.

    A targeted patch's result depends on the fuzzy match, so it is computed the same way
    the write path computes it - against the projected content, not the on-disk one. The
    fuzzy result is always computed: the old cheap bound ("the result only grows by the
    new text") holds for a single match only, and with ``replace_all`` the result grows by
    (matches) x len(new_string), so the bound under-counted and missed the breach.
    """

    def __init__(self) -> None:
        self._files: dict[tuple[str, str], Optional[str]] = {}  # (skill, target) -> projected content
        self._dirs: dict[str, Optional[Path]] = {}               # skill -> dir (None: no such skill)
        self._halted = False                                     # an op whose outcome is unknown

    def error(self, nm: str, act: str, args: dict) -> Optional[str]:
        """The size error this op would produce, or None when it stays within the ceiling."""
        if act == "create":
            return self._create_error(nm, args)
        if act == "write_file":
            return self._write_file_error(nm, args)
        if act == "patch":
            return self._patch_error(nm, args)
        if act == "remove_file":
            # The file is gone for the ops that follow: an op on it fails in the write
            # path, so nothing after THAT op runs either.
            self._files[(nm, _batch_target_key(act, args))] = None
        return None

    def halted(self) -> bool:
        """True once an op's outcome is unknown, so no later op's size can be projected."""
        return self._halted

    def _create_error(self, nm: str, args: dict) -> Optional[str]:
        """The size error a create would produce."""
        content = args.get("content") or ""
        if _validate_frontmatter(content):
            # The write path refuses this create on its frontmatter and reports the
            # better error, so no op after it runs either.
            self._halted = True
            return None
        err = _validate_content_size(content)
        if err is not None:
            return err
        if self._dir_of(nm) is not None:
            # The skill already exists: the write path refuses this create, so no op
            # after it runs either.
            self._halted = True
            return None
        # The skill does not exist yet: create puts it HERE, so the ops that follow
        # resolve their paths against this dir.
        self._dirs[nm] = _resolve_skill_dir(nm, args.get("category"))
        self._files[(nm, "SKILL.md")] = content
        return None

    def _write_file_error(self, nm: str, args: dict) -> Optional[str]:
        """The size error a supporting-file write would produce."""
        file_path = args.get("file_path") or ""
        content = args.get("file_content") or ""
        key = (nm, _batch_target_key("write_file", args))
        skill_dir = self._dir_of(nm)
        if self._refused_path(skill_dir, file_path):
            # The write path refuses this path BEFORE it reads the target. The pre-flight
            # must not read it either: the length of a file outside the skill tree is not
            # ours to disclose.
            self._halted = True
            return None
        current_len = self._current_len(nm, key, skill_dir, file_path)
        err = _validate_content_size(content, label=file_path, current_len=current_len)
        self._files[key] = content
        return err

    def _patch_error(self, nm: str, args: dict) -> Optional[str]:
        """The size error a patch would produce."""
        content = args.get("content")
        file_path = args.get("file_path") or ""
        key = (nm, _batch_target_key("patch", args))
        skill_dir = self._dir_of(nm)
        if content:
            # A full rewrite of SKILL.md: the result is the given text. The write path
            # checks the frontmatter FIRST, so a broken one is its error, not a size error.
            if _validate_frontmatter(content):
                self._halted = True
                return None
            current_len = self._current_len(nm, key, skill_dir, "SKILL.md")
            err = _validate_content_size(content, label="SKILL.md", current_len=current_len)
            self._files[key] = content
            return err
        if file_path and self._refused_path(skill_dir, file_path):
            # The write path refuses this path BEFORE it reads the target. The pre-flight
            # must not read it either: the length of a file outside the skill tree is not
            # ours to disclose.
            self._halted = True
            return None
        # A targeted patch: the result is the file with one region replaced.
        current, state = self._target_content(nm, key, skill_dir, file_path)
        if state == "binary":
            # A binary file is not text: the write path refuses it, so the pre-flight
            # refuses it too instead of letting a UnicodeDecodeError escape the tool.
            return _not_text_error(file_path or "SKILL.md")
        if state != "ok":
            # The target does not exist (or cannot be read): the write path reports that,
            # and no op after it runs.
            self._halted = True
            return None
        new_string = args.get("new_string")
        if new_string is None:
            self._halted = True
            return None
        from tools.fuzzy_match import fuzzy_find_and_replace
        new_content, _count, _strategy, match_error = fuzzy_find_and_replace(
            current, args.get("old_string") or "", new_string, bool(args.get("replace_all")))
        if match_error:
            # The write path reports the match failure, which is the better error; this
            # op's result is unknown, so the ops after it are not projected either.
            self._halted = True
            return None
        # The write path checks the size BEFORE the frontmatter, so a size breach is the
        # error it reports; a broken frontmatter is its error only when the size is fine.
        err = _validate_content_size(new_content, label=file_path or "SKILL.md",
                                     current_len=len(current))
        if err is not None:
            return err
        if not file_path and _validate_frontmatter(new_content):
            self._halted = True
            return None
        self._files[key] = new_content
        return None

    def _refused_path(self, skill_dir: Optional[Path], file_path: str) -> bool:
        """True when the write path would refuse this path before it reads anything."""
        if _validate_file_path(file_path):
            return True
        return skill_dir is not None and bool(_file_path_refusal(skill_dir, file_path))

    def _dir_of(self, nm: str) -> Optional[Path]:
        """The dir a skill's files live in, or None when the skill does not exist."""
        if nm not in self._dirs:
            existing = _find_skill(nm)
            self._dirs[nm] = existing["path"] if existing else None
        return self._dirs[nm]

    def _current_len(self, nm: str, key: tuple[str, str], skill_dir: Optional[Path],
                     file_path: str) -> Optional[int]:
        """The length of the file an op replaces: the projected one, else the on-disk one."""
        if key in self._files:
            content = self._files[key]
            return len(content) if content is not None else None
        if skill_dir is None:
            return None  # the skill does not exist: the write creates the file
        return _contained_file_len(skill_dir, file_path)

    def _target_content(self, nm: str, key: tuple[str, str], skill_dir: Optional[Path],
                        file_path: str) -> tuple[str, str]:
        """The content of the file an op patches, and its state.

        State is "ok" (the content is known), "binary" (not UTF-8 text) or "absent" (the
        write path will refuse the op, so nothing after it can be projected). The content
        is "" when the state is not "ok".
        """
        if key in self._files:
            content = self._files[key]
            return (content, "ok") if content is not None else ("", "absent")
        if skill_dir is None:
            return "", "absent"
        target = skill_dir / (file_path or "SKILL.md")
        try:
            if not target.exists():
                return "", "absent"
            return target.read_text(encoding="utf-8-sig"), "ok"
        except UnicodeDecodeError:
            return "", "binary"
        except OSError:
            return "", "absent"


def _batch_size_error(operations, names) -> Optional[tuple[int, str]]:
    """The first op whose projected size breaches the ceiling -> ``(index, message)``, or None.

    The projection SIMULATES the batch (see ``_BatchSizeProjector``): an op is measured
    against the content the ops before it leave behind, so a breach is caught before
    anything is written and the sibling lesson is never applied and then rolled back.
    """
    projector = _BatchSizeProjector()
    for i, (op, nm) in enumerate(zip(operations, names)):
        if (err := projector.error(nm, op["action"], op)) is not None:
            return i, err
        if projector.halted():
            # An op whose outcome is unknown failed in the write path, so no op after it
            # runs: a later breach is never reached.
            break
    return None


def _exceeds_content_cap(size: int, current_len: Optional[int] = None) -> bool:
    """True when a write of ``size`` characters must be refused by the ceiling.

    The ceiling bounds how much a skill may cost, so it refuses GROWTH past the limit. A
    file that is ALREADY over the limit may be written smaller: it is a legacy file that
    drifted past the cap (hand-placed, hub-installed, or grown by an older build), and
    refusing every write to it — including the shrink that brings it back under — leaves
    it frozen forever, with the only escape one whole-file rewrite that must land under
    the cap in a single shot. That escape is exactly how a lesson gets lost: the migration
    runs as a sequence of patches, and a file that is over the cap refuses every one of
    them. Refusing the growth of an over-cap file keeps the ceiling's meaning intact: no
    skill ever grows past it, and one that already did can always walk back down.
    """
    if size <= MAX_SKILL_CONTENT_CHARS:
        return False
    return current_len is None or size > current_len


def _validate_content_size(content: str, label: str = "SKILL.md",
                           current_len: Optional[int] = None) -> Optional[str]:
    """Reject content over MAX_SKILL_CONTENT_CHARS (see ``_exceeds_content_cap``).

    ``current_len`` is the character length of the file the write REPLACES (None when the
    write creates it), which is what lets an over-cap file be shrunk instead of frozen.
    """
    if not _exceeds_content_cap(len(content), current_len):
        return None
    msg = (f"{label} content is {len(content):,} characters (limit: {MAX_SKILL_CONTENT_CHARS:,}). "
           f"Consider splitting into a smaller SKILL.md with supporting files in references/ "
           f"or templates/.")
    if current_len is not None and current_len > MAX_SKILL_CONTENT_CHARS:
        msg += (f" {label} is already over the limit ({current_len:,} characters): shrink it "
                f"(move sections into references/) before adding to it.")
    return msg


def _description_of(content: Optional[str]) -> Optional[str]:
    """Normalized frontmatter description of a SKILL.md text; None when absent or unparseable."""
    with suppress(Exception):
        return str(_parse_frontmatter(content or "")[0]["description"]).strip().strip("'\"")
    return None


def _description_preview(content: str) -> str:
    """First 120 chars of the frontmatter description; '' on any failure."""
    with suppress(Exception):
        fm_end = _FRONTMATTER_END_RE.search(content[3:])
        if fm_end:
            return str(yaml.safe_load(content[3:fm_end.start() + 3]).get("description", ""))[:120]
    return ""


def _resolve_skill_dir(name: str, category: str | None = None) -> Path:
    """New-skill dir; honors ``skills.create_dir`` (e.g. a shared fleet dir)."""
    base = _skills_dir()
    try:
        from agent.skill_utils import get_skill_create_dir
        base = get_skill_create_dir() or base
    except Exception:
        logger.debug("skills.create_dir lookup failed", exc_info=True)
    return base / (category or "") / name


def _iter_skill_dirs(root: Path):
    from agent.skill_utils import is_excluded_skill_path
    for skill_md in root.rglob("SKILL.md"):
        if not is_excluded_skill_path(skill_md):
            yield skill_md.parent


def _read_frontmatter_name(skill_md: Path) -> Optional[str]:
    """Read a SKILL.md's frontmatter ``name:`` — the name skills_list displays.

    Fail-quiet on unreadable files: the fallback lookup in ``_find_skill``
    must never break the directory-name match, and an unreadable SKILL.md
    simply has no second name to offer. The value is truncated with the
    same ``MAX_NAME_LENGTH`` budget skills_list applies when displaying it.
    """
    try:
        from agent.skill_utils import parse_frontmatter

        content = skill_md.read_text(encoding="utf-8-sig", errors="replace")[:4000]
        frontmatter, _ = parse_frontmatter(content)
    except OSError:
        logger.debug("frontmatter read failed for %s", skill_md, exc_info=True)
        return None
    name = frontmatter.get("name")
    if isinstance(name, str):
        return name[:MAX_NAME_LENGTH]
    return None


def _find_skill(name: str) -> Optional[dict[str, Any]]:
    """Find a skill (local skills dir, then skills.external_dirs) -> ``{"path": Path}`` | None.

    Accepts the bare dir name (``axolotl``; matches category-nested skills too) and the
    categorized relative path (``mlops/axolotl``) — the two forms skill_view resolves. The
    categorized form matches RELATIVE to the local root only (relative_to raises for external dirs).

    As a last resort the frontmatter ``name:`` is matched too, so the name ``skills list``
    and the dashboard display (frontmatter name wins over the directory name there)
    resolves everywhere instead of failing with a misleading "not found in active
    profile" error when the two names diverge. Directory-name matches keep priority —
    a frontmatter match is only considered after the whole scan found no
    directory/categorized match, so one skill's frontmatter cannot shadow another
    skill's directory. A display name held by two or more distinct skills resolves to
    nothing: like skill_view's same-tier collision refusal, refusing to guess beats
    silently mutating the wrong skill; the directory name still resolves."""
    from agent.skill_utils import get_all_skills_dirs
    local_root = None
    if "/" in name or "\\" in name:
        try:
            local_root = _skills_dir().resolve()
        except OSError:
            logger.debug(
                "skills dir resolve failed; categorized lookups fall back to the unresolved path",
                exc_info=True)
            local_root = _skills_dir()
    display_matches: list[Path] = []
    seen_display: set = set()
    for skills_dir in get_all_skills_dirs():
        if not skills_dir.exists():
            continue
        for skill_dir in _iter_skill_dirs(skills_dir):
            if skill_dir.name == name:
                return {"path": skill_dir}
            if local_root is not None:
                resolved = skill_dir.resolve()
                if (resolved.is_relative_to(local_root)
                        and resolved.relative_to(local_root).as_posix() == name):  # POSIX form
                    return {"path": skill_dir}
            if _read_frontmatter_name(skill_dir / "SKILL.md") == name:
                key = skill_dir / "SKILL.md"
                with suppress(Exception):
                    key = key.resolve()
                if key not in seen_display:
                    seen_display.add(key)
                    display_matches.append(skill_dir)
    if len(display_matches) == 1:
        return {"path": display_matches[0]}
    if display_matches:
        logger.warning(
            "Skill display name '%s' is ambiguous (%d skills claim it: %s) — refusing to guess; "
            "resolve it by directory name instead",
            name, len(display_matches), "; ".join(str(p) for p in display_matches))
    return None


def _find_skill_in_other_profiles(name: str) -> list[tuple[str, Path]]:
    """``(profile, skill_dir)`` pairs for OTHER profiles holding ``name`` (so the not-found
    error can explain a wrong-profile mistake). Fail-quiet."""
    matches: list[tuple[str, Path]] = []
    try:
        from hermes_constants import get_default_hermes_root
        root = get_default_hermes_root()
    except Exception:
        return matches
    _active = _skills_dir()
    active_dir = _active.resolve() if _active.exists() else _active
    # Every profile's skills dir EXCEPT the active one (already searched). A candidate whose
    # path cannot be resolved is skipped (not a fatal error); is_dir() checks stay unguarded.
    candidates: list[tuple[str, Path]] = []
    with suppress(OSError, RuntimeError):
        if (root / "skills").resolve() != active_dir:
            candidates.append(("default", root / "skills"))
    if (root / "profiles").is_dir():
        with suppress(OSError):
            for entry in (root / "profiles").iterdir():
                if not entry.is_dir():
                    continue
                try:
                    if (entry / "skills").resolve() == active_dir:
                        continue
                except (OSError, RuntimeError):
                    continue
                candidates.append((entry.name, entry / "skills"))
    for profile_name, skills_dir in candidates:
        if not skills_dir.is_dir():
            continue
        with suppress(OSError):
            hit = next((d for d in _iter_skill_dirs(skills_dir) if d.name == name), None)
            if hit is not None:
                matches.append((profile_name, hit))  # one match per profile is enough
    return matches


def _skill_not_found_error(name: str, suffix: str = "") -> str:
    """Not-found error naming other profiles that hold the skill, plus ``suffix``."""
    from agent.file_safety import _resolve_active_profile_name
    base = f"Skill '{name}' not found in active profile '{_resolve_active_profile_name()}'."
    others = _find_skill_in_other_profiles(name)
    if len(others) == 1:
        other_profile, other_path = others[0]
        base += (
            f" A skill by that name exists in profile '{other_profile}' ({other_path}). To edit "
            f"it, switch profiles (`hermes -p {other_profile}`) or edit the file directly "
            f"(file tools / terminal).")
    elif others:
        names = ", ".join(f"'{p}'" for p, _ in others)
        base += (
            f" Skills by that name exist in other profiles: {names}. Switch profiles (`hermes -p "
            f"<name>`) to edit there, or edit the files directly (file tools / terminal).")
    else:
        base += " Use skills_list() to see available skills."
    return base + suffix


def _validate_file_path(file_path: str) -> Optional[str]:
    """Validate a write_file/remove_file path: under an allowed subdir, no escape."""
    from tools.path_security import has_traversal_component
    if not file_path:
        return "file_path is required."
    parts = Path(file_path).parts
    # Traversal first, so the SKILL.md exception is unreachable by a traversal-laden path.
    if has_traversal_component(file_path):
        return "Path traversal ('..') is not allowed."
    # SKILL.md lives at the skill root; accept 'SKILL.md' and '<skill>/SKILL.md'.
    if parts and parts[-1] == "SKILL.md" and len(parts) in (1, 2):
        return None
    if not parts or parts[0] not in ALLOWED_SUBDIRS:
        allowed = ", ".join(sorted(ALLOWED_SUBDIRS))
        return f"File must be under one of: {allowed}. Got: '{file_path}'"
    if len(parts) < 2:
        return f"Provide a file path, not just a directory. Example: '{parts[0]}/myfile.md'"
    return None


def _resolve_supporting_file(skill_dir: Path, file_path: str):
    """Validate ``file_path`` and resolve it inside ``skill_dir``
    -> ``(target, None)`` | ``(None, error_dict)``."""
    err = _file_path_refusal(skill_dir, file_path)
    return (None, _err(err)) if err else (skill_dir / (file_path or ""), None)


def _locate_for_write(name: str, action: str, not_found_suffix: str = ""):
    """Find the skill; a delete also runs the background-review delete guard -> ``(skill_dir, None)``
    | ``(None, error_dict)``."""
    existing = _find_skill(name)
    if not existing:
        return None, _err(_skill_not_found_error(name, not_found_suffix))
    skill_dir = existing["path"]
    guard = _background_review_delete_guard(name, skill_dir) if action == "delete" else None
    return (None, guard) if guard else (skill_dir, None)


def _guarded_write(name: str, skill_dir: Path, target: Path, action: str, label: str,
                   content: str) -> Optional[dict[str, Any]]:
    """Read-before-write guard (existing targets only), the description budget for the root SKILL.md
    (whichever action and file_path reach it), atomic write, then the security scan; a blocked scan
    restores the original (or unlinks a new file). Error dict or None."""
    original = None
    if target.exists():
        if read_guard := _background_review_read_before_write_guard(name, target, action, label):
            return read_guard
        original = target.read_text(encoding="utf-8-sig")
    if target == skill_dir / "SKILL.md" and (err := _description_budget_error(content, before=original)):
        return _err(err)
    from hermes_constants import mkdir_under_hermes_home
    mkdir_under_hermes_home(target.parent)
    atomic_write_text(target, content, preserve_mode=True, create_mode=0o644)
    scan_error = _security_scan_skill(skill_dir)
    if not scan_error:
        return None
    if original is not None:
        atomic_write_text(target, original, preserve_mode=True)
    else:
        target.unlink(missing_ok=True)
    return _err(scan_error)


def _add_description_prompt_preview(result: dict[str, Any], content: str) -> dict[str, Any]:
    fm, _ = _parse_frontmatter(content)
    if is_skill_description_truncated_for_prompt(fm):
        result["system_prompt_preview"] = (
            f"System prompt will show: \"{extract_skill_description(fm)}\" — shorten the "
            f"description to {SKILL_PROMPT_DESC_LIMIT} chars so the index shows all of it.")
    return result


def _attach_lint_findings(result: dict[str, Any], skill_md: Path, before: Optional[str] = None) -> None:
    """Attach ADVISORY authoring findings (hard rejects already ran in _validate_frontmatter).
    With ``before`` (the pre-write content) only rules the write INTRODUCED are attached, so a
    patch reports the line it crossed rather than re-listing the skill's standing findings."""
    try:
        from tools.skill_linter import lint_content, lint_skill  # local import: optional path
        findings = lint_skill(skill_md)
        if before is not None:
            standing = {f.rule for f in lint_content(before, skill_dir=skill_md.parent)}
            findings = [f for f in findings if f.rule not in standing]
    except Exception:
        findings = None
    if not findings:
        return
    result["lint_warnings"] = [
        {"severity": f.severity, "rule": f.rule, "message": f.message} for f in findings]
    result["lint_hint"] = (
        "The write succeeded. These are advisory authoring-convention findings (not blockers) "
        "— fix them with skill_manage(action='patch') to match Hermes skill standards.")


def _clip(text: str, n: int, ellipsis: str) -> str:
    return text[:n] + (ellipsis if len(text) > n else "")


# --- Core actions -------------------------------------------------------------

def _create_skill(name: str, content: str, category: str | None = None) -> dict[str, Any]:
    if err := (_validate_name(name) or _validate_category(category)
               or _validate_frontmatter(content) or _validate_content_size(content)):
        return _err(err)
    if existing := _find_skill(name):
        return _err(f"A skill named '{name}' already exists at {existing['path']}.")
    skill_dir = _resolve_skill_dir(name, category)
    from hermes_constants import mkdir_under_hermes_home
    mkdir_under_hermes_home(skill_dir.parent)
    try:
        skill_dir.mkdir(exist_ok=False)
    except FileExistsError:
        # mkdir raised EEXIST for a file, symlink (live or dangling) or dir alike; only an EMPTY
        # real directory (leftover of an earlier create whose SKILL.md write failed) may be used.
        # Anything else — or anything unstat-able/unlistable — is someone else's: never adopt.
        try:
            usable = (not _is_path_redirect(skill_dir) and skill_dir.is_dir()
                      and not any(skill_dir.iterdir()))
        except OSError:  # permissions / ACL
            usable = False
        if not usable:
            return _err(f"Cannot create skill '{name}': {skill_dir} already exists (not an empty "
                        "directory, or unreadable). Choose another name, or move/remove that path and retry.")
    skill_md = skill_dir / "SKILL.md"
    if guard := _guarded_write(name, skill_dir, skill_md, "create", "SKILL.md", content):
        with suppress(OSError):  # rmdir, not rmtree: only an empty dir goes, anything foreign stays
            skill_dir.rmdir()
        return guard
    root = _skills_dir()  # display relative under the profile dir; absolute under skills.create_dir
    display = skill_dir.relative_to(root) if skill_dir.is_relative_to(root) else skill_dir
    result = {
        "success": True, "message": f"Skill '{name}' created.", "path": str(display),
        "skill_md": str(skill_md), "_change": {"description": _description_preview(content)},
        **({"category": category} if category else {}),
        "hint": "To add reference files, templates, or scripts, use "
                f"skill_manage(action='write_file', name='{name}', file_path='references/example.md', "
                "file_content='...')"}
    _attach_lint_findings(_add_description_prompt_preview(result, content), skill_md)
    return result


def _edit_skill(name: str, content: str) -> dict[str, Any]:
    """Replace the SKILL.md of any existing skill (full rewrite)."""
    # current_len: the SKILL.md being replaced. A file already over the ceiling may be
    # rewritten SMALLER (the migration path); one at or under it may not grow past it.
    if err := (_validate_frontmatter(content)
               or _content_size_error(content, "SKILL.md", name)):
        return _err(err)
    skill_dir, guard = _locate_for_write(name, "edit")
    # SKILL.md always exists here (_find_skill requires it), so a blocked scan restores it.
    if guard := guard or _guarded_write(name, skill_dir, skill_dir / "SKILL.md", "edit", "SKILL.md", content):
        return guard
    result = {
        "success": True, "message": f"Skill '{name}' updated (full rewrite).",
        "path": str(skill_dir), "_change": {"description": _description_preview(content)}}
    return _add_description_prompt_preview(result, content)


def _patch_skill(name: str, old_string: str, new_string: str, file_path: str | None = None,
                 replace_all: bool = False) -> dict[str, Any]:
    """Targeted find-and-replace in SKILL.md (default) or a supporting file; unique match unless replace_all."""
    if not old_string:
        return _err(_PATCH_NEEDS_OLD_STRING)
    if new_string is None:
        return _err(_PATCH_NEEDS_NEW_STRING)
    # No old_string == new_string guard here: fuzzy_find_and_replace rejects that with a
    # richer error (file_preview) this layer cannot produce.
    skill_dir, guard = _locate_for_write(name, "patch")
    if guard:
        return guard
    target_label = file_path or "SKILL.md"
    if file_path:
        target, err = _resolve_supporting_file(skill_dir, file_path)
        if err:
            return err
    else:
        target = skill_dir / "SKILL.md"
    if not target.exists():
        return _err(f"File not found: {target.relative_to(skill_dir)}")
    if read_guard := _background_review_read_before_write_guard(name, target, "patch", target_label):
        return read_guard
    try:
        content = target.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        # A binary file is not text: skill_manage edits text, so refuse it with a JSON error
        # instead of letting a UnicodeDecodeError escape the tool.
        return _err(_not_text_error(target_label))

    # Use the same fuzzy matching engine as the file patch tool.
    from tools.fuzzy_match import fuzzy_find_and_replace
    correction_notes = []
    new_content, match_count, _strategy, match_error = fuzzy_find_and_replace(
        content, old_string, new_string, replace_all, correction_notes=correction_notes)
    if match_error:
        with suppress(Exception):
            from tools.fuzzy_match import format_no_match_hint
            match_error += format_no_match_hint(match_error, match_count, old_string, content)
        return _err(match_error) | {"file_preview": _clip(content, 500, "...")}
    # current_len: the file being patched. A file already over the ceiling may be patched
    # SMALLER (that is how it walks back under); one at or under it may not grow past it.
    if err := _validate_content_size(new_content, label=target_label, current_len=len(content)):
        return _err(err)
    if not file_path and (err := _validate_frontmatter(new_content)):
        return _err(f"Patch would break SKILL.md structure: {err}")
    if guard := _guarded_write(name, skill_dir, target, "patch", target_label, new_content):
        return guard
    result = {
        "success": True,
        "message": f"Patched {target_label} in skill '{name}' ({match_count} replacement{'s' if match_count > 1 else ''}).",
        "_change": {"old": _clip(old_string, 200, "…"), "new": _clip(new_string, 200, "…")}}
    if correction_notes:
        result["note"] = "\n".join(correction_notes)
    # SKILL.md grows by patches, not by creates: surface findings on the patch that crosses a line
    # (oversized-body, incident-log-shape) — a clean patch attaches nothing and stays quiet.
    if not file_path:
        _attach_lint_findings(result, target, before=content)
    return result


def _delete_skill(name: str, absorbed_into: Optional[str] = None) -> dict[str, Any]:
    """Delete a skill. ``absorbed_into``: None = undeclared (legacy, accepted); "" = explicit prune;
    "<skill>" = absorbed into that umbrella, which must exist (so the model can't claim one)."""
    skill_dir, guard = _locate_for_write(name, "delete")
    if guard := guard or _curator_consolidation_delete_guard(name, absorbed_into):
        return guard
    if pinned_err := _pinned_guard(name):
        return _err(pinned_err)
    absorbed_target = absorbed_into.strip() if isinstance(absorbed_into, str) else ""
    if absorbed_target:
        if absorbed_target == name:
            return _err(f"absorbed_into='{absorbed_target}' cannot equal the skill being deleted.")
        if not _find_skill(absorbed_target):
            return _err(f"absorbed_into='{absorbed_target}' does not exist. "
                        f"Create or patch the umbrella skill first, then retry the delete.")
    skills_root = _containing_skills_root(skill_dir)
    if unsafe := _validate_delete_target(skill_dir):  # defense-in-depth before rmtree
        return _err(unsafe)
    # Curator consolidations must be RECOVERABLE (`hermes curator restore`): archive instead
    # of rmtree. Foreground deletes keep hard-delete semantics.
    absorbed_note = f" Content absorbed into '{absorbed_target}'." if absorbed_target else ""
    if _is_background_review():
        try:
            from tools.skill_usage import archive_skill
            ok, archive_msg = archive_skill(name)
        except Exception as e:
            return _err(f"failed to archive '{name}': {e}")
        if not ok:
            return _err(archive_msg)
        return {"success": True,
                "message": f"Skill '{name}' archived ({archive_msg}).{absorbed_note}",
                "_archived": True}
    shutil.rmtree(skill_dir)
    _rmdir_if_empty(skill_dir.parent, skills_root)  # empty category dir, never the root
    return {"success": True, "message": f"Skill '{name}' deleted.{absorbed_note}"}


def _rmdir_if_empty(parent: Path, stop: Path) -> None:
    if parent != stop and parent.exists() and not any(parent.iterdir()):
        parent.rmdir()


def _write_file(name: str, file_path: str, file_content: str) -> dict[str, Any]:
    """Add or overwrite a supporting file within any skill directory."""
    if err := _validate_file_path(file_path):
        return _err(err)
    if not file_content and file_content != "":
        return _err("file_content is required.")
    if (content_bytes := len(file_content.encode("utf-8"))) > MAX_SKILL_FILE_BYTES:
        return _err(f"File content is {content_bytes:,} bytes (limit: {MAX_SKILL_FILE_BYTES:,} "
                    f"bytes / 1 MiB). Consider splitting into smaller files.")
    # current_len: the supporting file being replaced (None when it does not exist yet).
    if err := _content_size_error(file_content, file_path, name, file_path):
        return _err(err)
    skill_dir, guard = _locate_for_write(name, "write_file", " Create it first with action='create'.")
    if guard:
        return guard
    target, err = _resolve_supporting_file(skill_dir, file_path)
    if guard := err or _guarded_write(name, skill_dir, target, "write_file", file_path, file_content):
        return guard
    result = {"success": True, "message": f"File '{file_path}' written to skill '{name}'.", "path": str(target)}
    # references/ is where per-session hoarding shows up; surface the sprawl finding on the write
    # that crosses the line so the review fork sees it in the same turn.
    if file_path.startswith("references/") and (skill_dir / "SKILL.md").exists():
        _attach_lint_findings(result, skill_dir / "SKILL.md")
    return result


def _remove_file(name: str, file_path: str) -> dict[str, Any]:
    """Remove a supporting file from any skill directory."""
    if err := _validate_file_path(file_path):
        return _err(err)
    skill_dir, guard = _locate_for_write(name, "remove_file")
    if guard:
        return guard
    target, err = _resolve_supporting_file(skill_dir, file_path)
    if err:
        return err
    if not target.exists():  # list what IS there so the model can pick the right path
        available = [str(f.relative_to(skill_dir)) for subdir in ALLOWED_SUBDIRS
                     if (skill_dir / subdir).exists() for f in (skill_dir / subdir).rglob("*") if f.is_file()]
        return _err(f"File '{file_path}' not found in skill '{name}'.", available_files=available or None)
    if read_guard := _background_review_read_before_write_guard(name, target, "remove_file", file_path):
        return read_guard
    target.unlink()
    _rmdir_if_empty(target.parent, skill_dir)
    return {"success": True, "message": f"File '{file_path}' removed from skill '{name}'."}


# --- Main entry point ---------------------------------------------------------

# Set while replaying an approved staged skill write so skill_manage() does not re-gate it.
_skill_gate_bypass: _ctxvars.ContextVar[bool] = _ctxvars.ContextVar(
    "skill_gate_bypass", default=False)


def _run_write_gate(build_staging):
    """Shared write gate: None to proceed, else a JSON tool result (blocked/staged).
    ``build_staging(wa) -> (payload, gist)`` runs only when staging. Fails open if
    write_approval cannot be imported."""
    try:
        from tools import write_approval as wa
    except Exception:
        return None  # fail open
    decision = wa.evaluate_gate(wa.SKILLS)
    if decision.allow:
        return None
    if decision.blocked:
        return tool_error(decision.message, success=False)
    payload, gist = build_staging(wa)
    record = wa.stage_write(wa.SKILLS, payload, summary=gist, origin=wa.current_origin())
    return json.dumps({"success": True, "staged": True, "pending_id": record["id"],
                       "gist": gist, "message": decision.message}, ensure_ascii=False)


def _apply_skill_write_gate(action, name, **payload_kwargs):
    """Flat-shape gate: stage the full kwargs so approval can replay them; bypassed during replay."""
    if action not in _ACTION_HANDLERS or _skill_gate_bypass.get():
        return None
    def _staging(wa):
        payload = {"action": action, "name": name,
                   **{k: v for k, v in payload_kwargs.items() if v is not None}}
        gist_kw = {k: payload_kwargs.get(k) or ""
                   for k in ("content", "file_path", "old_string", "new_string")}
        return payload, wa.skill_gist(action, name, **gist_kw)
    return _run_write_gate(_staging)


_FLAT_OP_KEYS = ("content", "category", "file_path", "file_content", "old_string", "new_string",
                 "absorbed_into", "operations")


def _skill_manage_from(payload: dict[str, Any], **extra) -> str:
    """Call ``skill_manage`` with the flat-shape fields (and absorbed_into/operations) of ``payload``."""
    return skill_manage(
        action=payload.get("action", ""), name=payload.get("name", ""),
        replace_all=payload.get("replace_all", False),
        **{k: payload.get(k) for k in _FLAT_OP_KEYS}, **extra)


def apply_skill_pending(payload: dict[str, Any]) -> str:
    """Replay a staged skill write, bypassing the gate (the /skills approve handler)."""
    token = _skill_gate_bypass.set(True)
    try:
        return _skill_manage_from(payload)
    finally:
        _skill_gate_bypass.reset(token)


def _act_patch(a):
    """Two shapes: old_string/new_string = targeted replacement (validated in _patch_skill so the
    tool and the helper give the same guidance); content alone = full rewrite (the old 'edit')."""
    if a["content"] and (a["old_string"] or a["new_string"] is not None):
        return tool_error(_PATCH_EITHER_OR, success=False)
    if a["content"]:
        return _edit_skill(a["name"], a["content"])
    return _patch_skill(a["name"], a["old_string"], a["new_string"], a["file_path"], a["replace_all"])


# action -> handler(args dict) returning a result dict, or a tool_error JSON string for
# argument-shape errors. "edit" is a legacy alias for a full rewrite (not in the schema).
_ACTION_HANDLERS = {
    "create": lambda a: _create_skill(a["name"], a["content"], a["category"]),
    "edit": lambda a: _edit_skill(a["name"], a["content"]),
    "patch": _act_patch,
    "delete": lambda a: _delete_skill(a["name"], absorbed_into=a["absorbed_into"]),
    "write_file": lambda a: _write_file(a["name"], a["file_path"], a["file_content"]),
    "remove_file": lambda a: _remove_file(a["name"], a["file_path"])}


def _record_success(action, name, result, *, file_path, absorbed_into, task_id,
                    session_id, ledger_before) -> None:
    """Best-effort post-mutation side effects (never break the tool): ledger, prompt-cache
    clear, curator telemetry (which fires ``on_skill_lifecycle`` for plugins)."""
    with suppress(Exception):
        from tools import skill_ledger as _ledger
        _post = _find_skill(name)
        # delete: consolidation vs prune, and whether the recoverable archive handled it
        _evidence = ({"absorbed_into": absorbed_into, "archived": bool(result.get("_archived"))}
                     if action == "delete" else {})
        _evidence.update({k: v for k, v in (("session_id", session_id), ("file_path", file_path)) if v})
        _ledger.record_mutation(
            action, name, before=ledger_before if ledger_before is not None else [],
            after_root=_post["path"] if _post else None, evidence=_evidence)
    with suppress(Exception):
        from agent.prompt_builder import clear_skills_system_prompt_cache
        clear_skills_system_prompt_cache(clear_snapshot=True)
    # Curator telemetry: only the background review fork marks a skill agent-created
    # (foreground creates belong to the user). A recoverable curator archive keeps its
    # record as STATE_ARCHIVED (`hermes curator status`/`restore`); only a hard delete forgets.
    with suppress(Exception):
        from tools.skill_usage import bump_patch, forget, record_created
        # During the curator consolidation pass, a verified consolidation must be RECOVERABLE: archival into
        # ~/.hermes/skills/.archive/ is documented as the maximum destructive action the curator may take,
        # and `hermes curator restore` promises the skill can be brought back. Route through the recoverable
        # archive primitive instead of permanent rmtree so a misjudged consolidation can be undone (#29912).
        # Foreground, user-directed deletes keep their existing hard-delete semantics.
        from tools.skill_provenance import is_background_review
        if action == "create":
            record_created(name, agent_created=is_background_review(),
                           task_id=task_id, session_id=session_id)
        elif action in {"patch", "edit", "write_file", "remove_file"}:
            bump_patch(name, action=action, task_id=task_id, session_id=session_id)
        elif action == "delete" and not result.get("_archived"):
            forget(name)


def skill_manage(
    action: str, name: str, content: str | None = None, category: str | None = None, file_path: str | None = None,
    file_content: str | None = None, old_string: str | None = None, new_string: str | None = None,
    replace_all: bool = False, absorbed_into: str | None = None, task_id: str | None = None,
    session_id: str | None = None, operations=None) -> str:
    """Dispatch to the action handler -> JSON string. ``operations`` (atomic batch shape,
    see _skill_manage_batch) overrides the flat fields."""
    if operations is not None:
        return _skill_manage_batch(
            operations, default_name=name or None, task_id=task_id, session_id=session_id)
    if (preflight := _background_review_preflight(action, name)) is not None:
        return json.dumps(preflight, ensure_ascii=False)
    # Approval gate: skills are too large to review inline, so they always stage regardless
    # of origin; bypassed when replaying an approved staged write.
    args = dict(content=content, category=category, file_path=file_path, file_content=file_content,
                old_string=old_string, new_string=new_string, replace_all=replace_all,
                absorbed_into=absorbed_into)
    if (gate_result := _apply_skill_write_gate(action, name, **args)) is not None:
        return gate_result
    if (shape_err := _op_shape_error(action, args)) is not None:
        return tool_error(shape_err, success=False)
    # Validate before the lock is keyed on the name, so a rejected name never touches .locks/
    # (create takes a bare name; the other actions also accept ``category/name``).
    if (name_err := _validate_name(name if action == "create" or not name else Path(name).name)) is not None:
        return json.dumps(_err(name_err), ensure_ascii=False)
    # A mutation is read-modify-write even when its action eventually delegates
    # to a helper: guards, ledger capture, patch matching, validation, rollback,
    # and the atomic replacement all belong to the same ownership window.
    with _skill_mutation_lock(name):
        # Ledger pre-capture: telemetry, not a gate — failures must NEVER block the mutation. delete
        # destroys the whole package (consolidation may have re-homed support files first), so
        # complete it from the newest curator backup or a restore is hollow.
        # Audit ledger (tracker #79686 P3): capture the pre-mutation state of the skill directory so every
        # mutation — any actor — lands in the append-only JSONL ledger with before/after blobs.
        _ledger_before = None
        with suppress(Exception):
            from tools import skill_ledger as _ledger
            _pre = _find_skill(name)
            _ledger_before = _ledger.capture_before(
                _pre["path"] if _pre else None, complete_package=(action == "delete"), skill=name)
        handler = _ACTION_HANDLERS.get(action, lambda a: _err(
            f"Unknown action '{action}'. Use: create, edit, patch, delete, write_file, remove_file"))
        result = handler({"name": name, **args})
        if isinstance(result, str):
            return result  # tool_error JSON for argument-shape problems (patch)
        if result.get("success"):
            _record_success(
                action, name, result, file_path=file_path, absorbed_into=absorbed_into,
                task_id=task_id, session_id=session_id, ledger_before=_ledger_before)
    return json.dumps(result, ensure_ascii=False)


# --- OpenAI Function-Calling Schema -------------------------------------------

def _skill_manage_description() -> str:
    return (
        "Create, update or delete skills: your procedural memory for recurring "
        "tasks. The call is an operations array (a single "
        "edit is a list of one); it applies atomically — any failure rolls "
        "every touched skill back. Ops: create (full SKILL.md; lands in "
        "the profile's skills directory or configured skills.create_dir; must precede that skill's other "
        "ops), patch (targeted old_string/new_string fix — preferred; "
        "content alone REPLACES the whole file, read it via skill_view() "
        "first), write_file/remove_file (supporting files), delete (sole "
        "op only). Existing skills are modified wherever they live. "
        f"Description: max {SKILL_PROMPT_DESC_LIMIT} chars, 'Use when <trigger>. <behavior>.', never "
        "the category name (the index shows it). Write lessons, not logs: "
        "imperative rule + why, no PR numbers/dates/incident narration, one "
        "rule per lesson, references/ named by topic (extend before adding). "
        "skill_view() shows format conventions."
    )


_NAME = {"type": "string"}
_OLD_STRING = {"type": "string",
               "description": "Text to find (same matching semantics as the patch tool)."}
_NEW_STRING = {"type": "string", "description": "Replacement; empty string deletes the match."}
_FILE_PATH = {
    "type": "string",
    "description": (
        "Path RELATIVE to the skill's own directory, e.g. 'references/api.md' — no leading "
        "slash, never absolute; first segment references/, templates/, scripts/, or assets/."
    ),
}  # stated once (write_file); patch/remove_file point at it


def _op_schema(action: str, props: dict, required: tuple) -> dict:
    """One self-contained per-action op shape. ``additionalProperties: false`` is what lets a
    grammar-constrained backend refuse another action's text slot outright."""
    return {"type": "object", "additionalProperties": False,
            "properties": {"name": _NAME, "action": {"type": "string", "enum": [action]}, **props},
            "required": ["name", "action", *required]}


SKILL_MANAGE_SCHEMA = {
    "name": "skill_manage",
    # ONE advertised call shape (memory-tool pattern): the call IS an operations
    # array. The legacy flat shape (top-level action/name/content/...) is still
    # ACCEPTED for old transcripts and staged-write replay, but not advertised.
    "description": _skill_manage_description(),
    "parameters": {
        "type": "object",
        "properties": {
            "operations": {
                "type": "array",
                "description": (
                    "Ordered ops; each names its target skill (lowercase, hyphens/underscores, "
                    "max 64 chars). Each action is its own shape; another action's text slot is invalid."
                ),
                # Per-action branches, not one flat union: with one object holding content /
                # new_string / file_content side by side, a 27B model that just used write_file
                # kept emitting file_content on create and the whole batch rolled back (#112677).
                "items": {"anyOf": [
                    _op_schema("create", {
                        "content": {"type": "string",
                                    "description": "Full SKILL.md text (YAML frontmatter + markdown body)."},
                        "category": {"type": "string",
                                     "description": "Optional category subdir (e.g. 'devops')."},
                    }, ("content",)),
                    _op_schema("patch", {
                        "old_string": _OLD_STRING, "new_string": _NEW_STRING,
                        "replace_all": {"type": "boolean",
                                        "description": "Replace all occurrences (default false)."},
                        "file_path": {"type": "string",
                                      "description": "Optional supporting file (write_file's shape); default SKILL.md."},
                    }, ("old_string", "new_string")),
                    _op_schema("patch", {
                        "content": {"type": "string",
                                    "description": "Full SKILL.md rewrite (REPLACES the whole file; last resort)."},
                    }, ("content",)),
                    _op_schema("write_file", {
                        "file_path": _FILE_PATH,
                        "file_content": {"type": "string", "description": "Full text of the supporting file."},
                    }, ("file_path", "file_content")),
                    _op_schema("remove_file", {
                        "file_path": {"type": "string", "description": "Supporting file (write_file's shape)."},
                    }, ("file_path",)),
                    # `absorbed_into` stays in the delete shape: with additionalProperties:false a
                    # grammar-constrained backend would otherwise strip it and the curator's
                    # consolidation delete guard would fail-close every consolidation.
                    _op_schema("delete", {
                        "absorbed_into": {"type": "string",
                                          "description": "Curator consolidation only: umbrella skill "
                                                         "that absorbed this one (must exist)."},
                    }, ()),
                ]},
            },
            # Also accepted, never advertised: the legacy flat single-op fields.
        },
        "required": ["operations"],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="skill_manage", toolset="skills", schema=SKILL_MANAGE_SCHEMA, emoji="📝",
    handler=lambda args, **kw: _skill_manage_from(
        args, task_id=kw.get("task_id"), session_id=kw.get("session_id")))
