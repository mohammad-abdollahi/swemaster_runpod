"""Per-instance repository sandbox for the SWE agent tool loop.

Each SWE-bench-style task has a (repo, base_commit, gold_patch) triplet.  The
sandbox manages a local git clone, resets it to ``base_commit`` between
rollouts, exposes a few read-only & one write-style tool to the model, and
extracts the cumulative diff at submission time.

Designed for a *minimal* openhands-style loop — not a full reproduction.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
from dataclasses import dataclass, field
from typing import Optional


REPOS_ROOT = os.environ.get(
    "SWE_REPOS_ROOT",
    f"/scratch/{os.environ.get('USER','user')}/swemaster/repos",
)


def _repo_dir_for(repo: str) -> str:
    """``"fair-workflows/nanopub"`` → ``".../fair-workflows_nanopub"``."""
    return os.path.join(REPOS_ROOT, repo.replace("/", "_"))


def _parse_edit_body(body: str) -> tuple[Optional[str], str, str]:
    """Pull ``PATH`` / ``OLD`` / ``NEW`` blocks out of an edit_file body.

    Accepted variants the SFT model emits in practice:

      1. ``PATH: path/to/file.py\n OLD:\n ...\n NEW:\n ...``  (the spec)
      2. ``path/to/file.py\n OLD:\n ...\n NEW:\n ...``        (no PATH: prefix)
      3. ``path/to/file.py\n old:\n ...\n new:\n ...``        (lowercase keys)

    OLD body is everything between ``OLD:`` and ``NEW:``; NEW body is
    everything from ``NEW:`` to the end.
    """
    import re as _re
    # Case-insensitive headers, OLD/NEW on their own line.
    # Try explicit PATH: form first.
    m = _re.search(
        r"^\s*PATH\s*:\s*(?P<path>.+?)\s*$\s*"
        r"^\s*OLD\s*:\s*$\s*(?P<old>.*?)\s*"
        r"^\s*NEW\s*:\s*$\s*(?P<new>.*?)\s*\Z",
        body, _re.DOTALL | _re.MULTILINE | _re.IGNORECASE,
    )
    if m:
        return m.group("path").strip(), m.group("old"), m.group("new")
    # Fallback: first non-empty line is the path, then OLD:/NEW: anchors.
    m = _re.search(
        r"^\s*(?P<path>\S[^\n]*?)\s*$\s*"
        r"^\s*OLD\s*:\s*$\s*(?P<old>.*?)\s*"
        r"^\s*NEW\s*:\s*$\s*(?P<new>.*?)\s*\Z",
        body, _re.DOTALL | _re.MULTILINE | _re.IGNORECASE,
    )
    if m:
        return m.group("path").strip(), m.group("old"), m.group("new")
    return None, "", ""


@dataclass
class ToolResult:
    """Stdout of a tool call. Bounded so it never bloats the context."""
    ok: bool
    output: str
    error: str = ""

    def to_user_text(self, max_chars: int = 32000) -> str:
        """Bound the result before it lands in the chat history.

        Must be >= ``read_file``'s own cap (32 kB) so a full file read isn't
        re-truncated here, which would make the model loop and re-read."""
        body = self.output if self.ok else f"ERROR: {self.error}\n{self.output}"
        if len(body) > max_chars:
            body = body[:max_chars] + "\n...(truncated)"
        return body


@dataclass
class SweSandbox:
    repo: str          # e.g. "fair-workflows/nanopub"
    base_commit: str
    gold_patch: str = ""
    instance_id: str = ""
    submitted: bool = False
    final_patch: Optional[str] = field(default=None, repr=False)

    @property
    def cwd(self) -> str:
        return _repo_dir_for(self.repo)

    # ---- lifecycle -------------------------------------------------------
    def reset(self) -> None:
        """Hard-reset the worktree to ``base_commit``.  Called before each
        rollout so the next rollout starts from the same point in history.
        """
        self._git("reset", "--hard", self.base_commit, check=False)
        self._git("clean", "-fdx", check=False)
        self.submitted = False
        self.final_patch = None

    # ---- tools the model can call ----------------------------------------
    def list_files(self, path: str = ".") -> ToolResult:
        """``ls`` inside the repo.  Cheap context for the model."""
        target = self._safe_join(path)
        if target is None:
            return ToolResult(False, "", f"path outside repo: {path}")
        try:
            entries = sorted(os.listdir(target))
            out = "\n".join(entries[:200])
            if len(entries) > 200:
                out += f"\n...({len(entries)-200} more)"
            return ToolResult(True, out)
        except FileNotFoundError:
            return ToolResult(False, "", f"no such directory: {path}")

    def read_file(self, path: str, max_chars: int = 32000) -> ToolResult:
        target = self._safe_join(path)
        if target is None:
            return ToolResult(False, "", f"path outside repo: {path}")
        if not os.path.isfile(target):
            return ToolResult(False, "", f"no such file: {path}")
        try:
            with open(target, "r", errors="replace") as f:
                text = f.read()
        except OSError as e:
            return ToolResult(False, "", str(e))
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...(truncated)"
        return ToolResult(True, text)

    @staticmethod
    def _strip_mode_lines(diff_text: str) -> str:
        """Drop ``old mode 100644`` / ``new mode 100755`` / mode-only ``index``
        suffixes so a model-generated diff whose mode header disagrees with
        the file on disk doesn't get rejected outright.
        """
        out_lines = []
        for ln in diff_text.splitlines(keepends=True):
            stripped = ln.lstrip()
            if stripped.startswith(("old mode ", "new mode ",
                                    "deleted file mode ", "new file mode ")):
                continue
            # ``index abc..def 100644`` → ``index abc..def``
            if stripped.startswith("index "):
                parts = ln.rstrip("\n").split()
                if len(parts) >= 3 and parts[-1].isdigit() and len(parts[-1]) == 6:
                    ln = " ".join(parts[:-1]) + "\n"
            out_lines.append(ln)
        return "".join(out_lines)

    def apply_patch(self, diff_text: str) -> ToolResult:
        """``git apply`` the supplied diff with a forgiving fallback chain.

        Tries (in order):
          1. plain ``git apply`` on the original text
          2. same after stripping mode-related lines (executable-bit mismatch
             is the #1 reason an otherwise-fine diff is rejected)
          3. ``git apply --3way`` for context-aware merge
          4. ``git apply --reject`` so partial diffs land what they can.

        On success returns the actual ``git apply`` command that worked so
        the model gets a feedback signal. On failure returns the LAST error.
        """
        if not diff_text.strip():
            return ToolResult(False, "", "empty patch")

        attempts: list[tuple[str, list[str], str]] = [
            ("plain",        ["git", "apply", "--whitespace=nowarn"], diff_text),
            ("strip-mode",   ["git", "apply", "--whitespace=nowarn"], self._strip_mode_lines(diff_text)),
            ("3way",         ["git", "apply", "--3way", "--whitespace=nowarn"], self._strip_mode_lines(diff_text)),
            ("reject",       ["git", "apply", "--reject", "--whitespace=nowarn"], self._strip_mode_lines(diff_text)),
        ]
        last_err = "git apply failed"
        last_out = ""
        for label, cmd, text in attempts:
            proc = subprocess.run(
                cmd, input=text, capture_output=True, text=True,
                cwd=self.cwd, timeout=45,
            )
            if proc.returncode == 0:
                return ToolResult(
                    True,
                    f"patch applied via '{label}'."
                    + (f"\n{proc.stderr.strip()}" if proc.stderr.strip() else ""),
                )
            last_err = (proc.stderr or "git apply failed").strip()
            last_out = proc.stdout
        return ToolResult(False, last_out, f"all 4 apply strategies failed; "
                                          f"last error: {last_err}")

    def edit_file(self, body: str) -> ToolResult:
        """String-replacement edit. Body must contain ``PATH:``, ``OLD:``,
        ``NEW:`` blocks (in that order).  Replaces the first occurrence of
        ``OLD`` with ``NEW`` in the file at ``PATH``.

        This bypasses unified-diff hunk-header counting (which LLMs get wrong
        ~always) — equivalent to OpenHands' ``str_replace_editor``.
        """
        path, old, new = _parse_edit_body(body)
        if path is None:
            return ToolResult(False, "", "edit_file body must contain PATH:, "
                                         "OLD:, NEW: sections (in that order)")
        target = self._safe_join(path)
        if target is None or not os.path.isfile(target):
            return ToolResult(False, "", f"no such file: {path}")
        with open(target, "r", errors="replace") as f:
            content = f.read()
        if old == "":
            # Empty OLD == append/prepend doesn't make sense for the agent; refuse.
            return ToolResult(False, "", "OLD section must be non-empty")
        occurrences = content.count(old)
        if occurrences == 0:
            return ToolResult(
                False, "",
                f"OLD text not found in {path}. Make sure it matches EXACTLY "
                "(whitespace, newlines included).",
            )
        if occurrences > 1:
            return ToolResult(
                False, "",
                f"OLD text appears {occurrences} times in {path}; needs to "
                "be unique. Expand OLD with surrounding context.",
            )
        new_content = content.replace(old, new, 1)
        with open(target, "w") as f:
            f.write(new_content)
        return ToolResult(True, f"{path}: replaced 1 occurrence "
                                f"({len(old)} chars → {len(new)} chars).")

    def submit(self) -> ToolResult:
        """Mark the rollout as done. ``current_diff`` becomes the candidate."""
        self.final_patch = self.current_diff()
        self.submitted = True
        return ToolResult(True, "submitted")

    # Truly read-only commands only. The model uses `apply_patch` for any
    # mutation; `bash` is for fast exploration (grep/find/ls).  Commands like
    # `sed -i`, `awk > out`, `tr ... > out` could clobber the cloned repo, so
    # they're excluded.  Shell metacharacters that enable redirection / piping
    # to writeable destinations (`>`, `>>`, `|`, `&`, `;`, `` ` ``, `$(...)`)
    # are also rejected.
    _BASH_WHITELIST = ("ls", "find", "cat", "head", "tail", "grep", "rg",
                       "wc", "echo", "pwd", "tree", "file", "stat", "diff")
    _BASH_FORBIDDEN_CHARS = (">", "<", "|", ";", "&", "`", "$(")

    def bash(self, command: str, timeout: int = 15) -> ToolResult:
        """Run a read-only shell command inside the repo. Whitelisted only.

        Mutation goes through `apply_patch` only — never through `bash`.
        """
        command = command.strip()
        if not command:
            return ToolResult(False, "", "empty command")
        for bad in self._BASH_FORBIDDEN_CHARS:
            if bad in command:
                return ToolResult(False, "",
                                  f"shell metacharacter {bad!r} not allowed in bash; "
                                  f"use a single read-only command or apply_patch")
        first_token = command.split(None, 1)[0]
        if first_token not in self._BASH_WHITELIST:
            return ToolResult(False, "",
                              f"command {first_token!r} not permitted. "
                              f"allowed: {', '.join(self._BASH_WHITELIST)}")
        try:
            proc = subprocess.run(
                ["bash", "-c", command],
                capture_output=True, text=True,
                cwd=self.cwd, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, "", f"timeout after {timeout}s")
        output = proc.stdout
        if proc.stderr:
            output += "\n[stderr]\n" + proc.stderr
        return ToolResult(proc.returncode == 0, output,
                          "" if proc.returncode == 0 else f"exit {proc.returncode}")

    # ---- introspection ---------------------------------------------------
    def current_diff(self) -> str:
        """Diff of the worktree vs ``base_commit``.

        ``git add -N .`` marks any untracked files as intent-to-add so a
        newly-created file (typical apply_patch output) shows up in the
        diff.  The next rollout's ``reset --hard`` wipes the index again.
        """
        self._git("add", "-N", ".", check=False)
        proc = subprocess.run(
            ["git", "diff", self.base_commit, "--no-color"],
            capture_output=True, text=True, cwd=self.cwd, timeout=30,
        )
        return proc.stdout

    # ---- internals -------------------------------------------------------
    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args],
            capture_output=True, text=True,
            cwd=self.cwd, timeout=60, check=check,
        )

    def _safe_join(self, path: str) -> Optional[str]:
        """Resolve ``path`` inside the repo root and refuse escapes."""
        root = pathlib.Path(self.cwd).resolve()
        target = (root / path).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            return None
        return str(target)
