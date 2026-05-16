"""Minimal openhands-style multi-turn rollout for the SWE-bench task.

The model gets a system prompt describing tools, then alternates with a
``user`` (tool result) turn until it either calls ``submit()`` or hits the
turn budget.  All tool calls are parsed out of the assistant's text using a
very forgiving regex (TRL doesn't speak the OpenAI tool-call schema natively
and the SFT model emits prose-with-code-fences anyway).

Per rollout we return:
  * the cumulative ``messages`` list  (with assistant ranges flagged so the
    GRPO loss can be masked to the model's own outputs)
  * the candidate patch string (= ``sandbox.final_patch`` if it submitted,
    else the worktree diff at the turn budget)

The training loop calls this once per (prompt, generation) pair and feeds
the resulting trajectories into a GRPO-style policy update.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from swe_sandbox import SweSandbox, ToolResult


# Forgiving tool-call regexes — match a few permissive variants the SFT
# model emits in practice.

# (a) Strict, well-formed: <tool name="apply_patch">body</tool>
_TOOL_RE = re.compile(
    r"<tool\s+name=\"(?P<name>[a-z_]+)\"[^>]*>(?P<body>.*?)</tool>",
    re.DOTALL,
)

# (b) Unclosed: <tool name="apply_patch">body... [EOF, no </tool>]
#     Happens when the model runs out of generation budget mid-patch.
_TOOL_OPEN_RE = re.compile(
    r"<tool\s+name=\"(?P<name>[a-z_]+)\"[^>]*>(?P<body>.*?)\Z",
    re.DOTALL,
)

# (c) Bleeding-args: model writes the command INSIDE the name attribute, e.g.
#     <tool name="grep -rn 'def foo' .</tool>
#     We treat the first whitespace-separated token as the tool name and the
#     remainder (until ``</tool>`` or EOF) as the body.
_TOOL_BLED_RE = re.compile(
    r"<tool\s+name=\"(?P<name>[a-z_]+)(?P<inline_args>\s+[^\"]*)(?:\"[^>]*>|>|\Z)(?P<body>.*?)(?:</tool>|\Z)",
    re.DOTALL,
)

_FENCED_RE = re.compile(
    r"```(?P<name>read_file|list_files|apply_patch|edit_file|submit|think|bash)\s*\n(?P<body>.*?)```",
    re.DOTALL,
)

_KNOWN_TOOLS = {"list_files", "read_file", "apply_patch", "edit_file",
                "bash", "submit", "think"}
# Tools that take a shell command and should accept inline-args fixups.
_BASH_LIKE = {"bash"}


@dataclass
class TurnRecord:
    """One assistant turn + the tool result that was appended after it."""
    assistant_text: str
    tool_name: str = ""
    tool_arg: str = ""
    tool_result: str = ""


@dataclass
class RolloutResult:
    messages: list[dict[str, str]]
    turns: list[TurnRecord] = field(default_factory=list)
    candidate_patch: str = ""
    submitted: bool = False
    num_turns: int = 0


SYSTEM_PROMPT = """\
You are an expert software engineer fixing a bug in a Python repository. You
have access to the working tree and can call exactly ONE tool per turn. The
content INSIDE the <tool> ... </tool> tags is the literal argument — do not
include placeholder brackets.

Available tools (with real examples):

  <tool name="list_files">.</tool>
      ↑ lists the repo root. Use a real directory path like "src" or "tests".

  <tool name="read_file">setup.py</tool>
      ↑ prints up to 32000 chars of the named file. If the file is larger
      than that, the output is truncated and re-reading the same file gives
      the same truncated output — use `bash` with `grep -n` or `wc` instead
      to find specific lines.

  <tool name="edit_file">PATH: src/foo.py
OLD:
def old_name(a, b):
    return a + b
NEW:
def new_name(a, b, c):
    return a + b + c
</tool>
      ↑ String-replacement edit (PREFERRED). Replaces the first exact match
      of OLD with NEW in the named file. OLD must be unique in the file;
      expand it with surrounding context lines if not. No hunk-counting,
      no line numbers — just literal text. Highly recommended over
      `apply_patch`.

  <tool name="apply_patch">diff --git a/foo.py b/foo.py
  --- a/foo.py
  +++ b/foo.py
  @@ -10,3 +10,3 @@
   context line above
  -old line
  +new line
   context line below
  </tool>
      ↑ Unified-diff fallback. The hunk header `@@ -A,B +C,D @@` must be
      consistent with the actual line counts; LLMs get this wrong often.
      Prefer `edit_file` unless you really need multi-hunk replacement.

  <tool name="bash">grep -rn "click" --include="*.py" .</tool>
      ↑ runs a whitelisted read-only shell command (ls, find, cat, head, tail,
      grep, rg, wc, echo, pwd, tree, file, stat, diff). No shell metachars
      (>, |, ;, &, `, $()) and no mutation tools. Fastest way to locate the
      offending code.

  <tool name="think">your scratch notes</tool>
      ↑ a no-op for free-form reasoning visible only to you.

  <tool name="submit"></tool>
      ↑ ends the dialog. Your accumulated patch becomes the answer.

Rules:
1. Emit EXACTLY ONE <tool name="..."> ... </tool> block per turn — nothing
   else after it on that turn.
2. You have a STRICT turn budget. Spend at most a few turns exploring
   (prefer one `bash` `grep` over many `read_file`s), then patch and submit.
3. After `apply_patch` succeeds, IMMEDIATELY call `<tool name="submit"></tool>`
   on the next turn. Do not keep exploring after patching.

Today's task:"""


def parse_tool_call(text: str) -> tuple[str, str]:
    """Return ``(tool_name, body)`` for the first recognized call, else ('', '').

    Tries variants in order of strictness:
      1. well-formed ``<tool name="x">...</tool>``
      2. fenced ``` ```read_file\n...\n``` ```
      3. inline args ``<tool name="grep ...">...</tool>`` (treat first token as
         the name, glue the rest into the body — useful for the SFT model's
         common `bash` slip)
      4. unclosed ``<tool name="apply_patch">...`` (truncation, take to EOF)
    """
    m = _TOOL_RE.search(text)
    if m:
        return m.group("name").strip(), m.group("body").strip()
    m = _FENCED_RE.search(text)
    if m:
        return m.group("name").strip(), m.group("body").strip()
    m = _TOOL_BLED_RE.search(text)
    if m:
        name = m.group("name").strip()
        inline = (m.group("inline_args") or "").strip()
        body = m.group("body").strip()
        if name in _BASH_LIKE:
            # Glue the bleed-in command back: "bash" + "grep ..." + body
            combined = inline + ("\n" + body if body else "")
            return name, combined.strip()
        if name in _KNOWN_TOOLS:
            # e.g. <tool name="read_file foo.py"> — stitch inline as the arg.
            return name, (inline + (" " + body if body else "")).strip()
        # Unknown name + inline command → the model meant `bash <name> <args>`.
        # e.g. <tool name="grep -rn ...">
        full_cmd = (name + " " + inline + (" " + body if body else "")).strip()
        return "bash", full_cmd
    m = _TOOL_OPEN_RE.search(text)
    if m:
        return m.group("name").strip(), m.group("body").strip()
    return "", ""


def run_one_tool(sandbox: SweSandbox, name: str, arg: str) -> ToolResult:
    if name == "list_files":
        return sandbox.list_files(arg or ".")
    if name == "read_file":
        return sandbox.read_file(arg)
    if name == "apply_patch":
        return sandbox.apply_patch(arg)
    if name == "edit_file":
        return sandbox.edit_file(arg)
    if name == "submit":
        return sandbox.submit()
    if name == "think":
        return ToolResult(True, "noted")
    if name == "bash":
        return sandbox.bash(arg)
    return ToolResult(False, "", f"unknown tool: {name}")


def build_initial_messages(sandbox: SweSandbox, user_problem: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Repo: {sandbox.repo}\n\nIssue:\n{user_problem}\n"},
    ]


def rollout(
    sandbox: SweSandbox,
    user_problem: str,
    generate_fn,           # callable(messages) -> assistant_text
    max_turns: int = 8,
) -> RolloutResult:
    """Drive a multi-turn dialog. ``generate_fn`` is supplied by the caller
    so this module stays decoupled from any specific LLM library.
    """
    sandbox.reset()
    messages = build_initial_messages(sandbox, user_problem)
    res = RolloutResult(messages=messages)

    for _ in range(max_turns):
        assistant_text = generate_fn(messages)
        name, arg = parse_tool_call(assistant_text)
        tr = run_one_tool(sandbox, name, arg) if name else ToolResult(
            False, "",
            "no tool call detected. Emit exactly one <tool name=\"NAME\">BODY"
            "</tool>. NAME must be a single word (one of list_files, "
            "read_file, apply_patch, bash, think, submit) and BODY goes "
            "between the tags. Don't put the command inside the name "
            "attribute. End the tag with </tool>.",
        )

        res.turns.append(TurnRecord(
            assistant_text=assistant_text,
            tool_name=name,
            tool_arg=arg,
            tool_result=tr.to_user_text(),
        ))
        messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": tr.to_user_text()})

        if sandbox.submitted:
            break

    res.num_turns = len(res.turns)
    res.submitted = sandbox.submitted
    # If the model never submitted, we still take the current diff so it's
    # not just zero. This is a soft policy choice — prevents starving the
    # gradient when the model hasn't learned to submit yet.
    res.candidate_patch = sandbox.final_patch or sandbox.current_diff()
    return res
