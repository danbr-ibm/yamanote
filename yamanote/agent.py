"""Native tool-calling agent loop over OpenRouter.

An Agent is given a system prompt, a task, a root directory it may touch, and
a set of tools. It loops: model → tool calls → results → model, until the
model calls `finish` (or a step/time/budget limit trips). Every model turn and
tool call is reported to a Recorder so the dashboard can show the run live.

With a Journal the loop is durable: the conversation, each model reply and each
tool call's intent are appended before anything acts on them, so a new process
can `resume` the run from the journal. A tool call that was in flight when the
process died is re-run only if its tool is in REPLAY_SAFE_TOOLS; otherwise the
agent gets an interruption report as the call's result and investigates.

File tools are confined to `root` (paths are resolved and must stay inside).
`run` executes shell commands with `root` as cwd and secrets stripped from the
environment; like the Claude Code CLI it replaces, it is not a hard sandbox.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from . import settings
from .llm import Client, LLMError

READ_TOOLS = ("list_dir", "read_file", "grep")
WRITE_TOOLS = ("write_file", "edit_file")
RUN_TOOLS = ("run",)
# Tools whose interrupted call can simply run again: reads, and a whole-file
# write (same content, same result). edit_file is not (the edit may already be
# applied, so old_string is gone or matches twice) and neither is run.
REPLAY_SAFE_TOOLS = frozenset(READ_TOOLS + ("write_file",))

MAX_TOOL_OUTPUT = 12_000
MAX_READ_CHARS = 60_000
CONTEXT_SOFT_LIMIT = 320_000  # characters of history before old tool output is trimmed
KEEP_RECENT_TOOL_RESULTS = 8
EVIDENCE_OUTPUT_CHARS = 1500  # tail of a cut-off command's output shown on resume
SNAPSHOT_STATUS_LINES = 20    # git status lines shown on resume
# Set on every `run` command (inherited by what it starts) so processes left
# behind by a dead factory are recognised by more than a pid that may be reused.
RUN_TAG_ENV = "YAMANOTE_RUN_TAG"
RUN_OUTPUT_PREFIX = "yamanote-run-"
_SECRET_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.I)


class Recorder(Protocol):
    def step(self, kind: str, name: str, detail: str, *, tokens_in: int = 0,
             tokens_out: int = 0, cost_usd: float = 0.0, model: str = "") -> None: ...


class NullRecorder:
    def step(self, *args, **kwargs) -> None:
        pass


class Journal(Protocol):
    """Append-only, durable record of a run. Entries are written before the
    loop acts on them; `resume` rebuilds the run from them in order.

    kinds: start {messages} · model {message, usage, elapsed} · message {message}
    · call {key, name, args} (before the tool runs) · spawn {key, pgid, output, tag}
    (a `run` command's process group, output file and RUN_TAG_ENV value)
    · result {key, content} · reaped {pgids} (left running by a dead process,
    stopped at startup) · resume {interrupted, replayed} · end {status}"""
    def append(self, kind: str, data: dict) -> None: ...


class NullJournal:
    def append(self, kind: str, data: dict) -> None:
        pass


@dataclass
class AgentResult:
    status: str  # ok | error | timeout | cancelled | budget | max_steps
    summary: str = ""
    result: dict | None = None
    steps: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    model: str = ""
    error: str = ""
    error_code: int | None = None  # HTTP status of a fatal LLM error (401/402/429...)
    cached_tokens: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "ok"


# ─── Tool schemas ────────────────────────────────────────────────────────────

def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }}


TOOL_SCHEMAS = {
    "list_dir": _fn("list_dir", "List files under a directory (relative to the project root), git-ignored files excluded.",
                    {"path": {"type": "string", "description": "Directory, default '.'"},
                     "depth": {"type": "integer", "description": "Max depth, default 2"}}, []),
    "read_file": _fn("read_file", "Read a text file with line numbers. Use start/end to page through large files.",
                     {"path": {"type": "string"},
                      "start": {"type": "integer", "description": "First line (1-based)"},
                      "end": {"type": "integer", "description": "Last line (inclusive)"}}, ["path"]),
    "grep": _fn("grep", "Search file contents with a regular expression. Returns file:line:text matches.",
                {"pattern": {"type": "string"},
                 "path": {"type": "string", "description": "File or directory, default '.'"},
                 "glob": {"type": "string", "description": "Optional filename glob, e.g. '*.py'"}}, ["pattern"]),
    "write_file": _fn("write_file", "Create or overwrite a file with the given content.",
                      {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    "edit_file": _fn("edit_file", "Replace an exact string in a file. old_string must match exactly and be unique unless replace_all is true.",
                     {"path": {"type": "string"}, "old_string": {"type": "string"},
                      "new_string": {"type": "string"}, "replace_all": {"type": "boolean"}},
                     ["path", "old_string", "new_string"]),
    "run": _fn("run", "Run a shell command (bash) in the project root. Use for builds, tests and inspection. Output is truncated.",
               {"command": {"type": "string"},
                "timeout": {"type": "integer", "description": f"Seconds, max {settings.TOOL_TIMEOUT_SECONDS}"}},
               ["command"]),
}


def finish_schema(result_schema: dict | None) -> dict:
    props = {"summary": {"type": "string", "description": "One short paragraph: what you did / concluded."}}
    required = ["summary"]
    if result_schema:
        props["result"] = result_schema
        required.append("result")
    return _fn("finish", "Call exactly once when the task is complete to report your outcome.", props, required)


# ─── Agent ───────────────────────────────────────────────────────────────────

@dataclass
class Agent:
    role: str
    model: str
    system: str
    root: str
    tools: tuple[str, ...] = READ_TOOLS
    result_schema: dict | None = None
    recorder: Recorder = field(default_factory=NullRecorder)
    client: Client | None = None
    fallbacks: list[str] = field(default_factory=list)
    max_steps: int = settings.AGENT_MAX_STEPS
    timeout: float = settings.AGENT_TIMEOUT_SECONDS
    budget_usd: float = settings.RUN_BUDGET_USD
    cancel: threading.Event = field(default_factory=threading.Event)
    max_tokens: int = 8000
    journal: Journal = field(default_factory=NullJournal)

    def __post_init__(self):
        self.client = self.client or Client()
        self.root = os.path.realpath(self.root)
        self._pgids: set[int] = set()  # every process group started by `run`, killed when the run ends
        self.resumed: dict | None = None  # set by resume(): interrupted/replayed call counts
        self._evidence: dict[str, str] = {}  # interrupted call key -> what resume found out about it
        self._calling: str | None = None  # key of the tool call being dispatched
        self._kept_outputs: list[str] = []  # output files of cancelled commands, kept for a resume

    # -- loop --

    def run(self, task: str) -> AgentResult:
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": task}]
        self.journal.append("start", {"messages": messages})
        return self._guarded(messages, AgentResult(status="ok", model=self.model), 0.0, [], set())

    def resume(self, entries: list[dict], note: str = "") -> AgentResult:
        """Continue a run from its journal entries ({kind, data}, oldest first).

        Calls of the last model reply that never started run normally; one that
        started but has no result is replayed if REPLAY_SAFE_TOOLS allows it,
        and otherwise answered with an interruption report. `note` tells the
        agent what else changed while it was down."""
        messages, res, elapsed, tail, started, spawns, last_ts, reaped = _replay(entries, self.model)
        if not messages:
            res.status, res.error = "error", "journal has no start entry"
            return res
        # A `run` journaled as called but with no process spawned never ran.
        started = {k for k in started if _tool_of(tail, k)[0] != "run" or k in spawns}
        interrupted = [k for k in started if not _replay_safe(tail, k)]
        replayed = [k for k in started if k not in interrupted]
        # Gather evidence before anything runs (this also stops a command still running).
        self._evidence = {k: self._call_evidence(*_tool_of(tail, k), spawns.get(k), reaped) for k in interrupted}
        note = "\n\n".join(part for part in (note, self._workspace_snapshot(last_ts)) if part)
        self.journal.append("resume", {"interrupted": interrupted, "replayed": replayed})
        self.resumed = {"interrupted": len(interrupted), "replayed": len(replayed)}
        self.recorder.step("resume", self.role, f"Resumed from the journal after {res.steps} model turns: "
                           f"{len(interrupted)} interrupted call(s) reported, {len(replayed)} replayed")
        return self._guarded(messages, res, elapsed, tail, started, note)

    def _guarded(self, *args) -> AgentResult:
        try:
            res = self._loop(*args)
        finally:
            self._kill_process_groups()
        # A cancelled run (shutdown or cancel) is left open so it can be resumed.
        if res.status != "cancelled":
            self.journal.append("end", {"status": res.status})
        return res

    def abandon(self) -> None:
        """A cancelled run that will not be resumed: close its journal and
        remove the output files kept for a resume."""
        for path in self._kept_outputs:
            _unlink(path)
        self._kept_outputs.clear()
        self.journal.append("end", {"status": "cancelled"})

    def _say(self, messages: list[dict], content: str) -> None:
        msg = {"role": "user", "content": content}
        self.journal.append("message", {"message": msg})
        messages.append(msg)

    def _loop(self, messages: list[dict], res: AgentResult, elapsed: float,
              calls: list[dict], started: set[str], note: str = "") -> AgentResult:
        schemas = [TOOL_SCHEMAS[t] for t in self.tools] + [finish_schema(self.result_schema)]
        start = time.monotonic() - elapsed
        nudged = False
        empty_replies = 0

        while True:
            if calls:
                finished = self._call_tools(calls, messages, started)
                started = set()
                if self.cancel.is_set():
                    res.status = "cancelled"
                    return res
                if finished is not None:
                    return self._finish(res, finished)
                _trim_history(messages)
                calls = []
            if note:
                self._say(messages, note)
                note = ""

            if self.cancel.is_set():
                res.status = "cancelled"
                return res
            if time.monotonic() - start > self.timeout:
                res.status, res.error = "timeout", f"exceeded {self.timeout:.0f}s"
                return res
            if res.steps >= self.max_steps:
                res.status, res.error = "max_steps", f"exceeded {self.max_steps} steps"
                return res
            if res.cost_usd >= self.budget_usd:
                res.status, res.error = "budget", f"run cost ${res.cost_usd:.2f} reached limit ${self.budget_usd:.2f}"
                return res

            try:
                completion = self.client.chat(self.model, messages, tools=schemas,
                                              fallbacks=self.fallbacks, max_tokens=self.max_tokens)
            except LLMError as e:
                res.status, res.error, res.error_code = "error", str(e), e.status
                self.recorder.step("error", "llm", str(e)[:500], model=self.model)
                return res

            msg = dict(completion.message)
            msg["role"] = "assistant"
            msg.pop("refusal", None)
            msg.pop("reasoning_details", None)
            usage = {"tokens_in": completion.tokens_in, "tokens_out": completion.tokens_out,
                     "cost_usd": completion.cost_usd, "cached_tokens": completion.cached_tokens,
                     "model": completion.model}
            self.journal.append("model", {"message": msg, "usage": usage,
                                          "elapsed": time.monotonic() - start})
            _count(res, usage)
            messages.append(msg)
            thought = completion.text.strip()
            empty = not thought and not completion.tool_calls
            self.recorder.step("model", completion.model,
                               thought[:2000] if not empty else f"(empty reply; finish_reason={completion.finish_reason or '?'})",
                               tokens_in=completion.tokens_in, tokens_out=completion.tokens_out,
                               cost_usd=completion.cost_usd, model=completion.model,
                               cached_tokens=completion.cached_tokens)

            if empty and empty_replies < 2:
                # Some models occasionally return nothing at all; re-ask directly
                # rather than ending the run (which costs a retry with backoff).
                empty_replies += 1
                self._say(messages, "Your last reply was empty. Continue: use a tool, or call `finish` now with your result.")
                continue
            if not completion.tool_calls:
                if completion.finish_reason == "length":
                    self._say(messages, "Your reply was cut off. Continue, using tools; keep each reply short.")
                    continue
                if not nudged:
                    nudged = True
                    self._say(messages, "Continue the task using the tools. When you are done, call the `finish` tool.")
                    continue
                # The model insists on plain text: accept it as the outcome.
                res.summary = thought
                res.result = _extract_json(thought) if self.result_schema else None
                if self.result_schema and res.result is None:
                    res.status, res.error = "error", "agent ended without calling finish"
                return res
            calls = completion.tool_calls

    def _call_tools(self, calls: list[dict], messages: list[dict], started: set[str]) -> dict | None:
        """Run a model reply's tool calls in order and return finish's arguments
        if it was called. Calls that already have a result in `messages` (a
        resumed run) are skipped; those in `started` were cut off mid-call."""
        last = max(i for i, m in enumerate(messages) if m.get("role") == "assistant")
        answered = {m.get("tool_call_id") for m in messages[last + 1:] if m.get("role") == "tool"}
        finished = None
        for i, call in enumerate(calls):
            key = _call_key(call, i)
            fn = call.get("function") or {}
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments must be an object")
            except (json.JSONDecodeError, ValueError) as e:
                output = f"ERROR: could not parse arguments as JSON ({e}). Retry with valid JSON."
                args = None
            if args is not None and name == "finish":
                finished = args
            if key in answered:
                continue
            if args is not None and name == "finish":
                output = "ok"
            elif args is not None and key in started and name not in REPLAY_SAFE_TOOLS:
                output = _interruption_report(name, self._evidence.get(key, ""))
                self.recorder.step("interrupted", name, _describe_call(name, args, output))
            elif args is not None:
                if key not in started:
                    self.journal.append("call", {"key": key, "name": name, "args": args})
                self._calling = key
                try:
                    output = self._dispatch(name, args)
                finally:
                    self._calling = None
                if self.cancel.is_set():
                    return None  # the result may be cut short: leave the call open for resume
                self.recorder.step("tool", name, _describe_call(name, args, output)
                                   + ("\n(replayed after restart)" if key in started else ""))
            self.journal.append("result", {"key": key, "content": output})
            messages.append({"role": "tool", "tool_call_id": key, "content": output})
        return finished

    def _finish(self, res: AgentResult, finished: dict) -> AgentResult:
        res.summary = str(finished.get("summary", "")).strip()
        result = finished.get("result")
        if isinstance(result, str):
            result = _extract_json(result) or {"value": result}
        res.result = result if isinstance(result, dict) else None
        if self.result_schema and res.result is None:
            res.status, res.error = "error", "finish called without a result object"
        self.recorder.step("finish", self.role, res.summary[:2000])
        return res

    def stop(self):
        """Cancel the run and kill any running shell command (and anything it backgrounded)."""
        self.cancel.set()
        self._kill_process_groups()

    def _kill_process_groups(self):
        for pgid in list(self._pgids):
            _kill_group(pgid)
            self._pgids.discard(pgid)

    # -- tools --

    def _dispatch(self, name: str, args: dict) -> str:
        if name not in self.tools:
            return f"ERROR: tool '{name}' is not available to the {self.role} agent."
        try:
            return getattr(self, f"_tool_{name}")(**args)
        except TypeError as e:
            return f"ERROR: bad arguments for {name}: {e}"
        except PathError as e:
            return f"ERROR: {e}"
        except OSError as e:
            return f"ERROR: {e}"

    def _path(self, path: str | None) -> Path:
        raw = (path or ".").strip()
        candidate = raw if os.path.isabs(raw) else os.path.join(self.root, raw)
        real = os.path.realpath(candidate)
        if real != self.root and not real.startswith(self.root + os.sep):
            raise PathError(f"path '{raw}' is outside the project root {self.root}")
        return Path(real)

    def _rel(self, p: Path) -> str:
        return os.path.relpath(p, self.root)

    def _tool_list_dir(self, path: str = ".", depth: int = 2) -> str:
        base = self._path(path)
        if not base.is_dir():
            return f"ERROR: {path} is not a directory"
        files = _git_ls_files(self.root, self._rel(base))
        if files is None:  # not a git repo: walk, skipping junk
            files = []
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in
                               ("node_modules", "venv", ".venv", "__pycache__", "dist", "build")]
                for f in filenames:
                    files.append(os.path.relpath(os.path.join(dirpath, f), self.root))
        depth = max(1, int(depth or 2))
        base_rel = self._rel(base)
        prefix_parts = 0 if base_rel == "." else len(Path(base_rel).parts)
        shown, dirs = [], set()
        for f in sorted(files):
            parts = Path(f).parts
            rel_parts = parts[prefix_parts:]
            if len(rel_parts) <= depth:
                shown.append(f)
            else:
                dirs.add(str(Path(*parts[:prefix_parts + depth])) + "/")
        lines = sorted(set(shown) | dirs)
        out = "\n".join(lines[:800])
        if len(lines) > 800:
            out += f"\n... ({len(lines) - 800} more)"
        return out or "(empty)"

    def _tool_read_file(self, path: str, start: int | None = None, end: int | None = None) -> str:
        p = self._path(path)
        if not p.is_file():
            return f"ERROR: {path} does not exist or is not a file"
        text = p.read_text(errors="replace")
        lines = text.splitlines()
        s = max(1, int(start or 1))
        e = min(len(lines), int(end or len(lines)))
        out, size = [], 0
        for n in range(s, e + 1):
            line = f"{n:>5}\t{lines[n - 1]}"
            size += len(line) + 1
            if size > MAX_READ_CHARS:
                out.append(f"... (truncated at line {n - 1} of {len(lines)}; use start/end to read more)")
                break
            out.append(line)
        return "\n".join(out) if out else f"(file has {len(lines)} lines)"

    def _tool_grep(self, pattern: str, path: str = ".", glob: str | None = None) -> str:
        target = self._path(path)
        if shutil.which("rg"):
            cmd = ["rg", "-n", "--no-heading", "-S", "--max-columns", "300", "-e", pattern]
            if glob:
                cmd += ["-g", glob]
            cmd.append(str(target))
        else:
            cmd = ["grep", "-rnE", "--exclude-dir=.git", "--exclude-dir=node_modules", "-e", pattern]
            if glob:
                cmd.append(f"--include={glob}")
            cmd.append(str(target))
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=self.root).stdout
        except subprocess.TimeoutExpired:
            return "ERROR: search timed out; narrow the path or pattern"
        out = out.replace(self.root + os.sep, "")
        return _truncate(out) or "(no matches)"

    def _tool_write_file(self, path: str, content: str) -> str:
        p = self._path(path)
        if ".git" in Path(self._rel(p)).parts:
            return "ERROR: writing inside .git is not allowed"
        p.parent.mkdir(parents=True, exist_ok=True)
        existed = p.exists()
        p.write_text(content)
        return f"{'Updated' if existed else 'Created'} {self._rel(p)} ({len(content.splitlines())} lines)"

    def _tool_edit_file(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        p = self._path(path)
        if not p.is_file():
            return f"ERROR: {path} does not exist"
        text = p.read_text(errors="replace")
        count = text.count(old_string)
        if not old_string or count == 0:
            return "ERROR: old_string not found. Read the file again and copy the exact text."
        if count > 1 and not replace_all:
            return f"ERROR: old_string occurs {count} times; add surrounding context or set replace_all."
        p.write_text(text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1))
        return f"Edited {self._rel(p)} ({count if replace_all else 1} replacement{'s' if replace_all and count > 1 else ''})"

    def _tool_run(self, command: str, timeout: int | None = None) -> str:
        """Output goes to a temp file, not a pipe: a process the command leaves
        running in the background (a dev server) would otherwise hold the pipe
        open and block forever. Backgrounded processes keep running for later
        commands and are killed when the agent's run ends."""
        if self.cancel.is_set():
            return "[exit cancelled]"
        limit = min(int(timeout or 120), settings.TOOL_TIMEOUT_SECONDS)
        env = {k: v for k, v in os.environ.items() if not _SECRET_ENV.search(k) and k != "CLAUDECODE"}
        env["GIT_TERMINAL_PROMPT"] = "0"
        # Named, so a resumed run can show what a cut-off command printed.
        tag = os.urandom(8).hex()
        env[RUN_TAG_ENV] = tag
        fd, out_path = tempfile.mkstemp(prefix=RUN_OUTPUT_PREFIX, suffix=".log")
        with os.fdopen(fd, "w+", errors="replace") as out_file:
            proc = subprocess.Popen(["bash", "-c", command], cwd=self.root, env=env, text=True,
                                    stdout=out_file, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
            self._pgids.add(proc.pid)  # new session: pgid == pid
            if self._calling:
                self.journal.append("spawn", {"key": self._calling, "pgid": proc.pid, "output": out_path,
                                              "tag": tag})
            deadline = time.monotonic() + limit
            rc: int | str
            while True:
                try:
                    rc = proc.wait(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    if self.cancel.is_set():
                        _kill_group(proc.pid)
                        proc.wait()
                        rc = "cancelled"
                        break
                    if time.monotonic() >= deadline:
                        _kill_group(proc.pid)
                        proc.wait()
                        rc = f"timeout after {limit}s"
                        break
            out_file.seek(0)
            out = out_file.read(MAX_TOOL_OUTPUT * 4)
        if self.cancel.is_set():  # the call stays open in the journal; resume (or abandon) removes the file
            self._kept_outputs.append(out_path)
        else:
            _unlink(out_path)
        return f"[exit {rc}]\n{_truncate(out)}"

    # -- resume evidence --

    def _call_evidence(self, name: str, args: dict, spawn: dict | None, reaped: set[int]) -> str:
        """Facts about an interrupted call, gathered without the model."""
        try:
            if name == "run" and spawn:
                return _run_evidence(spawn, reaped)
            if name == "edit_file":
                return self._edit_evidence(args)
        except (OSError, PathError, TypeError, ValueError) as e:
            return f"(Could not inspect the call: {e})"
        return ""

    def _edit_evidence(self, args: dict) -> str:
        p = self._path(args.get("path"))
        if not p.is_file():
            return f"{self._rel(p)} does not exist now."
        text = p.read_text(errors="replace")
        old, new = args.get("old_string") or "", args.get("new_string") or ""
        n_old, n_new = text.count(old) if old else 0, text.count(new) if new else 0
        facts = f"{self._rel(p)} now contains old_string {n_old} time(s) and new_string {n_new} time(s)"
        if n_old == 0 and n_new > 0 and old not in new:
            return facts + ": the edit looks applied."
        if n_old > 0 and n_new == 0:
            return facts + ": the edit looks not applied."
        return facts + "."

    def _workspace_snapshot(self, last_ts: float | None) -> str:
        """How long the run was down and what the worktree looks like now, in a few lines."""
        parts = []
        if last_ts:
            parts.append(f"You were stopped {_ago(time.time() - last_ts)} ago.")
        status = _git(self.root, "status", "--short")
        if status is None:
            return " ".join(parts)
        lines = status.splitlines()
        shown = "\n".join(lines[:SNAPSHOT_STATUS_LINES]) or "(clean)"
        if len(lines) > SNAPSHOT_STATUS_LINES:
            shown += f"\n… {len(lines) - SNAPSHOT_STATUS_LINES} more"
        parts.append(f"Worktree now (git status --short):\n{shown}")
        stat = (_git(self.root, "diff", "HEAD", "--stat") or "").strip().splitlines()
        if stat:
            parts.append(f"Uncommitted vs HEAD: {stat[-1].strip()}")
        head = (_git(self.root, "log", "-1", "--format=%h %s (%cr)") or "").strip()
        if head:
            parts.append(f"Last commit: {head}")
        return "\n".join(parts)


class PathError(ValueError):
    pass


# ─── helpers ─────────────────────────────────────────────────────────────────

def _kill_group(pgid: int) -> None:
    """SIGTERM then SIGKILL a process group, whether or not its leader is still alive."""
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        return
    for _ in range(20):
        time.sleep(0.05)
        try:
            os.killpg(pgid, 0)
        except (ProcessLookupError, PermissionError, OSError):
            return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _git_ls_files(root: str, sub: str) -> list[str] | None:
    try:
        r = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "--", sub],
                           cwd=root, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    return [l for l in r.stdout.splitlines() if l and not l.startswith(".worktrees/")]


def _truncate(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2:]
    return f"{head}\n... [{len(text) - limit} characters omitted] ...\n{tail}"


def _describe_call(name: str, args: dict, output: str) -> str:
    if name == "run":
        head = f"$ {args.get('command', '')}"
    elif name in ("read_file", "write_file", "edit_file", "list_dir"):
        head = str(args.get("path", "."))
        if name == "read_file" and (args.get("start") or args.get("end")):
            head += f" [{args.get('start', 1)}-{args.get('end', '…')}]"
    elif name == "grep":
        head = f"/{args.get('pattern', '')}/ {args.get('path', '.')}"
    else:
        head = json.dumps(args)[:200]
    first = output.strip().splitlines()[0] if output.strip() else ""
    return f"{head}\n→ {first[:300]}"


def _call_key(call: dict, index: int) -> str:
    """A tool call's id, or its position when the provider sent none."""
    return call.get("id") or f"call_{index}"


def _count(res: AgentResult, usage: dict) -> None:
    res.steps += 1
    res.tokens_in += usage.get("tokens_in", 0)
    res.tokens_out += usage.get("tokens_out", 0)
    res.cost_usd += usage.get("cost_usd", 0.0)
    res.cached_tokens += usage.get("cached_tokens", 0)
    res.model = usage.get("model") or res.model


def _replay(entries: list[dict], model: str):
    """Rebuild a run from its journal: the messages so far, the counters
    (steps and spend count against the resumed run's limits), the elapsed
    time, the last reply's tool calls, the calls that started without
    finishing, the processes those started, and when the journal was last
    written."""
    messages: list[dict] = []
    res = AgentResult(status="ok", model=model)
    elapsed = 0.0
    tail: list[dict] = []
    started: set[str] = set()
    spawns: dict[str, dict] = {}
    reaped: set[int] = set()
    last_ts = None
    for entry in entries:
        kind, data = entry["kind"], entry["data"]
        last_ts = entry.get("ts", last_ts)
        if kind == "start":
            messages = list(data["messages"])
        elif kind == "model":
            messages.append(data["message"])
            _count(res, data.get("usage") or {})
            elapsed = data.get("elapsed", elapsed)
            tail, started = data["message"].get("tool_calls") or [], set()
        elif kind == "message":
            messages.append(data["message"])
            tail, started = [], set()
        elif kind == "call":
            started.add(data["key"])
        elif kind == "spawn":
            spawns[data["key"]] = data
        elif kind == "reaped":
            reaped.update(data.get("pgids", []))
        elif kind == "result":
            started.discard(data["key"])
            spawns.pop(data["key"], None)
            messages.append({"role": "tool", "tool_call_id": data["key"], "content": data["content"]})
    spawns = {k: v for k, v in spawns.items() if k in started}
    return messages, res, elapsed, tail, started, spawns, last_ts, reaped


def _tool_of(calls: list[dict], key: str) -> tuple[str, dict]:
    """(name, args) of the call with `key` in a model reply."""
    for i, call in enumerate(calls):
        if _call_key(call, i) == key:
            fn = call.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            return fn.get("name", ""), args if isinstance(args, dict) else {}
    return "", {}


def _replay_safe(calls: list[dict], key: str) -> bool:
    return _tool_of(calls, key)[0] in REPLAY_SAFE_TOOLS


def _interruption_report(name: str, evidence: str = "") -> str:
    report = (f"INTERRUPTED: the process running you stopped while this `{name}` call was in progress, so it "
              "may have completed, partly completed, or not run at all. It was not re-run because it is not "
              "safe to replay. Investigate before continuing: check the state this call would have changed "
              "(git status, git diff, the files involved), then redo it only if it is still needed.")
    return f"{report}\n{evidence}" if evidence else report


def _run_evidence(spawn: dict, reaped: set[int]) -> str:
    """Stop a cut-off command that outlived its agent, and show its last output."""
    path = spawn.get("output")
    if spawn.get("pgid") in reaped:
        lines = ["The command was still running when the factory restarted; it has been stopped."]
    elif _stop_pids(_tagged_pids(spawn)):
        lines = ["The command was still running when you resumed; it has been stopped."]
    else:
        lines = ["The command is no longer running; its exit status is unknown."]
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - EVIDENCE_OUTPUT_CHARS))
            out = f.read().decode(errors="replace")
    except (OSError, TypeError):
        lines.append("Its output was not kept.")
    else:
        lines.append(f"Its last output:\n{out}" if out.strip() else "It had printed nothing.")
        _unlink(path)
    return "\n".join(lines)


def reap(entries: list[dict]) -> list[int]:
    """Stop every process a dead run's `run` commands left behind (dev
    servers included). Returns the process groups that had any."""
    stopped = []
    for entry in entries:
        if entry["kind"] == "spawn" and _stop_pids(_tagged_pids(entry["data"])):
            stopped.append(entry["data"]["pgid"])
    return stopped


def sweep_run_outputs(max_age: float) -> int:
    """Remove `run` output files older than max_age seconds (kept for a resume that never came)."""
    removed, cutoff = 0, time.time() - max_age
    for path in Path(tempfile.gettempdir()).glob(f"{RUN_OUTPUT_PREFIX}*.log"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def _tagged_pids(spawn: dict) -> list[int]:
    """Live processes in the spawn's group that carry its tag: a reused pid,
    or a process from another run, is never matched."""
    pgid, tag = spawn.get("pgid"), spawn.get("tag")
    if not pgid or not tag:
        return []
    marker = f"{RUN_TAG_ENV}={tag}".encode()
    pids = []
    try:
        candidates = [d for d in os.listdir("/proc") if d.isdigit()]
    except OSError:
        return []
    for pid in candidates:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            fields = stat[stat.rfind(")") + 2:].split()  # state, ppid, pgrp, ...
            if len(fields) < 3 or fields[2] != str(pgid) or fields[0] == "Z":
                continue
            if marker in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
                pids.append(int(pid))
        except OSError:
            continue
    return pids


def _stop_pids(pids: list[int]) -> bool:
    """SIGTERM, then SIGKILL whatever is left after a second. True if there was anything to stop."""
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    for _ in range(20):
        if not any(_alive(pid) for pid in pids):
            break
        time.sleep(0.05)
    for pid in pids:
        if _alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    return bool(pids)


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return state[state.rfind(")") + 2:][:1] != "Z"


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _git(root: str, *args: str) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout if r.returncode == 0 else None


def _ago(seconds: float) -> str:
    if seconds < 90:
        return f"{max(1, round(seconds))}s"
    if seconds < 5400:
        return f"{round(seconds / 60)} min"
    return f"{seconds / 3600:.1f} h"


def _trim_history(messages: list[dict]) -> None:
    """Replace old tool outputs with stubs once the history gets large."""
    total = sum(len(str(m.get("content") or "")) for m in messages)
    if total <= CONTEXT_SOFT_LIMIT:
        return
    tool_idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    for i in tool_idx[:-KEEP_RECENT_TOOL_RESULTS]:
        content = str(messages[i].get("content") or "")
        if len(content) > 300:
            messages[i]["content"] = content[:200] + f"\n[... trimmed {len(content) - 200} chars of old output]"
            total -= len(content) - 250
            if total <= CONTEXT_SOFT_LIMIT * 0.7:
                break


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of free text (handles ```json fences)."""
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [fence.group(1)] if fence else []
    start = text.find("{")
    if start != -1:
        candidates.append(text[start: text.rfind("}") + 1])
    for c in candidates:
        try:
            value = json.loads(c)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return None


def single_shot(client: Client, model: str, system: str, prompt: str, *,
                fallbacks: list[str] | None = None, recorder: Recorder | None = None,
                max_tokens: int = 4000) -> tuple[dict | None, AgentResult]:
    """One model call that must answer with a JSON object (no tools)."""
    recorder = recorder or NullRecorder()
    res = AgentResult(status="ok", model=model)
    try:
        c = client.chat(model, [{"role": "system", "content": system},
                                {"role": "user", "content": prompt}],
                        fallbacks=fallbacks, max_tokens=max_tokens,
                        response_format={"type": "json_object"})
    except LLMError as e:
        res.status, res.error, res.error_code = "error", str(e), e.status
        recorder.step("error", "llm", str(e)[:500], model=model)
        return None, res
    res.steps, res.model = 1, c.model
    res.tokens_in, res.tokens_out, res.cost_usd = c.tokens_in, c.tokens_out, c.cost_usd
    res.cached_tokens = c.cached_tokens
    recorder.step("model", c.model, c.text[:2000], tokens_in=c.tokens_in,
                  tokens_out=c.tokens_out, cost_usd=c.cost_usd, model=c.model, cached_tokens=c.cached_tokens)
    data = _extract_json(c.text)
    if data is None:
        res.status, res.error = "error", "model did not return a JSON object"
    res.result, res.summary = data, (data or {}).get("summary", "") if data else c.text[:500]
    return data, res
