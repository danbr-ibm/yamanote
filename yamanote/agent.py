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
    · call {key, name, args} (before the tool runs) · result {key, content}
    · resume {interrupted, replayed} · end {status}"""
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
        messages, res, elapsed, tail, started = _replay(entries, self.model)
        if not messages:
            res.status, res.error = "error", "journal has no start entry"
            return res
        interrupted = [k for k in started if not _replay_safe(tail, k)]
        replayed = [k for k in started if k not in interrupted]
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
                output = _interruption_report(name)
                self.recorder.step("interrupted", name, _describe_call(name, args, output))
            elif args is not None:
                if key not in started:
                    self.journal.append("call", {"key": key, "name": name, "args": args})
                output = self._dispatch(name, args)
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
        with tempfile.TemporaryFile(mode="w+", errors="replace") as out_file:
            proc = subprocess.Popen(["bash", "-c", command], cwd=self.root, env=env, text=True,
                                    stdout=out_file, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
            self._pgids.add(proc.pid)  # new session: pgid == pid
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
        return f"[exit {rc}]\n{_truncate(out)}"


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


def _replay(entries: list[dict], model: str) -> tuple[list[dict], AgentResult, float, list[dict], set[str]]:
    """Rebuild a run from its journal: the messages so far, the counters
    (steps and spend count against the resumed run's limits), the elapsed
    time, the last reply's tool calls, and the calls that started without
    finishing."""
    messages: list[dict] = []
    res = AgentResult(status="ok", model=model)
    elapsed = 0.0
    tail: list[dict] = []
    started: set[str] = set()
    for entry in entries:
        kind, data = entry["kind"], entry["data"]
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
        elif kind == "result":
            started.discard(data["key"])
            messages.append({"role": "tool", "tool_call_id": data["key"], "content": data["content"]})
    return messages, res, elapsed, tail, started


def _replay_safe(calls: list[dict], key: str) -> bool:
    for i, call in enumerate(calls):
        if _call_key(call, i) == key:
            return (call.get("function") or {}).get("name") in REPLAY_SAFE_TOOLS
    return False


def _interruption_report(name: str) -> str:
    report = (f"INTERRUPTED: the process running you stopped while this `{name}` call was in progress, so it "
              "may have completed, partly completed, or not run at all. It was not re-run because it is not "
              "safe to replay. Investigate before continuing: check the state this call would have changed "
              "(git status, git diff, the files involved), then redo it only if it is still needed.")
    if name == "run":
        report += " Anything the command started in the background may or may not still be running."
    return report


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
