"""Agent loop and tool sandbox tests (offline)."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path

from tests.helpers import FakeClient, finish, reply, sh, tool_call
from yamanote.agent import (READ_TOOLS, RUN_TOOLS, WRITE_TOOLS, Agent, _boot_id, _extract_json, _kill_group,
                            _trim_history)
from yamanote.llm import _parse, _with_cache_breakpoints

SYSTEM = "ROLE: Builder. test agent"


class Recorder:
    def __init__(self):
        self.steps = []

    def step(self, kind, name, detail, **kw):
        self.steps.append((kind, name, detail, kw))


def scripted(*turns):
    """FakeClient whose builder replies with `turns` in order."""
    seq = list(turns)
    return FakeClient({"builder": lambda m, t: seq[t] if t < len(seq) else finish("done")})


class ToolSandboxTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "proj"
        self.root.mkdir()
        (self.root / "a.py").write_text("x = 1\nx = 1\ny = 2\n")
        self.outside = Path(self.tmp.name) / "secret.txt"
        self.outside.write_text("top secret")
        self.agent = Agent(role="build", model="m", system=SYSTEM, root=str(self.root),
                           tools=READ_TOOLS + WRITE_TOOLS + RUN_TOOLS, client=scripted())

    def tearDown(self):
        self.tmp.cleanup()

    def test_paths_outside_root_are_refused(self):
        for path in ("../secret.txt", str(self.outside), "sub/../../secret.txt"):
            out = self.agent._dispatch("read_file", {"path": path})
            self.assertIn("outside the project root", out)
        out = self.agent._dispatch("write_file", {"path": "../evil.txt", "content": "x"})
        self.assertIn("outside", out)
        self.assertFalse((Path(self.tmp.name) / "evil.txt").exists())

    def test_symlink_escape_is_refused(self):
        os.symlink(self.outside, self.root / "link.txt")
        self.assertIn("outside", self.agent._dispatch("read_file", {"path": "link.txt"}))

    def test_edit_requires_unique_match(self):
        out = self.agent._dispatch("edit_file", {"path": "a.py", "old_string": "x = 1", "new_string": "x = 3"})
        self.assertIn("occurs 2 times", out)
        out = self.agent._dispatch("edit_file", {"path": "a.py", "old_string": "y = 2", "new_string": "y = 5"})
        self.assertIn("Edited", out)
        self.assertIn("y = 5", (self.root / "a.py").read_text())
        out = self.agent._dispatch("edit_file", {"path": "a.py", "old_string": "x = 1", "new_string": "x = 0",
                                                 "replace_all": True})
        self.assertEqual((self.root / "a.py").read_text().count("x = 0"), 2)

    def test_read_file_numbers_and_pages(self):
        out = self.agent._dispatch("read_file", {"path": "a.py", "start": 2, "end": 3})
        self.assertEqual(out.splitlines()[0].split("\t"), ["    2", "x = 1"])
        self.assertEqual(len(out.splitlines()), 2)

    def test_run_strips_secrets_and_uses_root(self):
        os.environ["SOME_API_KEY"] = "sk-test"
        self.addCleanup(os.environ.pop, "SOME_API_KEY")
        out = self.agent._dispatch("run", {"command": "pwd; echo key=${SOME_API_KEY:-none}"})
        self.assertIn(os.path.realpath(self.root), out)
        self.assertIn("key=none", out)

    def test_run_timeout_kills_process(self):
        out = self.agent._dispatch("run", {"command": "sleep 5", "timeout": 1})
        self.assertIn("timeout after 1s", out)

    def test_backgrounded_process_does_not_block_run(self):
        import time
        t = time.monotonic()
        out = self.agent._dispatch("run", {"command": "sleep 30 & echo started", "timeout": 5})
        self.assertLess(time.monotonic() - t, 3, "run must return when bash exits, not when the pipe closes")
        self.assertIn("started", out)
        self.assertTrue(self.agent._pgids, "background process group is tracked")
        pgid = next(iter(self.agent._pgids))
        self.agent._kill_process_groups()
        with self.assertRaises(ProcessLookupError):
            os.killpg(pgid, 0)

    def test_background_server_survives_between_commands_and_dies_with_the_run(self):
        out = self.agent._dispatch("run", {"command": "python3 -m http.server 8997 >/dev/null 2>&1 & sleep 1; echo up"})
        self.assertIn("up", out)
        out = self.agent._dispatch("run", {"command": "curl -s -o /dev/null -w '%{http_code}' localhost:8997/"})
        self.assertIn("200", out)
        self.agent._kill_process_groups()
        out = self.agent._dispatch("run", {"command": "curl -s -o /dev/null -w '%{http_code}' localhost:8997/ || echo down"})
        self.assertIn("down", out)

    def test_read_only_agent_cannot_write(self):
        ro = Agent(role="inspect", model="m", system=SYSTEM, root=str(self.root), tools=READ_TOOLS, client=scripted())
        self.assertIn("not available", ro._dispatch("write_file", {"path": "b.py", "content": "x"}))
        self.assertIn("not available", ro._dispatch("run", {"command": "touch b.py"}))
        self.assertFalse((self.root / "b.py").exists())

    def test_writes_into_git_dir_refused(self):
        (self.root / ".git").mkdir()
        self.assertIn("not allowed", self.agent._dispatch("write_file", {"path": ".git/config", "content": "x"}))


class LoopTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def run_agent(self, *turns, schema=None, **kw):
        rec = Recorder()
        agent = Agent(role="build", model="m", system=SYSTEM, root=self.root, tools=READ_TOOLS + WRITE_TOOLS,
                      result_schema=schema, recorder=rec, client=scripted(*turns), **kw)
        return agent.run("task"), rec

    def test_tool_then_finish_with_structured_result(self):
        res, rec = self.run_agent(
            reply(tool_call("write_file", {"path": "f.txt", "content": "hi"})),
            finish("wrote it", {"verdict": "APPROVED"}),
            schema={"type": "object"})
        self.assertTrue(res.ok)
        self.assertEqual(res.result, {"verdict": "APPROVED"})
        self.assertEqual(res.steps, 2)
        self.assertAlmostEqual(res.cost_usd, 0.002)
        self.assertEqual([s[0] for s in rec.steps], ["model", "tool", "model", "finish"])
        self.assertTrue((Path(self.root) / "f.txt").exists())

    def test_plain_text_is_nudged_then_accepted(self):
        res, _ = self.run_agent(reply(text="thinking..."), reply(text='All done. {"verdict": "APPROVED"}'),
                                schema={"type": "object"})
        self.assertTrue(res.ok)
        self.assertEqual(res.result, {"verdict": "APPROVED"})

    def test_empty_reply_is_reasked_not_fatal(self):
        res, rec = self.run_agent(reply(), reply(), finish("ok", {"v": 1}), schema={"type": "object"})
        self.assertTrue(res.ok)
        self.assertIn("empty reply", rec.steps[0][2])

    def test_bad_json_arguments_are_reported_back(self):
        bad = reply({"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{nope"}})
        res, _ = self.run_agent(bad, finish("ok"))
        self.assertTrue(res.ok)

    def test_step_limit(self):
        loop = [reply(tool_call("list_dir", {}))] * 10
        res, _ = self.run_agent(*loop, max_steps=3)
        self.assertEqual(res.status, "max_steps")

    def test_budget_limit(self):
        loop = [reply(tool_call("list_dir", {}), cost=0.5)] * 10
        res, _ = self.run_agent(*loop, budget_usd=1.0)
        self.assertEqual(res.status, "budget")
        self.assertLessEqual(res.steps, 2)

    def test_missing_result_is_an_error(self):
        res, _ = self.run_agent(finish("no result"), schema={"type": "object"})
        self.assertEqual(res.status, "error")


class ListJournal:
    def __init__(self, entries=None):
        self.entries = list(entries or [])

    def append(self, kind, data):
        self.entries.append({"kind": kind, "data": json.loads(json.dumps(data))})

    def kinds(self):
        return [e["kind"] for e in self.entries]


class DurableLoopTest(unittest.TestCase):
    """A journaled run, cut off at some point, continues in a fresh Agent."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.seen = []  # messages the model was shown, per call

    def tearDown(self):
        self.tmp.cleanup()

    def agent(self, *turns, journal=None, **kw):
        seq = list(turns)

        def builder(messages, turn):
            self.seen.append(list(messages))
            return seq[turn] if turn < len(seq) else finish("done")
        return Agent(role="build", model="m", system=SYSTEM, root=str(self.root),
                     tools=READ_TOOLS + WRITE_TOOLS + RUN_TOOLS, recorder=Recorder(),
                     client=FakeClient({"builder": builder}), journal=journal or ListJournal(), **kw)

    def interrupted_after_call(self, call, spawn=None):
        """The journal of a run whose process died while `call` was executing."""
        j = ListJournal()
        j.append("start", {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "task"}]})
        msg = reply(call).message
        j.append("model", {"message": msg, "usage": {"tokens_in": 100, "cost_usd": 0.25}, "elapsed": 5.0})
        j.append("call", {"key": call["id"], "name": call["function"]["name"],
                          "args": json.loads(call["function"]["arguments"])})
        if spawn:
            j.append("spawn", {"key": call["id"], **spawn})
        return j

    def spawn(self, command):
        """Start `command` the way the run tool does; return its spawn entry."""
        fd, out = tempfile.mkstemp(prefix="yamanote-test-", suffix=".log")
        proc = subprocess.Popen(["bash", "-c", command], cwd=self.root, stdout=fd, stderr=subprocess.STDOUT,
                                start_new_session=True)
        os.close(fd)
        self.addCleanup(proc.wait)
        self.addCleanup(lambda: _kill_group(proc.pid))
        self.proc = proc
        return {"pgid": proc.pid, "output": out, "boot": _boot_id()}

    def test_every_turn_is_journaled_and_each_call_before_it_runs(self):
        j = ListJournal()
        a = self.agent(reply(tool_call("write_file", {"path": "f.txt", "content": "hi"})), finish("ok"), journal=j)
        dispatch = a._dispatch
        a._dispatch = lambda name, args: (self.assertEqual(j.kinds()[-1], "call"), dispatch(name, args))[1]
        self.assertTrue(a.run("task").ok)
        self.assertEqual(j.kinds(), ["start", "model", "call", "result", "model", "result", "end"])
        self.assertEqual(j.entries[2]["data"], {"key": "call_write_file_0", "name": "write_file",
                                               "args": {"path": "f.txt", "content": "hi"}})

    def test_unsafe_interrupted_call_is_reported_not_rerun(self):
        call = tool_call("run", {"command": "echo x >> ran.txt"})
        j = self.interrupted_after_call(call, {"pgid": 0, "output": None, "boot": ""})
        res = self.agent(None, finish("checked"), journal=j).resume(j.entries)
        self.assertTrue(res.ok)
        self.assertFalse((self.root / "ran.txt").exists(), "run is not replay-safe")
        report = self.seen[-1][-1]
        self.assertEqual((report["role"], report["tool_call_id"]), ("tool", call["id"]))
        self.assertIn("INTERRUPTED", report["content"])
        self.assertIn("Investigate", report["content"])
        self.assertEqual(j.entries[4]["data"], {"interrupted": [call["id"]], "replayed": []})

    def test_safe_interrupted_call_is_replayed(self):
        (self.root / "a.txt").write_text("hello\n")
        j = self.interrupted_after_call(tool_call("read_file", {"path": "a.txt"}))
        a = self.agent(None, finish("read it"), journal=j)
        self.assertTrue(a.resume(j.entries).ok)
        self.assertIn("hello", self.seen[-1][-1]["content"])
        self.assertEqual(a.resumed, {"interrupted": 0, "replayed": 1})

    def test_calls_that_never_started_run_normally(self):
        first, second = (tool_call("write_file", {"path": "a.txt", "content": "1"}, 0),
                         tool_call("run", {"command": "echo ran > b.txt"}, 1))
        j = ListJournal()
        j.append("start", {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "task"}]})
        j.append("model", {"message": reply(first, second).message, "usage": {}, "elapsed": 1.0})
        j.append("call", {"key": first["id"], "name": "write_file", "args": {"path": "a.txt", "content": "1"}})
        j.append("result", {"key": first["id"], "content": "Wrote a.txt"})
        self.assertTrue(self.agent(None, finish("ok"), journal=j).resume(j.entries).ok)
        self.assertFalse((self.root / "a.txt").exists(), "a call with a result is not run again")
        self.assertEqual((self.root / "b.txt").read_text(), "ran\n")
        self.assertEqual([m.get("tool_call_id") for m in self.seen[-1][-2:]], [first["id"], second["id"]])

    def test_a_journaled_finish_completes_without_another_model_call(self):
        j = ListJournal()
        self.agent(finish("all done", {"verdict": "APPROVED"}), journal=j, result_schema={"type": "object"}).run("t")
        open_run = [e for e in j.entries if e["kind"] != "end"]  # died before the factory saw the result
        self.seen.clear()
        res = self.agent(journal=ListJournal(open_run), result_schema={"type": "object"}).resume(open_run)
        self.assertEqual((res.status, res.result), ("ok", {"verdict": "APPROVED"}))
        self.assertEqual(self.seen, [])

    def test_resumed_run_keeps_counting_against_its_limits(self):
        j = self.interrupted_after_call(tool_call("list_dir", {}))
        res = self.agent(None, reply(tool_call("list_dir", {})), journal=j, max_steps=2).resume(j.entries)
        self.assertEqual(res.status, "max_steps")
        self.assertAlmostEqual(res.cost_usd, 0.251)

    def test_resume_note_is_shown_to_the_agent(self):
        j = self.interrupted_after_call(tool_call("list_dir", {}))
        self.agent(None, finish("ok"), journal=j).resume(j.entries, note="worktree was reset")
        self.assertEqual(self.seen[-1][-1], {"role": "user", "content": "worktree was reset"})

    def test_cancelled_run_leaves_the_journal_open_and_the_call_unanswered(self):
        j = ListJournal()
        a = self.agent(reply(tool_call("run", {"command": "sleep 30"})), journal=j)
        threading.Timer(0.5, a.stop).start()
        self.assertEqual(a.run("task").status, "cancelled")
        self.assertEqual(j.kinds(), ["start", "model", "call", "spawn"])
        self.assertTrue(os.path.exists(j.entries[3]["data"]["output"]), "output kept for the resume")
        os.unlink(j.entries[3]["data"]["output"])

    def test_run_that_never_spawned_a_process_runs_normally(self):
        j = self.interrupted_after_call(tool_call("run", {"command": "echo x >> ran.txt"}))
        a = self.agent(None, finish("ok"), journal=j)
        self.assertTrue(a.resume(j.entries).ok)
        self.assertEqual((self.root / "ran.txt").read_text(), "x\n")
        self.assertEqual(a.resumed, {"interrupted": 0, "replayed": 0})

    def test_command_still_running_is_stopped_and_its_last_output_shown(self):
        spawn = self.spawn("echo early-line; echo partial-output; sleep 30")
        time.sleep(0.3)
        j = self.interrupted_after_call(tool_call("run", {"command": "make test"}), spawn)
        self.agent(None, finish("ok"), journal=j).resume(j.entries)
        report = self.seen[-1][-1]["content"] if self.seen[-1][-1]["role"] == "tool" else self.seen[-1][-2]["content"]
        self.assertIn("still running when you resumed; it has been stopped", report)
        self.assertIn("partial-output", report)
        self.assertIsNotNone(self.proc.wait(timeout=2), "the leftover command was killed")
        self.assertFalse(os.path.exists(spawn["output"]), "output file removed once read")

    def test_finished_command_output_is_tailed(self):
        spawn = self.spawn("seq 1 5000")
        time.sleep(0.5)
        j = self.interrupted_after_call(tool_call("run", {"command": "seq 1 5000"}), spawn)
        a = self.agent(None, finish("ok"), journal=j)
        a.resume(j.entries)
        report = a._evidence[j.entries[1]["data"]["message"]["tool_calls"][0]["id"]]
        self.assertIn("no longer running", report)
        self.assertIn("\n5000\n", report)
        self.assertNotIn("\n1\n", report, "only the tail of the output")
        self.assertLess(len(report), 1700)

    def test_interrupted_edit_reports_whether_it_landed(self):
        (self.root / "a.py").write_text("x = 2\n")
        j = self.interrupted_after_call(tool_call("edit_file", {"path": "a.py", "old_string": "x = 1",
                                                                "new_string": "x = 2"}))
        self.agent(None, finish("ok"), journal=j).resume(j.entries)
        report = self.seen[-1][-1]["content"]
        self.assertIn("old_string 0 time(s) and new_string 1 time(s): the edit looks applied", report)

    def test_resume_note_carries_downtime_and_worktree_snapshot(self):
        sh("git init -q -b main && git -c user.email=t@t -c user.name=t commit -q --allow-empty -m first",
           self.root)
        (self.root / "new.py").write_text("x\n")
        j = self.interrupted_after_call(tool_call("list_dir", {}))
        for e in j.entries:
            e["ts"] = time.time() - 600
        self.agent(None, finish("ok"), journal=j).resume(j.entries, note="restarted")
        note = self.seen[-1][-1]["content"]
        self.assertTrue(note.startswith("restarted\n\nYou were stopped 10 min ago."), note)
        self.assertIn("?? new.py", note)
        self.assertIn("Last commit:", note)
        self.assertIn("first", note)


class HelperTest(unittest.TestCase):
    def test_extract_json(self):
        self.assertEqual(_extract_json('x ```json\n{"a": 1}\n``` y'), {"a": 1})
        self.assertEqual(_extract_json('verdict {"a": {"b": 2}} end'), {"a": {"b": 2}})
        self.assertIsNone(_extract_json("no json"))

    def test_trim_history_keeps_recent_tool_output(self):
        msgs = [{"role": "system", "content": "s"}] + [
            {"role": "tool", "content": "x" * 50_000} for _ in range(12)]
        _trim_history(msgs)
        self.assertIn("trimmed", msgs[1]["content"])
        self.assertEqual(len(msgs[-1]["content"]), 50_000)

    def test_cache_breakpoints_only_for_anthropic(self):
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a"}, {"role": "tool", "content": "t", "tool_call_id": "1"}]
        self.assertIs(_with_cache_breakpoints("deepseek/x", msgs), msgs)
        out = _with_cache_breakpoints("anthropic/claude-sonnet-5.5", msgs)
        self.assertEqual(out[0]["content"][0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(out[3]["content"][0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(out[1]["content"], "u")
        self.assertEqual(msgs[0]["content"], "sys", "input not mutated")

    def test_parse_usage_cost(self):
        c = _parse({"model": "deepseek/x", "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0123,
                              "prompt_tokens_details": {"cached_tokens": 4}}}, "req", 0.1)
        self.assertEqual((c.model, c.tokens_in, c.tokens_out, c.cached_tokens, c.cost_usd),
                         ("deepseek/x", 10, 5, 4, 0.0123))


if __name__ == "__main__":
    unittest.main()
