"""Offline tests for tally.checked: the real Terminus-2 loop with a scripted model,
terminal and container. No Docker, no API key.

  .venv-harbor/Scripts/python tests/test_checked.py      (Harbor's Python, 3.12+)
"""
import asyncio
import base64
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harbor.llms.base import LLMResponse, OutputLengthExceededError
from harbor.models.agent.context import AgentContext
from harbor.models.metric.usage_info import UsageInfo

from tally.checked import CheckedTerminus

WORKS = "CHECK: output file exists\n```bash\ntest -f /app/out.txt || { echo missing; exit 1; }\n```\n"
UNSAFE = "CHECK: cleans up\n```bash\nrm -rf /app/build\n```\n"
BROKEN = "CHECK: does not parse\n```bash\nif then fi\n```\n"
NO_TOOL = "CHECK: uses a tool nobody mentioned\n```bash\nfrobnicate /app/out.txt\n```\n"
SAYS_FAIL = "CHECK: says FAIL but exits 0\n```bash\npython3 - <<'PY'\nprint('FAIL: crashed')\nPY\nexit 0\n```\n"


def turn(claim, keys=None):
    return json.dumps({"analysis": "a", "plan": "p", "task_complete": claim,
                       "commands": [{"keystrokes": k, "duration": 0.1} for k in (keys or [])]})


def in_reasoning(answer):
    """What Token Factory returned for a quarter of Nano's turns: nothing in content, a draft
    and then the real answer in reasoning_content."""
    return ("", "Maybe {\"analysis\": \"draft\", \"plan\": \"x\", \"commands\": [{\"keystrokes\": \"ls\\n\"}]} "
                "no, better:\n" + answer)


class Model:
    """Plays the agent from a list of turns and the check writer from a fixed reply.
    A turn or reply is content, or (content, reasoning_content)."""

    def __init__(self, turns, writer_reply):
        self.turns, self.writer_reply = list(turns), writer_reply
        self.prompts, self.writer_calls, self.writer_kwargs = [], 0, []

    async def call(self, prompt, message_history=(), **kwargs):
        usage = UsageInfo(prompt_tokens=10, completion_tokens=5, cache_tokens=0, cost_usd=0.0)
        if prompt.startswith("You write acceptance checks"):
            self.writer_calls += 1
            self.writer_kwargs.append(kwargs)
            assert "<task>\nmake /app/out.txt\n</task>" in prompt
            reply = self.writer_reply.pop(0) if isinstance(self.writer_reply, list) else self.writer_reply
            if isinstance(reply, Exception):
                raise reply
        else:
            self.prompts.append(prompt)
            reply = self.turns.pop(0)
        content, reasoning = reply if isinstance(reply, tuple) else (reply, None)
        return LLMResponse(content=content, reasoning_content=reasoning, usage=usage)


class Container:
    def __init__(self):
        self.files = set()

    async def exec(self, command, timeout_sec=None, **kwargs):
        m = re.search(r"printf %s (\S+) \| base64 -d", command)
        if not m:                                        # the directory listing
            return Result(0, "total 0\n")
        script = base64.b64decode(m.group(1)).decode()
        if script.startswith("if then"):
            return Result(99, "syntax error near unexpected token `then'")
        if script.startswith("frobnicate"):
            return Result(127, "/tmp/.tally_check.sh: line 1: frobnicate: command not found")
        if "print('FAIL: crashed')" in script:
            return Result(0, "FAIL: crashed\n")
        if "test -f /app/out.txt" in script:
            return Result(0, "") if "/app/out.txt" in self.files else Result(1, "missing\n")
        raise AssertionError("unsafe or unexpected script ran: %r" % script)


class Result:
    def __init__(self, code, out):
        self.return_code, self.stdout, self.stderr = code, out, ""


class Terminal:
    def __init__(self, container):
        self.container = container

    async def is_session_alive(self):
        return True

    async def send_keys(self, keys, block=False, min_timeout_sec=0.0):
        if keys.startswith("touch "):
            self.container.files.add(keys.split()[1])

    async def get_incremental_output(self):
        return "$ "


def run(turns, writer_reply, **options):
    container = Container()
    agent = CheckedTerminus(logs_dir=Path(tempfile.mkdtemp()), model_name="openai/scripted", max_turns=10,
                            enable_summarize=False, record_terminal_session=False,
                            suppress_max_turns_warning=True, **options)
    model = Model(turns, writer_reply)
    agent._llm, agent._session = model, Terminal(container)
    context = AgentContext()
    asyncio.run(agent.run("make /app/out.txt", container, context))
    log = context.metadata["verification"]
    return model, log, [r["decision"] for r in log["rounds"]], container, agent


def test_rejects_a_false_claim_then_accepts_the_fix():
    model, log, decisions, _, _ = run([turn(True), turn(False, ["touch /app/out.txt\n"]), turn(True)],
                                      WORKS + UNSAFE + BROKEN + NO_TOOL)
    assert decisions == ["rejected", "accepted"], decisions
    assert model.writer_calls == 1                       # checks are written once per trial
    assert "FAILED: output file exists" in model.prompts[1] and "missing" in model.prompts[1]
    assert "does not parse" not in model.prompts[1]      # broken checks are not held against the agent
    assert [c["requirement"] for c in log["writer"]["unsafe"]] == ["cleans up"]
    statuses = [r["status"] for r in log["rounds"][0]["results"]]
    assert statuses == ["fail", "broken", "broken"], statuses
    assert not model.turns


def test_accepts_once_rejections_run_out():
    model, log, decisions, _, _ = run([turn(True)] * 3, WORKS, max_rejections=2)
    assert decisions == ["rejected", "rejected", "accepted_out_of_rejections"], decisions
    assert "rejection 2 of 2" in model.prompts[2]


def test_zero_rejections_only_measures():
    model, log, decisions, _, _ = run([turn(True)], WORKS, max_rejections=0)
    assert decisions == ["accepted_out_of_rejections"], decisions
    assert log["rounds"][0]["results"][0]["status"] == "fail"


def test_commands_in_the_claiming_turn_run_before_the_checks():
    model, log, decisions, _, _ = run([turn(True, ["touch /app/out.txt\n"])], WORKS)
    assert decisions == ["accepted"], decisions
    assert len(model.prompts) == 1


def test_no_usable_checks_falls_back_to_confirm_twice():
    model, log, decisions, _, _ = run([turn(True), turn(True)], "I could not think of any checks.")
    assert decisions == ["stock", "stock"], decisions
    assert "Are you sure you want to mark the task as complete?" in model.prompts[1]


def test_a_printed_fail_counts_even_when_the_script_exits_0():
    model, log, decisions, _, _ = run([turn(True), turn(True)], SAYS_FAIL, max_rejections=1)
    assert decisions == ["rejected", "accepted_out_of_rejections"], decisions
    assert "FAIL: crashed" in model.prompts[1]


def test_recovers_the_answer_filed_as_reasoning():
    model, log, decisions, container, agent = run(
        [in_reasoning(turn(False, ["touch /app/out.txt\n"])), turn(True)], WORKS)
    assert log["recovered_turns"] == 1
    assert container.files == {"/app/out.txt"}          # the last answer ran, not the draft
    assert not model.prompts[1].startswith("Previous response had parsing errors")
    assert agent._chat.messages[1]["content"] == turn(False, ["touch /app/out.txt\n"])
    assert decisions == ["accepted"], decisions


def test_recovery_can_be_switched_off():
    model, log, decisions, container, _ = run(
        [in_reasoning(turn(False, ["touch /app/out.txt\n"])), turn(False, ["touch /app/out.txt\n"]), turn(True)],
        WORKS, recover_answers=False)
    assert log["recovered_turns"] == 0
    assert model.prompts[1].startswith("Previous response had parsing errors")
    assert decisions == ["accepted"], decisions


def test_recovers_checks_filed_as_reasoning():
    model, log, decisions, _, _ = run([turn(True), turn(True)], ("", "thinking...\n" + WORKS), max_rejections=1)
    assert log["writer"]["from_reasoning"] is True
    assert decisions == ["rejected", "accepted_out_of_rejections"], decisions


def too_long():
    return OutputLengthExceededError("hit max_tokens", truncated_response="")


def test_writer_samples_again_after_running_out_of_tokens():
    model, log, decisions, _, _ = run([turn(True), turn(False, ["touch /app/out.txt\n"]), turn(True)],
                                      [too_long(), WORKS])
    assert log["writer"]["length_errors"] == 1 and log["writer"]["thinking"] is True
    assert decisions == ["rejected", "accepted"], decisions


def test_writer_switches_thinking_off_last():
    model, log, decisions, _, _ = run([turn(True), turn(False, ["touch /app/out.txt\n"]), turn(True)],
                                      [too_long(), too_long(), WORKS])
    assert log["writer"]["length_errors"] == 2 and log["writer"]["thinking"] is False
    assert model.writer_kwargs[2]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert "extra_body" not in model.writer_kwargs[0]
    assert decisions == ["rejected", "accepted"], decisions


def test_writer_gives_up_after_three_tries():
    model, log, decisions, _, _ = run([turn(True), turn(True)], [too_long(), too_long(), too_long()])
    assert model.writer_calls == 3 and log["writer"]["length_errors"] == 3
    assert decisions == ["stock", "stock"], decisions


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print("%d passed" % len(tests))
