"""Terminus-2 that reads the model's whole answer and has to pass checks before it may stop.

Two things went wrong in 237 stock Terminus-2 trials of Nemotron 3 Nano on Token Factory.

A quarter of the turns were thrown away. On the dev tasks, 366 turns came back with empty
content, and in 359 of them the complete JSON answer was sitting in reasoning_content: the
served reasoning parser had filed the answer as thinking, and Terminus-2 reads only
content. When content does not parse, this agent takes the last complete answer from the
reasoning instead (recover_answers, on by default). It is the model's own output.

Claims of completion were mostly false: 223 trials ended with one, and 202 of those failed
the hidden tests. The stock agent answers a first "task_complete" with "are you sure?" and
takes the second one at its word. This agent answers it with evidence:

1. On the first claim, a separate model call writes up to six shell checks, one per
   observable requirement. It sees the task instruction and a listing of the working
   directory -- not the agent's work or reasoning, not the hidden tests (Harbor
   uploads those after the agent stops), not anyone's solution. Written once per trial.
2. The checks run in the task container. If every usable check passes, the claim stands.
3. If one fails, the agent gets the failures with their output and keeps working.
   After max_rejections rejected claims, the next claim stands regardless.

Checks that cannot run (a bash syntax error, a tool the task never mentions) are left
out of the verdict, and checks that would change the container are never run.

  harbor run ... -a tally.checked:CheckedTerminus --ak max_rejections=2

max_rejections=0 still writes and runs the checks but never rejects: it measures how often
the checks agree with the hidden tests without changing what the agent does.

Imported by Harbor's own Python, so this module needs nothing beyond Harbor.
"""
import base64
import json
import re
import time

from pydantic import Field

from harbor.agents.terminus_2.terminus_2 import Terminus2, Terminus2Options
from harbor.llms.base import OutputLengthExceededError

MAX_CHECKS = 6
CHECK_TIMEOUT = 60
CHECK_BLOCK = re.compile(r"CHECK:\s*(.+?)\s*\n\s*```[a-z]*\n(.*?)```", re.S)
UNSAFE = re.compile(
    r"\b(rm|rmdir|mv|shred|truncate|mkfs|kill|pkill|killall|reboot|shutdown|apt|apt-get|apk|yum|dnf)\b"
    r"|\b(pip3?|uv|npm|conda)\s+(install|uninstall)\b"
    r"|\bsed\s+-i|\bgit\s+(checkout|reset|clean|stash|commit|push)\b"
    r"|>\s*/(?!tmp\b|dev/null\b)"
)

WRITER = """You write acceptance checks for a command-line task. An agent claims it has finished the task below inside a Linux container. You cannot see its work. Your checks will run in that container to decide whether the claim holds.

Task given to the agent:
<task>
{instruction}
</task>

Working directory listing:
<listing>
{listing}
</listing>

Write the checks a strict grader would run:
- One check per observable requirement: an output file and its exact format, a value it must contain, a program that must run, a command whose output must match, a service that must answer.
- Test substance, not just existence. If the task provides example code, an example input and output, a test or evaluation script, or a tool to measure with, run it and check the result against the task's stated criteria.
- Each check is a bash script that prints what it found, then prints PASS or FAIL as its last line, and exits 1 on FAIL.
- Checks must not change anything: write only under /tmp, install nothing, delete nothing, stop no processes.
- Each check must finish within {timeout} seconds.
- At most {max_checks} checks, most important first.

Answer in exactly this format and nothing else:
CHECK: <the requirement, in a few words>
```bash
<script>
```"""


class CheckedOptions(Terminus2Options):
    max_rejections: int = Field(
        default=2,
        description="Completion claims to reject on failed checks before a claim is accepted regardless.",
    )
    recover_answers: bool = Field(
        default=True,
        description="When content does not parse, use the last complete answer found in reasoning_content.",
    )


class CheckedTerminus(Terminus2):
    options_model = CheckedOptions
    options: CheckedOptions

    def version(self) -> str | None:
        return "2.0.0-checked.3"          # .3: no outline recovery; the original writer prompt, which graded best

    async def run(self, instruction, environment, context) -> None:
        self._instruction, self._env = instruction, environment
        self._checks, self._report, self._claimed, self._rejections = None, None, False, 0
        self._log = {"recovered_turns": 0, "writer": None, "rounds": []}
        try:
            await super().run(instruction, environment, context)
        finally:
            context.metadata = dict(context.metadata or {}, verification=self._log)

    async def _query_llm(self, chat, prompt, original_instruction="", session=None):
        response = await super()._query_llm(chat, prompt, original_instruction, session)
        if (self.options.recover_answers and response.reasoning_content
                and self._parser.parse_response(response.content or "").error):
            found = self._answer_in(response.reasoning_content)
            if found:
                response.content = found
                if chat.messages and chat.messages[-1].get("role") == "assistant":
                    chat.messages[-1]["content"] = found     # next turn the model sees its answer, not a blank
                self._log["recovered_turns"] += 1
        return response

    def _answer_in(self, text: str) -> str | None:
        """The last JSON object in text that the stock parser accepts, re-serialised.
        Reasoning can hold drafts before the answer, so search from the end."""
        decoder = json.JSONDecoder()
        for i in range(len(text) - 1, -1, -1):
            if text[i] != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(text, i)
            except ValueError:
                continue
            if isinstance(obj, dict) and ("commands" in obj or "task_complete" in obj):
                candidate = json.dumps(obj)
                if self._parser.parse_response(candidate).error:
                    continue
                if any("..." in c.get("keystrokes", "") or "…" in c.get("keystrokes", "")
                       for c in obj.get("commands") or [] if isinstance(c, dict)):
                    # An outline, not an answer: on prove-plus-comm a recovered
                    # `cat > plus_comm.v <<'EOF'...EOF` left the shell inside a heredoc
                    # for the rest of the trial. 9 of 349 recoverable dev turns look like this.
                    return None
                return candidate
        return None

    async def _handle_llm_interaction(self, *args, **kwargs):
        result = await super()._handle_llm_interaction(*args, **kwargs)
        self._claimed = result[1]
        return result

    async def _execute_commands(self, commands, session):
        # A claim is judged after the claiming turn's own commands have run.
        timeout_occurred, output = await super()._execute_commands(commands, session)
        if self._claimed:
            self._claimed = False
            await self._judge_claim()
        return timeout_occurred, output

    def _get_completion_confirmation_message(self, terminal_output: str) -> str:
        if self._report is None:
            return super()._get_completion_confirmation_message(terminal_output)
        return "Current terminal state:\n%s\n\n%s" % (terminal_output, self._report)

    async def _judge_claim(self) -> None:
        """The stock loop reads _pending_completion right after this: True ends the run,
        False sends _get_completion_confirmation_message, which then carries the report."""
        self._report = None
        if self._checks is None:
            self._checks = await self._write_checks()
        results = [await self._run_check(c) for c in self._checks]
        usable = [r for r in results if r["status"] in ("pass", "fail")]
        failed = [r for r in usable if r["status"] == "fail"]
        entry = {"turn": self._n_episodes, "results": results}
        self._log["rounds"].append(entry)
        if not usable:
            entry["decision"] = "stock"                  # nothing to judge by: confirm-twice as before
        elif not failed:
            entry["decision"] = "accepted"
            self._pending_completion = True
        elif self._rejections >= self.options.max_rejections:
            entry["decision"] = "accepted_out_of_rejections"
            self._pending_completion = True
        else:
            self._rejections += 1
            entry["decision"] = "rejected"
            self._pending_completion = False
            self._report = self._format_report(usable)

    async def _write_checks(self) -> list[dict]:
        entry = {}
        self._log["writer"] = entry
        try:
            listing = await self._env.exec("ls -la 2>&1 | head -50", timeout_sec=20)
            start = time.time()
            checks, response = await write_checks(self._llm, WRITER, self._instruction, listing.stdout or "",
                                                   self._llm_call_kwargs, entry, self.options.recover_answers)
        except Exception as e:
            entry["error"] = repr(e)[:500]
            return []
        if response is not None:
            self._track_api_request_time(start)
            self._update_subagent_metrics(response.usage)
        return checks

    async def _run_check(self, check: dict) -> dict:
        return await run_check(self._env, check, self._instruction)

    def _format_report(self, usable: list[dict]) -> str:
        failed = [r for r in usable if r["status"] == "fail"]
        parts = ["You set task_complete, but %d of %d checks written independently from the task "
                 "description failed in your environment. The grader's hidden tests will likely fail "
                 "for the same reasons, so the task is not complete yet." % (len(failed), len(usable))]
        for r in failed:
            parts.append("FAILED: %s\n```bash\n%s\n```\nexit %s, output:\n%s"
                         % (r["requirement"], r["script"][:400], r["exit"], r["output"].strip() or "(none)"))
        passed = [r["requirement"] for r in usable if r["status"] == "pass"]
        if passed:
            parts.append("PASSED: " + "; ".join(passed))
        parts.append("Fix what failed, run the check yourself to confirm, then set task_complete again. "
                     "A check can be wrong: if one misreads the task, verify that requirement another way "
                     "instead of bending a correct solution to fit it. This was rejection %d of %d."
                     % (self._rejections, self.options.max_rejections))
        return "\n\n".join(parts)


async def write_checks(llm, template, instruction, listing, call_kwargs, entry, recover=True):
    """Ask llm for checks; log into entry. -> (checks, response, or None when every sample ran out).
    Nano sometimes thinks until max_tokens (1 in 3 calls on one writer prompt), which Harbor raises
    as OutputLengthExceededError: sample again, and last with thinking switched off."""
    entry["length_errors"] = 0
    prompt = template.format(instruction=instruction, listing=listing[-2000:],
                             timeout=CHECK_TIMEOUT, max_checks=MAX_CHECKS)
    response = None
    for extra in ({}, {}, {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}):
        try:
            response = await llm.call(prompt=prompt, message_history=[], **dict(call_kwargs, **extra))
            entry["thinking"] = not extra
            break
        except OutputLengthExceededError:
            entry["length_errors"] += 1
    if response is None:
        return [], None
    if response.usage:
        entry["tokens"] = [response.usage.prompt_tokens, response.usage.completion_tokens]
    text = response.content or ""
    blocks = CHECK_BLOCK.findall(text)[:MAX_CHECKS]
    if not blocks and recover and response.reasoning_content:
        text = response.reasoning_content                # the same misfiling as the agent's turns
        blocks = CHECK_BLOCK.findall(text)[-MAX_CHECKS:]
        entry["from_reasoning"] = True
    entry["response"] = text[-6000:]
    checks = []
    for requirement, script in blocks:
        requirement, script = requirement.strip(), script.strip()
        if UNSAFE.search(script):
            entry.setdefault("unsafe", []).append({"requirement": requirement, "script": script})
            continue
        checks.append({"requirement": requirement, "script": script})
    return checks, response


async def run_check(env, check: dict, instruction: str) -> dict:
    b64 = base64.b64encode(check["script"].encode("utf-8")).decode("ascii")
    cmd = ("printf %s {b} | base64 -d > /tmp/.tally_check.sh && bash -n /tmp/.tally_check.sh 2>&1 || exit 99; "
           "timeout {t} bash /tmp/.tally_check.sh 2>&1; code=$?; rm -f /tmp/.tally_check.sh; exit $code"
           ).format(b=b64, t=CHECK_TIMEOUT)
    result = dict(check)
    try:
        r = await env.exec(cmd, timeout_sec=CHECK_TIMEOUT + 30)
    except Exception as e:
        return dict(result, status="error", output=repr(e)[:300])
    full = (r.stdout or "") + (r.stderr or "")
    out = full[-600:]
    # A script can print FAIL and still exit 0: a heredoc that fails, followed by "exit 0", does.
    status = "pass" if r.return_code == 0 and not re.search(r"^\s*FAIL", full, re.M) else "fail"
    if r.return_code == 99:
        status = "broken"                                # the script itself does not parse
    elif r.return_code == 127:
        missing = re.findall(r"([\w.+-]+): (?:command )?not found", full)
        if missing and missing[-1] not in instruction:
            status = "broken"                            # the check needs a tool the task never mentions
    return dict(result, status=status, exit=r.return_code, output=out)
