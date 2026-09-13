"""Grade check writers against each task's reference solution. No agent model is involved.

In one container per task:
1. every writer writes checks from the task text and the starting directory listing
2. the checks run on the untouched starting state   -> a check worth having fails here
3. the task's reference solution runs, the way Harbor's oracle agent runs it
4. the checks run again                               -> a correct check passes here
Harbor's verifier then runs the hidden tests on the solved state; a task whose reference
solution fails them is left out of the grade. Dev tasks only: the reference solutions tune
the layer, and held-out stays unseen.

  python -m tally.run --plan data/plan_terminalbench.json --split dev --agent tally.checkeval:CheckEval \
      --ak writers=nano-v1,nano,super,ultra --agent-timeout-multiplier 4
  python -m tally.boost checkeval JOB_DIR

Imported by Harbor's own Python, like tally.checked.
"""
import asyncio

from pydantic import Field

from harbor.agents.base import BaseAgent
from harbor.agents.oracle import OracleAgent
from harbor.agents.terminus_2.terminus_2 import Terminus2Options
from harbor.llms.lite_llm import LiteLLM

from tally.checked import WRITER, run_check, write_checks

# The prompt the dev check ran with (commit dc7f5c9): today's minus the three lines added after it.
_ADDED = [
    "- Test it the way the task says it will be used: if the task says the result will be installed, "
    "imported, called, served or run in a certain way, do exactly that.\n",
    "- At least one check must fail for a solution that only looks right: one that hard-codes an answer, "
    "ignores its input, or prints the right format with the wrong content. For example, give it a second "
    "input you construct yourself and verify the result independently.\n",
    " If running the solution would modify files the task gave, copy them to /tmp first and run it there.",
]
WRITER_V1 = WRITER
for _line in _ADDED:
    assert _line in WRITER_V1, _line
    WRITER_V1 = WRITER_V1.replace(_line, "")

WRITERS = {
    "nano-v1": ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", WRITER_V1),
    "nano": ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", WRITER),
    "super": ("nvidia/nemotron-3-super-120b-a12b", WRITER),
    "ultra": ("nvidia/Nemotron-3-Ultra-550b-a55b", WRITER),
}


class CheckEvalOptions(Terminus2Options):
    """Terminus-2's options are accepted and ignored, so tally.run launches this like the agent it grades."""
    writers: str = Field(default="nano,super", description="Comma-separated names from tally.checkeval.WRITERS.")


class CheckEval(BaseAgent):
    options_model = CheckEvalOptions
    options: CheckEvalOptions

    @staticmethod
    def name() -> str:
        return "tally-checkeval"

    def version(self) -> str | None:
        return "1"

    async def setup(self, environment) -> None:
        return

    async def run(self, instruction, environment, context) -> None:
        names = [n.strip() for n in self.options.writers.split(",") if n.strip()]
        unknown = [n for n in names if n not in WRITERS]
        if unknown:
            raise ValueError("unknown writers %s; known: %s" % (unknown, sorted(WRITERS)))
        out = {"writers": {}}
        context.metadata = {"checkeval": out}
        listing = (await environment.exec("ls -la 2>&1 | head -50", timeout_sec=20)).stdout or ""

        async def write(name):
            model, template = WRITERS[name]
            entry = {"model": model}
            try:
                llm = LiteLLM(model_name="openai/" + model, api_base=self.options.api_base)
                checks, _ = await write_checks(llm, template, instruction, listing, {}, entry)
            except Exception as e:
                entry["error"] = repr(e)[:500]
                checks = []
            return name, {"writer": entry, "checks": checks}

        for name, rec in await asyncio.gather(*(write(n) for n in names)):
            out["writers"][name] = rec
        await self._run_all(environment, instruction, out, "start")

        oracle = OracleAgent(logs_dir=self.logs_dir, task_dir=environment.environment_dir.parent,
                             trial_paths=environment.trial_paths)
        await oracle.run(instruction, environment, context)
        context.metadata = {"checkeval": out}             # the oracle does not touch metadata; keep ours on top

        await self._run_all(environment, instruction, out, "solved")
        context.n_input_tokens = sum((r["writer"].get("tokens") or [0, 0])[0] for r in out["writers"].values())
        context.n_output_tokens = sum((r["writer"].get("tokens") or [0, 0])[1] for r in out["writers"].values())

    @staticmethod
    async def _run_all(environment, instruction, out, state):
        for rec in out["writers"].values():
            rec[state] = []
            for check in rec["checks"]:
                r = await run_check(environment, check, instruction)
                rec[state].append({"status": r["status"], "exit": r.get("exit"), "output": r.get("output", "")[-300:]})
