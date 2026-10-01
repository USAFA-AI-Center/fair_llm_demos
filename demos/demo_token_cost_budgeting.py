"""Demo: CostBudget settles one real-model call, then refuses the next.

Builds a SimpleAgent on FAIR_LLM_DEMO_MODEL with a session ceiling above one
call's estimate and below two, so step 1 runs and settles from reported
Usage, then a later call is refused. Also shows the EXTERNAL tool-call
ceiling on BasicSecurityManager.

Requires the local extra (pip install 'fair-llm[local]') and a local model;
defaults to HuggingFaceAdapter("qwen25-7b"). Set FAIR_LLM_DEMO_MODEL
to override.
"""

from __future__ import annotations

import asyncio
import os

from pydantic import BaseModel

from fairlib import (
    BasicSecurityManager,
    BudgetExceededError,
    BudgetExceededEvent,
    CostBudget,
    CostRates,
    HuggingFaceAdapter,
    SideEffect,
    SimpleAgent,
    SimpleReActPlanner,
    TextResult,
    ToolExecutor,
    ToolRegistry,
    WorkingMemory,
)
from fairlib.core.interfaces.tools import AbstractTool, ToolOutput

MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "qwen25-7b")


class _Empty(BaseModel):
    pass


class _ExternalPing(AbstractTool):
    name = "external_ping"
    description = "counts against the EXTERNAL budget"
    input_schema = _Empty
    output_schema = TextResult
    side_effect = SideEffect.EXTERNAL

    async def acall(self, tool_input: _Empty) -> ToolOutput:
        return TextResult(result="pong")


async def demo_cost_gate() -> bool:
    seen: list[BudgetExceededEvent] = []
    # Ceiling above one real-prompt estimate (~0.35 USD at these rates) and
    # below two, so the first call settles from Usage and a later call refuses.
    budget = CostBudget(
        session_usd_ceiling=0.5,
        rates=CostRates(usd_per_prompt_token=0.001, usd_per_completion_token=0.002),
        estimated_completion_tokens=16,
    )
    llm = HuggingFaceAdapter(MODEL_NAME, max_new_tokens=64)
    registry = ToolRegistry()
    agent = SimpleAgent(
        llm=llm,
        planner=SimpleReActPlanner(llm, registry),
        tool_executor=ToolExecutor(registry, BasicSecurityManager()),
        memory=WorkingMemory(),
        max_steps=2,
        budget=budget,
    )
    agent.events.subscribe(BudgetExceededEvent, seen.append)
    print(f"--- cost gate on {MODEL_NAME} ---")
    print(f"ceiling=0.5 USD; session_spent before run={budget.session_spent_usd:.6f}")
    first_ok = False
    try:
        answer = await agent.arun("Reply with exactly the word ok.")
        print(f"first call answered: {answer!r}")
        print(f"settled spend after first call={budget.session_spent_usd:.6f} USD")
        first_ok = budget.session_spent_usd > 0.0
    except BudgetExceededError as exc:
        print(
            f"unexpected first-call refusal: scope={exc.scope} "
            f"limit={exc.limit} observed={exc.observed} estimate={exc.estimate}"
        )
        return False

    refused = False
    try:
        await agent.arun("Reply with exactly the word ok again.")
        print("unexpected: second call allowed")
    except BudgetExceededError as exc:
        print(f"refused: scope={exc.scope} limit={exc.limit} observed={exc.observed}")
        print(f"estimate={exc.estimate}")
        refused = True
    print(f"settled spend after run={budget.session_spent_usd:.6f} USD")
    print(f"BudgetExceededEvent count={len(seen)}")
    return first_ok and refused and len(seen) >= 1


async def demo_external_budget() -> bool:
    security = BasicSecurityManager(external_call_ceiling=1)
    registry = ToolRegistry()
    registry.register_tool(_ExternalPing())
    executor = ToolExecutor(registry, security_manager=security)
    print("--- EXTERNAL tool-call budget ---")
    await executor.aexecute("external_ping", {})
    print(f"after first call: count={security.get_external_tool_count()}")
    ok = False
    try:
        await executor.aexecute("external_ping", {})
        print("unexpected: second EXTERNAL call allowed")
    except BudgetExceededError as exc:
        print(f"refused second call: scope={exc.scope} limit={exc.limit}")
        ok = True
    return ok


def main() -> None:
    cost_ok = asyncio.run(demo_cost_gate())
    external_ok = asyncio.run(demo_external_budget())
    if cost_ok and external_ok:
        print("demo_token_cost_budgeting: ok")
    else:
        print("demo_token_cost_budgeting: incomplete")


if __name__ == "__main__":
    main()
