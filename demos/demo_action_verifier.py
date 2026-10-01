# demo_action_verifier.py

"""
Post-action verification in the SimpleAgent ReAct loop.

After a tool dispatches successfully, an optional action verifier runs
deterministic checks (rules, linters, tests, tool probes) and feeds
structured feedback back into the loop as an augmented observation.
Failed checks emit ActionVerificationEvent and append
"Verification failed: ..." to the observation committed to memory.

This demo uses a real local model and a real calculator tool - the same
wiring a cadet would copy into an application. The verifier enforces two
house rules for a budget assistant: every successful calculator
observation must include a number, and a result must not be negative (a
shortfall is reported as "over budget by N", never as a negative amount).
The first question passes both rules. The second one computes a negative
balance, so the verifier rejects it and the agent reads the feedback on
its next step. Watch the event bus print pass/fail as the agent works, and
the observation the agent actually saw after each check.

Requirements: a local HuggingFace model (transformers; a GPU is
recommended). The first run may download weights. Set FAIR_LLM_DEMO_MODEL
to override the default (a HuggingFaceAdapter registry alias or hub id).

Run:
    python demos/demo_action_verifier.py
"""

import asyncio
import os
import re

from fairlib import (
    ActionVerificationEvent,
    HuggingFaceAdapter,
    RoleDefinition,
    SafeCalculatorTool,
    SimpleAgent,
    SimpleReActPlanner,
    ToolCallPostEvent,
    ToolExecutor,
    ToolRegistry,
    VerificationContext,
    VerificationResult,
    WorkingMemory,
)

# Default matches demos/demo_single_agent_calculator.py; override with
# FAIR_LLM_DEMO_MODEL for a stronger instruct model if needed.
MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "qwen25-7b")


# The number at the end of a calculator observation, such as "... is -55".
_RESULT = re.compile(r"is (-?\d+(?:\.\d+)?)\s*$")


async def check_budget_result(ctx: VerificationContext) -> VerificationResult:
    """House rules: the output carries a number, and that number is not negative.

    Real deployments typically close over expected values, schema checks, or
    secondary tool probes. The framework only sees VerificationResult.
    """
    match = _RESULT.search(ctx.observation)
    if match is None:
        return VerificationResult.reject(
            "Calculator output must end with a numeric result."
        )
    if float(match.group(1)) < 0:
        return VerificationResult.reject(
            "The result is negative. This assistant never reports a negative "
            "amount: report the shortfall as 'over budget by N dollars', "
            "where N is the absolute value."
        )
    return VerificationResult.approve()


# The observation each tool call returned, keyed by loop step and tool name,
# so the verification handler can show what the verifier judged.
_observations: dict[tuple[int | None, str], str] = {}


def _on_tool_call(event: ToolCallPostEvent) -> None:
    _observations[(event.step, event.tool_name)] = event.observation
    print(f"  [tool] {event.tool_name} {event.tool_input!r} -> {event.observation}")


def _on_verification(event: ActionVerificationEvent) -> None:
    status = "passed" if event.passed else "failed"
    print(f"  [verify] tool={event.tool_name!r} step={event.step} {status}")
    if not event.passed:
        # A failed check appends its feedback to the observation, and the
        # agent reads both on its next step.
        observation = _observations.get((event.step, event.tool_name), "")
        print(f"    observation the agent read: {observation}")
        print(f"    with the verifier's feedback: {event.feedback}")


async def main() -> None:
    print(f"Loading local model {MODEL_NAME!r}...")
    llm = HuggingFaceAdapter(MODEL_NAME)

    tool_registry = ToolRegistry()
    tool_registry.register_tool(SafeCalculatorTool())

    planner = SimpleReActPlanner(llm, tool_registry)
    planner.prompt_builder.role_definition = RoleDefinition(
        "You are a budget assistant. Your job is to perform the budget "
        "calculations the user asks for.\n"
        "You reason step-by-step to determine the best course of action. "
        "Use safe_calculator for the arithmetic. Keep final answers short."
    )

    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=ToolExecutor(tool_registry),
        memory=WorkingMemory(max_size=30),
        max_steps=8,
        action_verifier=check_budget_result,
    )
    agent.events.subscribe(ActionVerificationEvent, _on_verification)
    agent.events.subscribe(ToolCallPostEvent, _on_tool_call)

    questions = (
        "We bought 45 chairs at 11 dollars each. What did they cost in total?",
        "Our budget is 120 dollars and we spent 175. What is 120 - 175, "
        "our remaining budget?",
    )
    for question in questions:
        print(f"\nYou: {question}")
        print("Running gather -> act -> verify...")
        answer = await agent.arun(question)
        print(f"Agent: {answer}")
    print("\nPost-action verification demo complete.")


if __name__ == "__main__":
    asyncio.run(main())
