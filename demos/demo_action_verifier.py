# demo_action_verifier.py

"""
Post-action verification in the SimpleAgent ReAct loop (issue #73).

After a tool dispatches successfully, an optional action verifier runs
deterministic checks (rules, linters, tests, tool probes) and feeds
structured feedback back into the loop as an augmented observation.
Failed checks emit ActionVerificationEvent and append
"Verification failed: ..." to the observation committed to memory.

This demo uses a real local model and a real calculator tool - the same
wiring a cadet would copy into an application. The verifier requires
every successful calculator observation to include at least one digit;
watch the event bus print pass/fail as the agent works.

Requirements: a local HuggingFace model (transformers; a GPU is
recommended). The first run may download weights. Set FAIR_LLM_DEMO_MODEL
to override the default (a HuggingFaceAdapter registry alias or hub id).

Run:
    python demos/demo_action_verifier.py
"""

import asyncio
import os

from fairlib import (
    ActionVerificationEvent,
    HuggingFaceAdapter,
    RoleDefinition,
    SafeCalculatorTool,
    SimpleAgent,
    SimpleReActPlanner,
    ToolExecutor,
    ToolRegistry,
    VerificationContext,
    VerificationResult,
    WorkingMemory,
)

# Default matches demos/demo_single_agent_calculator.py; override with
# FAIR_LLM_DEMO_MODEL for a stronger instruct model if needed.
MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "dolphin3-qwen25-3b")


async def require_numeric_observation(ctx: VerificationContext) -> VerificationResult:
    """House rule: successful tool output must include at least one digit.

    Real deployments typically close over expected values, schema checks, or
    secondary tool probes. The framework only sees VerificationResult.
    """
    if any(ch.isdigit() for ch in ctx.observation):
        return VerificationResult.approve()
    return VerificationResult.reject(
        "Calculator output must include at least one digit."
    )


def _on_verification(event: ActionVerificationEvent) -> None:
    status = "passed" if event.passed else f"failed ({event.feedback})"
    print(f"  [verify] tool={event.tool_name!r} step={event.step} {status}")


async def main() -> None:
    print(f"Loading local model {MODEL_NAME!r}...")
    llm = HuggingFaceAdapter(MODEL_NAME)

    tool_registry = ToolRegistry()
    tool_registry.register_tool(SafeCalculatorTool())

    planner = SimpleReActPlanner(llm, tool_registry)
    planner.prompt_builder.role_definition = RoleDefinition(
        "You are an expert mathematical calculator. Your job is to perform "
        "mathematical calculations.\n"
        "You reason step-by-step to determine the best course of action. "
        "When you call safe_calculator, pass only a pure arithmetic "
        "expression in the expression field (for example '45 * 11'), with "
        "no words or question marks. Keep final answers short."
    )

    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=ToolExecutor(tool_registry),
        memory=WorkingMemory(max_size=30),
        max_steps=8,
        action_verifier=require_numeric_observation,
    )
    agent.events.subscribe(ActionVerificationEvent, _on_verification)

    question = "What is 45 * 11?"
    print(f"You: {question}")
    print("Running gather -> act -> verify...")
    answer = await agent.arun(question)
    print(f"Agent: {answer}")
    print("Post-action verification demo complete.")


if __name__ == "__main__":
    asyncio.run(main())
