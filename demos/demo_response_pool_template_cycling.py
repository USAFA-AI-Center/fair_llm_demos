# demo_response_pool_template_cycling.py

"""
Response pool and template cycling demonstration.

Tutor and fallback flows often need varied wording without repeating the same
phrase every time. ResponsePool cycles deterministically through a list of
templates and can persist its cursor across restarts.

Part A shows direct ResponsePool cycling with no LLM. Part B wires the pool
into a small inline tool inside a real HuggingFace-driven agent so the pool
runs inside a full ReAct loop. The demo also shows state and load_state so the
cursor survives a restart.

Run: PYTHONPATH=. python demos/demo_response_pool_template_cycling.py
Requires a GPU for Part B. Set FAIR_LLM_DEMO_MODEL to override the default model.
"""

import asyncio
import os

from pydantic import BaseModel, Field

from fairlib import (
    HuggingFaceAdapter,
    ResponsePool,
    RoleDefinition,
    SimpleAgent,
    SimpleReActPlanner,
    ToolExecutor,
    ToolRegistry,
    WorkingMemory,
)
from fairlib.core.interfaces.tools import AbstractTool, SideEffect, TextResult, ToolOutput

MODEL_NAME = os.getenv("FAIR_LLM_DEMO_MODEL", "qwen25-7b")


class HintInput(BaseModel):
    """A short hint topic the tutor should weave into the redirect."""

    hint: str = Field(description="The core idea to redirect the student toward.")


class TutorRedirectTool(AbstractTool):
    """Returns the next deterministic tutor redirect from a ResponsePool."""

    name = "tutor_redirect"
    description = (
        "Returns a varied, non-repetitive tutor redirect message. "
        "Pass the hint topic as tool_input."
    )
    input_schema = HintInput
    output_schema = TextResult
    side_effect = SideEffect.READ_ONLY

    def __init__(self, pool: ResponsePool) -> None:
        self.pool = pool

    async def acall(self, tool_input: HintInput) -> ToolOutput:
        return TextResult(result=self.pool.render(hint=tool_input.hint))


def demo_pool_primitive() -> ResponsePool:
    """Part A: show deterministic cycling without any model."""
    print("=== Part A: ResponsePool primitive (no LLM) ===")
    pool = ResponsePool(
        [
            "Let's try a smaller step: {hint}",
            "Another way to approach it: {hint}",
            "Redirecting to the core idea: {hint}",
        ]
    )
    for topic in ("factor first", "draw the graph", "check units"):
        print("Redirect:", pool.render(hint=topic))
    return pool


async def demo_pool_in_agent(pool: ResponsePool) -> None:
    """Part B: the same pool inside a real agent tool loop."""
    print("\n=== Part B: ResponsePool inside a real agent loop ===")
    print(f"Loading {MODEL_NAME} via HuggingFaceAdapter...")
    llm = HuggingFaceAdapter(MODEL_NAME, max_new_tokens=256)

    registry = ToolRegistry()
    registry.register_tool(TutorRedirectTool(pool))
    executor = ToolExecutor(registry)
    planner = SimpleReActPlanner(llm, registry)
    planner.prompt_builder.role_definition = RoleDefinition(
        "You are a patient math tutor. When the student is stuck, call tutor_redirect "
        "with a short hint topic, then summarize the redirect for the student."
    )
    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=executor,
        memory=WorkingMemory(),
        max_steps=6,
    )

    prompt = "The student is stuck on factoring a quadratic. Give a gentle redirect."
    print(f"\nUser: {prompt}")
    print("Agent:", await agent.arun(prompt))


def demo_pool_persistence(pool: ResponsePool) -> None:
    """Show cursor save/load for deterministic behavior across restarts."""
    print("\n=== Persistence: state() / load_state() ===")
    saved = pool.state()
    print("Saved cursor index:", saved.index)
    restored = ResponsePool(["A: {hint}", "B: {hint}", "C: {hint}"])
    restored.load_state({"index": saved.index})
    print("Restored peek:", restored.peek().format(hint="same cursor"))


async def main() -> None:
    pool = demo_pool_primitive()
    await demo_pool_in_agent(pool)
    demo_pool_persistence(pool)


if __name__ == "__main__":
    asyncio.run(main())
