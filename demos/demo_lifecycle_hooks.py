# demo_lifecycle_hooks.py

"""
Lifecycle hooks in the SimpleAgent ReAct loop.

Unlike the event bus (observe-only), lifecycle hooks can intercept,
modify, or veto actions at pre-model, pre-tool, and post-tool points.
This demo uses a real local model and a real calculator tool - the same
wiring a cadet would copy into an application. A pre-tool hook blocks
every tool but the calculator; a post-tool hook appends an audit label to
every observation. Two turns run: an arithmetic question, whose calculator
observation comes back audit-labelled, and a weather question, whose
weather_lookup call the pre-tool hook blocks before the tool runs. Each
turn prints its LifecycleHookEvent ticker lines from the event bus and the
observations the agent's memory holds, so the modified and the blocked
observation are both visible.

Requirements: a local HuggingFace model (transformers; a GPU is
recommended). The first run may download weights. Set FAIR_LLM_DEMO_MODEL
to override the default (a HuggingFaceAdapter registry alias or hub id).

Run:
    python demos/demo_lifecycle_hooks.py
"""

import asyncio
import os

from pydantic import BaseModel, Field

from fairlib import (
    CallableLifecycleHooks,
    HookResult,
    HuggingFaceAdapter,
    LifecycleHookEvent,
    PostToolHookContext,
    PreToolHookContext,
    RoleDefinition,
    SafeCalculatorTool,
    SimpleAgent,
    SimpleReActPlanner,
    ToolExecutor,
    ToolRegistry,
    WorkingMemory,
)
from fairlib.core.interfaces.tools import (
    AbstractTool,
    SideEffect,
    TextResult,
    ToolOutput,
)
from fairlib.core.message import OBSERVATION_MARKER_KEY, has_marker

# Default matches demos/demo_single_agent_calculator.py; override with
# FAIR_LLM_DEMO_MODEL for a stronger instruct model if needed.
MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "qwen25-7b")


class CityInput(BaseModel):
    city: str = Field(description="The city to look up.")


class WeatherLookupTool(AbstractTool):
    """A registered tool the application's policy does not allow.

    It is in the registry, so the model sees it in the catalog and may call
    it; the pre-tool hook is what stops the call before it runs.
    """

    name = "weather_lookup"
    description = "Returns the current weather for a city."
    input_schema = CityInput
    output_schema = TextResult
    side_effect = SideEffect.READ_ONLY

    async def acall(self, tool_input: CityInput) -> ToolOutput:
        return TextResult(result=f"Sunny and 21 C in {tool_input.city}.")


async def allow_calculator_only(ctx: PreToolHookContext) -> HookResult:
    if ctx.tool_name != "safe_calculator":
        return HookResult.veto(
            f"blocked by policy: only safe_calculator may run, not {ctx.tool_name!r}"
        )
    return HookResult.proceed_default()


async def audit_observation(ctx: PostToolHookContext) -> HookResult:
    return HookResult.modify_observation(
        f"[audited] {ctx.observation}",
        reason="append audit label",
    )


def _on_hook(event: LifecycleHookEvent) -> None:
    print(
        f"  [hook] step={event.step} point={event.hook_point.value} "
        f"action={event.action.value} tool={event.tool_name!r} "
        f"reason={event.reason!r}"
    )


async def main() -> None:
    print(f"Loading local model {MODEL_NAME!r}...")
    llm = HuggingFaceAdapter(MODEL_NAME)

    tool_registry = ToolRegistry()
    tool_registry.register_tool(SafeCalculatorTool())
    tool_registry.register_tool(WeatherLookupTool())

    planner = SimpleReActPlanner(llm, tool_registry)
    planner.prompt_builder.role_definition = RoleDefinition(
        "You are an expert mathematical calculator. Your job is to perform "
        "mathematical calculations.\n"
        "You reason step-by-step to determine the best course of action. "
        "Use safe_calculator for the arithmetic. For a weather question, "
        "call weather_lookup. If a tool call is blocked, say so in your "
        "final answer. Keep final answers short."
    )

    hooks = CallableLifecycleHooks(
        pre_tool=allow_calculator_only,
        post_tool=audit_observation,
    )

    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=ToolExecutor(tool_registry),
        memory=WorkingMemory(max_size=30),
        max_steps=8,
        lifecycle_hooks=hooks,
    )
    agent.events.subscribe(LifecycleHookEvent, _on_hook)

    questions = (
        ("Allowed tool: the post-tool hook labels the observation", "What is 12 * 7?"),
        (
            "Blocked tool: the pre-tool hook vetoes the call before it runs",
            "What is the weather in Denver right now?",
        ),
    )
    for title, question in questions:
        print(f"\n=== {title} ===")
        print(f"You: {question}")
        # The two questions are independent, so each starts from a clean
        # history.
        agent.memory.clear()
        answer = await agent.arun(question)
        print("  Observations this turn, as memory holds them:")
        for message in agent.memory.get_history():
            if has_marker(message, OBSERVATION_MARKER_KEY):
                print(f"    {message.content}")
        print(f"Agent: {answer}")
    print("\nLifecycle hooks demo complete.")


if __name__ == "__main__":
    asyncio.run(main())
