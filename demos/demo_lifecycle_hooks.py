# demo_lifecycle_hooks.py

"""
Lifecycle hooks in the SimpleAgent ReAct loop (issue #74).

Unlike the event bus (observe-only), lifecycle hooks can intercept,
modify, or veto actions at pre-model, pre-tool, and post-tool points.
This demo uses a real local model and a real calculator tool - the same
wiring a cadet would copy into an application. A pre-tool hook blocks
non-calculator tools; a post-tool hook appends an audit label to every
observation. Watch LifecycleHookEvent ticker lines on the event bus.

Requirements: a local HuggingFace model (transformers; a GPU is
recommended). The first run may download weights. Set FAIR_LLM_DEMO_MODEL
to override the default (a HuggingFaceAdapter registry alias or hub id).

Run:
    python demos/demo_lifecycle_hooks.py
"""

import asyncio
import os

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

# Default matches demos/demo_single_agent_calculator.py; override with
# FAIR_LLM_DEMO_MODEL for a stronger instruct model if needed.
MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "dolphin3-qwen25-3b")


async def allow_calculator_only(ctx: PreToolHookContext) -> HookResult:
    if ctx.tool_name != "safe_calculator":
        return HookResult.veto(f"tool {ctx.tool_name!r} is not allowed")
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

    planner = SimpleReActPlanner(llm, tool_registry)
    planner.prompt_builder.role_definition = RoleDefinition(
        "You are an expert mathematical calculator. Your job is to perform "
        "mathematical calculations.\n"
        "You reason step-by-step to determine the best course of action. "
        "When you call safe_calculator, pass only a pure arithmetic "
        "expression in the expression field (for example '12 * 7'), with "
        "no words or question marks. Keep final answers short."
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

    question = "What is 12 * 7?"
    print(f"You: {question}")
    print("Running gather -> hook -> act -> hook...")
    answer = await agent.arun(question)
    print(f"Agent: {answer}")
    print("Lifecycle hooks demo complete.")


if __name__ == "__main__":
    asyncio.run(main())
