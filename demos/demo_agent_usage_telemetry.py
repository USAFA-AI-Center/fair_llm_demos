"""Usage telemetry as an agent author sees it: cost and truncation by subscription.

A fairlib user defines agents; they do not talk to adapters. This demo
builds a SimpleAgent and subscribes to ModelInvocationEvent on the agent's
own bus, so every model call the agent makes reports its token counts,
its stop reason, and its wall-clock time without the demo touching the
adapter at all. The same function runs the agent under two different
local backends (Ollama and HuggingFace) and never branches on which one
it holds: the event carries the provider label, and the usage record has
the same shape under both.

The second run under each backend gives the model a tiny generation
budget, so the reply is cut off mid-answer and the event's done_reason
reads LENGTH. That is the signal a budget gate or a diagnostic branches
on, and it is also written into the turn the agent stored in memory, so
a session trace carries it after the run is over.

Set FAIR_LLM_DEMO_MODEL to the Ollama model and FAIR_LLM_DEMO_HF_MODEL to
the HuggingFace model served on this machine.
"""

from __future__ import annotations

import asyncio
import os
from typing import List

from fairlib import (
    AbstractChatModel,
    DoneReason,
    HuggingFaceAdapter,
    ModelInvocationEvent,
    OllamaAdapter,
    PlannerParseError,
    SimpleAgent,
    SimpleReActPlanner,
    ToolExecutor,
    ToolRegistry,
    WorkingMemory,
)

OLLAMA_MODEL = os.environ.get("FAIR_LLM_DEMO_MODEL", "qwen3-vl-instruct-16k")
HF_MODEL = os.environ.get("FAIR_LLM_DEMO_HF_MODEL", "dolphin3-qwen25-3b")
QUESTION = "In three sentences, why does the sky look blue?"
BUDGET = 6


def build_agent(llm: AbstractChatModel) -> SimpleAgent:
    """Assemble the smallest working agent around an interface-typed model."""
    tool_registry = ToolRegistry()
    planner = SimpleReActPlanner(llm, tool_registry)
    return SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=ToolExecutor(tool_registry),
        memory=WorkingMemory(),
        max_steps=3,
    )


def describe(event: ModelInvocationEvent) -> str:
    usage = event.usage
    if usage is None:
        return f"  {event.provider}/{event.model_name}: no usage reported"
    done = usage.done_reason.value if usage.done_reason is not None else "unknown"
    return (
        f"  {event.provider}/{event.model_name}: "
        f"prompt={usage.prompt_tokens} completion={usage.completion_tokens} "
        f"done_reason={done} wall_clock={event.duration_ms:.0f} ms "
        f"outcome={event.outcome.value}"
    )


async def run(llm: AbstractChatModel, label: str) -> None:
    """One agent run: subscribe, ask, report; the model is only ever the interface type."""
    agent = build_agent(llm)
    events: List[ModelInvocationEvent] = []
    agent.events.subscribe(ModelInvocationEvent, events.append)

    print(f"\n--- {label} ---")
    try:
        answer = await agent.arun(QUESTION)
    except PlannerParseError as exc:
        # A budget small enough to cut the planner's own turn short ends
        # the run on the typed parse error after the agent's retries; the
        # bus has already said why, one event per attempt.
        print(f"the run ended with {type(exc).__name__}: the planner turn was cut off")
    else:
        print(f"answer: {answer[:160]!r}")
    print("model calls seen on the bus:")
    for event in events:
        print(describe(event))

    stored = [m for m in agent.memory.get_history() if m.role == "assistant"]
    last = stored[-1].usage if stored else None
    if last is None:
        print("stored turn: carries no usage record")
    else:
        print(
            "stored turn: the assistant message in memory carries "
            f"done_reason={last.done_reason.value if last.done_reason else None}"
        )
    if any(
        e.usage is not None and e.usage.done_reason is DoneReason.LENGTH for e in events
    ):
        print("truncation detected: a call stopped on LENGTH, the overflow sentinel")


async def main() -> None:
    print("Ollama backend")
    await run(OllamaAdapter(model_name=OLLAMA_MODEL), "normal budget")
    await run(
        OllamaAdapter(model_name=OLLAMA_MODEL, options={"num_predict": BUDGET}),
        f"budget of {BUDGET} tokens",
    )

    print("\nHuggingFace backend")
    await run(HuggingFaceAdapter(HF_MODEL), "normal budget")
    await run(
        HuggingFaceAdapter(HF_MODEL, max_new_tokens=BUDGET),
        f"budget of {BUDGET} tokens",
    )
    print(
        "\nThe run function never named a provider: the event and the usage "
        "record have one shape under both backends."
    )


if __name__ == "__main__":
    asyncio.run(main())
