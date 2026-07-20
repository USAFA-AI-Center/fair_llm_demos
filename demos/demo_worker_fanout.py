# demo_worker_fanout.py

"""
Parallel worker fan-out: a real model managing a team of real agents.

A real local model (loaded through the HuggingFaceAdapter) plays the
manager. It reads the worker roster from the rendered tool catalog,
decides on its own that the request splits into independent subtasks, and
delegates them together in one turn. The manager is not a special
orchestrator: it is a plain SimpleAgent built by build_worker_manager,
whose tools are worker agents wrapped in WorkerAgentTool. The delegation
turn is therefore an ordinary ToolCallBatch, so the side-effect-aware
executor runs the READ_ONLY worker delegations concurrently, commits one
observation per delegation in call order, and emits the usual scheduling
and per-call events.

What it shows:
  - WorkerAgentTool adapts any BaseAgent into a typed tool; each worker
    declares its own SideEffect (READ_ONLY here, so the fan-out overlaps;
    the conservative default is EXTERNAL, a sequential barrier).
  - build_worker_manager is pure wiring: MultiActionReActPlanner plus
    ToolExecutor over a registry of worker tools, one shared event bus.
  - Workers are ordinary stateless SimpleAgents with their own planners
    and tools - the same agents you would build standalone.
  - All instrumentation rides the shared event bus: the scheduler's
    grouping decision (ToolBatchScheduledEvent) and per-delegation timing
    and concurrency (ToolCallPreEvent/ToolCallPostEvent). No subclassing
    is needed to observe the system.

A note on the concurrency you will see: both workers share one locally
loaded model, so the two delegations' generate calls run in separate
threads against the same weights. The delegations are genuinely dispatched
and awaited concurrently; how much wall-clock that saves depends on the
backend (GPU scheduling, model size). The demo reports measured numbers
and only claims real overlap when the combined worker time exceeds the
total wall-clock.

Requirements: a local HuggingFace model (transformers; a GPU is
recommended). The first run downloads the weights. Set FAIR_LLM_DEMO_MODEL
to override the default. The model-driven fan-out is best-effort on the
model emitting a clean multi-action JSON turn - if it delegates one
subtask per turn instead, the run still completes; re-run or use a
stronger instruct model to see the parallel batch.

Run:
    python demos/demo_worker_fanout.py
"""

import asyncio
import os
import time
from typing import List

from fairlib import (
    AbstractChatModel,
    AgentEventBus,
    HuggingFaceAdapter,
    ReActPlanner,
    SafeCalculatorTool,
    SimpleAgent,
    ToolExecutor,
    ToolRegistry,
    WorkerAgentTool,
    WorkingMemory,
    build_worker_manager,
)
from fairlib.core.events import (
    ToolBatchScheduledEvent,
    ToolCallPostEvent,
    ToolCallPreEvent,
)
from fairlib.core.interfaces.tools import (
    AbstractTool,
    SideEffect,
    StringInput,
    TextResult,
    ToolOutput,
)
from fairlib.core.message import OBSERVATION_PREFIX

MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# A tiny canned knowledge base so the researcher worker is self-contained.
# The TOOL is deterministic; the agents driving it are real models.
_FACTS = {
    "france": "The capital of France is Paris; its population is about 68 million.",
    "japan": "The capital of Japan is Tokyo; its population is about 125 million.",
    "brazil": "The capital of Brazil is Brasilia; its population is about 203 million.",
}

# Bus-fed instrumentation: delegation start times, finished durations, and
# how many delegations were in flight at once. ToolCallPreEvent fires as
# each call starts and ToolCallPostEvent as it completes, so the manager's
# shared bus observes the whole fan-out without touching the tools.
_starts: dict = {}
_durations: List[float] = []
_in_flight = {"now": 0, "max": 0}


class _CountryFactsTool(AbstractTool):
    """A read-only lookup over the canned facts table."""

    name = "country_facts"
    description = "Look up basic facts (capital, population) about a country by name."
    input_schema = StringInput
    output_schema = TextResult
    side_effect = SideEffect.READ_ONLY

    async def acall(self, tool_input: StringInput) -> ToolOutput:
        key = tool_input.input.strip().lower()
        return TextResult(result=_FACTS.get(key, "No facts on record for that country."))


def create_worker(llm: AbstractChatModel, tools: List[AbstractTool]) -> SimpleAgent:
    """Build an ordinary stateless worker agent around its own tools.

    This is the same construction a standalone agent uses; nothing about a
    worker is manager-specific until WorkerAgentTool wraps it. The manager
    model learns what each worker is for from the WorkerAgentTool
    description in its rendered tool catalog.
    """
    registry = ToolRegistry()
    for tool in tools:
        registry.register_tool(tool)
    return SimpleAgent(
        llm,
        ReActPlanner(llm, registry),
        ToolExecutor(registry),
        WorkingMemory(),
        stateless=True,
    )


def _on_schedule(event: ToolBatchScheduledEvent) -> None:
    print(f"\n[scheduler] {event.batch_size} delegations in one turn:")
    for i, group in enumerate(event.groups, start=1):
        how = "PARALLEL" if group.parallel else "sequential"
        print(f"  group {i}: {how:11} [{group.side_effect.value}] {', '.join(group.tool_names)}")


def _on_pre(event: ToolCallPreEvent) -> None:
    _in_flight["now"] += 1
    _in_flight["max"] = max(_in_flight["max"], _in_flight["now"])
    _starts[(event.step, event.tool_name)] = time.perf_counter()


def _on_post(event: ToolCallPostEvent) -> None:
    _in_flight["now"] -= 1
    started = _starts.pop((event.step, event.tool_name), None)
    duration = time.perf_counter() - started if started is not None else 0.0
    _durations.append(duration)
    print(
        f"  [done] {event.tool_name} in {duration:.1f}s -> "
        f"{event.observation[:100]} (ok={event.succeeded})"
    )


async def main() -> None:
    print(f"Loading {MODEL_NAME} via the HuggingFaceAdapter (first run downloads weights)...")
    llm = HuggingFaceAdapter(MODEL_NAME, max_new_tokens=512)

    # Two real specialist agents; the whole team shares one loaded model.
    researcher = create_worker(llm, [_CountryFactsTool()])
    analyst = create_worker(llm, [SafeCalculatorTool()])

    # Wrap each worker as a typed tool. READ_ONLY is the author's assertion
    # that the worker mutates nothing, which is what allows delegations to
    # overlap; the conservative default is EXTERNAL (a sequential barrier).
    worker_tools = [
        WorkerAgentTool(
            researcher,
            name="researcher",
            description=(
                "Delegate a research subtask, phrased as a complete question, "
                "to an agent that can look up country facts (capitals, populations)."
            ),
            side_effect=SideEffect.READ_ONLY,
        ),
        WorkerAgentTool(
            analyst,
            name="analyst",
            description=(
                "Delegate a math subtask, phrased as a complete question, "
                "to an agent with a calculator."
            ),
            side_effect=SideEffect.READ_ONLY,
        ),
    ]

    bus = AgentEventBus()
    bus.subscribe(ToolBatchScheduledEvent, _on_schedule)
    bus.subscribe(ToolCallPreEvent, _on_pre)
    bus.subscribe(ToolCallPostEvent, _on_post)

    # The manager is a plain SimpleAgent over the worker tools; the model
    # sees the workers through the rendered tool catalog and decides for
    # itself when to fan out.
    manager = build_worker_manager(llm, worker_tools, events=bus)

    question = (
        "What is the capital of France, and separately, what is 125 * 8? "
        "These are independent questions."
    )
    print(f"Model: {MODEL_NAME}")
    print(f"Question: {question}")

    started = time.perf_counter()
    answer = await manager.arun(question)
    elapsed = time.perf_counter() - started

    worker_total = sum(_durations)
    print(f"\nFinal answer: {answer}")
    print(
        f"\nWall-clock: {elapsed:.1f}s total. Combined worker time: "
        f"{worker_total:.1f}s. Peak concurrent delegations: {_in_flight['max']}."
    )
    if _in_flight["max"] >= 2 and worker_total > elapsed:
        print(
            "The manager issued both delegations in one turn and they "
            "genuinely overlapped: the workers' combined time exceeds the "
            "whole run's wall-clock. The manager is a plain SimpleAgent - "
            "the batch loop did all the work."
        )
    elif _in_flight["max"] >= 2:
        print(
            "The manager issued both delegations in one turn and they were "
            "dispatched concurrently, but compute largely serialized on the "
            "shared local model this run. With an I/O-bound backend or "
            "per-worker models, the same wiring converts the overlap into "
            "wall-clock savings."
        )
    else:
        print(
            "The model delegated one subtask per turn this run, so each took "
            "the single-call path. Re-run, or set FAIR_LLM_DEMO_MODEL to a "
            "stronger instruct model, to see a parallel fan-out turn."
        )

    print("\nManager memory (one observation per delegation, in call order):")
    for message in manager.memory.get_history():
        if message.content.startswith(OBSERVATION_PREFIX):
            print(f"  {message.content[:160]}")


if __name__ == "__main__":
    asyncio.run(main())
