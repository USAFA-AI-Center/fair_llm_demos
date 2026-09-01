# demo_multi_tool_turn.py

"""
A real local model driving a multi-tool turn.

A real local model (loaded through the HuggingFaceAdapter) decides, on its
own, that a turn needs two independent tool calls and emits them together; the
agent loop dispatches that batch through the side-effect-aware executor (the
two read-only lookups run in parallel), then the model synthesizes a single
final answer in a follow-up turn that takes the ordinary one-call path.

What it shows:
  - MultiActionReActPlanner turns one completion into one OR several actions.
  - SimpleAgent runs a ToolCallBatch through aexecute_batch (parallel reads),
    and a single action through aexecute - the LLM chooses which, not the demo.
  - One shared event bus carries the executor's ToolBatchScheduledEvent and
    per-call ToolCallPostEvent, so the scheduling decision is observable.
  - A closing pass drives the executor directly with a mixed read/mutate
    batch and prints what the scheduler did: result order, how many reads
    overlapped, and where the mutating call ran.

Requirements: a local HuggingFace model (transformers; a GPU is recommended).
The first run downloads the weights. Set FAIR_LLM_DEMO_MODEL to override the
default. The model-driven part depends on the model emitting clean
multi-action JSON on a given run.

Run:
    python demos/demo_multi_tool_turn.py
"""

import asyncio
import os
import time

from fairlib import (
    AgentEventBus,
    HuggingFaceAdapter,
    MultiActionReActPlanner,
    SimpleAgent,
    ToolExecutor,
    ToolRegistry,
    WorkingMemory,
)
from fairlib.core.events import ToolBatchScheduledEvent, ToolCallPostEvent
from fairlib.core.interfaces.tools import (
    AbstractTool,
    SideEffect,
    StringInput,
    TextResult,
    ToolOutput,
)

MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# Tiny canned knowledge bases so the demo is self-contained and deterministic.
_CAPITALS = {"france": "Paris", "japan": "Tokyo", "brazil": "Brasilia"}
_POPULATIONS = {"france": "68 million", "japan": "125 million", "brazil": "203 million"}

# A shared concurrency tracker so we can prove the two read-only calls overlapped.
_in_flight = {"now": 0, "max": 0}


class _SlowLookupTool(AbstractTool):
    """A read-only lookup that sleeps to simulate I/O, so parallelism is visible."""

    input_schema = StringInput
    output_schema = TextResult
    side_effect = SideEffect.READ_ONLY

    def __init__(self, name: str, description: str, table: dict) -> None:
        self.name = name
        self.description = description
        self._table = table

    async def acall(self, tool_input: StringInput) -> ToolOutput:
        _in_flight["now"] += 1
        _in_flight["max"] = max(_in_flight["max"], _in_flight["now"])
        try:
            await asyncio.sleep(0.5)
            value = self._table.get(tool_input.input.strip().lower(), "unknown")
            return TextResult(result=value)
        finally:
            _in_flight["now"] -= 1


class _ReportWriterTool(AbstractTool):
    """A mutating write that appends to a shared log, so barrier ordering is
    provable: every read that entered before the write must have finished
    before the write starts."""

    input_schema = StringInput
    output_schema = TextResult
    side_effect = SideEffect.MUTATING

    name = "save_report"
    description = "Persist a report line to the shared log."

    def __init__(self, log: list) -> None:
        self._log = log

    async def acall(self, tool_input: StringInput) -> ToolOutput:
        # A mutating barrier never overlaps a read; record how many were in
        # flight when the write started so the closing pass can show it.
        _in_flight["reads_during_write"] = _in_flight["now"]
        self._log.append(tool_input.input)
        return TextResult(result="saved")


async def show_scheduling_contract() -> None:
    """Drive the executor directly with a mixed read/mutate batch.

    The model-driven section can only show a parallel batch when the model
    happens to emit one; this pass shows the scheduling contract on demand:
    results come back in original call order, the read-only calls overlap,
    and the mutating call runs as a barrier with no read in flight.
    """
    from fairlib.core.message import Action

    _in_flight["now"] = 0
    _in_flight["max"] = 0
    _in_flight["reads_during_write"] = 0
    write_log: list = []
    registry = ToolRegistry()
    registry.register_tool(
        _SlowLookupTool(
            "get_capital", "Return the capital city of a country.", _CAPITALS
        )
    )
    registry.register_tool(
        _SlowLookupTool(
            "get_population", "Return the population of a country.", _POPULATIONS
        )
    )
    registry.register_tool(_ReportWriterTool(write_log))

    executor = ToolExecutor(registry)
    calls = [
        Action("get_capital", "France"),
        Action("get_population", "Japan"),
        Action("save_report", "capitals-and-populations"),
        Action("get_capital", "Brazil"),
    ]
    results = await executor.aexecute_batch(calls)

    print("\n[scheduling] batch of four calls: read, read, write, read")
    print(f"  results in original order: {[r.tool_name for r in results]}")
    print(f"  every call succeeded:      {all(r.succeeded for r in results)}")
    print(f"  peak concurrent reads:     {_in_flight['max']}")
    print(f"  reads in flight at write:  {_in_flight['reads_during_write']}")
    print(f"  write log after the batch: {write_log}")


def _on_schedule(event: ToolBatchScheduledEvent) -> None:
    print(
        f"\n[scheduler] {event.batch_size} calls, max parallel {event.max_parallel_tools}:"
    )
    for i, group in enumerate(event.groups, start=1):
        how = "PARALLEL" if group.parallel else "sequential"
        print(
            f"  group {i}: {how:11} [{group.side_effect.value}] {', '.join(group.tool_names)}"
        )


async def main() -> None:
    registry = ToolRegistry()
    registry.register_tool(
        _SlowLookupTool(
            "get_capital", "Return the capital city of a country.", _CAPITALS
        )
    )
    registry.register_tool(
        _SlowLookupTool(
            "get_population", "Return the population of a country.", _POPULATIONS
        )
    )

    # One bus shared by the executor (which owns batch-path emission) and the agent.
    bus = AgentEventBus()
    bus.subscribe(ToolBatchScheduledEvent, _on_schedule)
    bus.subscribe(
        ToolCallPostEvent,
        lambda e: print(
            f"  [done] {e.tool_name} -> {e.observation} (ok={e.succeeded})"
        ),
    )

    print(
        f"Loading {MODEL_NAME} via the HuggingFaceAdapter (first run downloads weights)..."
    )
    llm = HuggingFaceAdapter(MODEL_NAME, max_new_tokens=512)
    executor = ToolExecutor(registry, events=bus)
    planner = MultiActionReActPlanner(llm, registry)
    memory = WorkingMemory()
    agent = SimpleAgent(llm, planner, executor, memory, events=bus)

    question = "What is the capital of France and the population of France?"
    print(f"Model: {MODEL_NAME}")
    print(f"Question: {question}")

    started = time.perf_counter()
    answer = await agent.arun(question)
    elapsed = time.perf_counter() - started

    print(f"\nFinal answer: {answer}")
    print(
        f"\nWall-clock: {elapsed:.2f}s. Peak concurrent tool calls: {_in_flight['max']}."
    )
    if _in_flight["max"] >= 2:
        print(
            "The model issued the two independent lookups in one turn and they ran "
            "in parallel (peak concurrency 2). Two sequential 0.5s reads would have "
            "cost ~1.0s for that turn alone."
        )
    else:
        print(
            "The model issued the lookups one per turn this run, so each took the "
            "single-call path. Re-run, or set FAIR_LLM_DEMO_MODEL to a stronger "
            "instruct model, to see a parallel batch."
        )


if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(show_scheduling_contract())
