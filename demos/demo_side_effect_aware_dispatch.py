# demo_side_effect_aware_dispatch.py

"""
Side-effect-aware tool dispatch: run a batch of tool calls safely and fast.

Every tool declares a side-effect class - READ_ONLY, MUTATING, or EXTERNAL. The
executor uses that declaration to schedule a turn's batch of tool calls: a run of
read-only calls is safe to run concurrently, while a mutating or external call is
a sequential barrier that runs alone. Results always come back in the original
call order, no matter how they were scheduled, and one call's failure never
aborts another.

This demo builds three slow read-only tools and one mutating tool, then dispatches
a mixed batch through ToolExecutor.aexecute_batch. It subscribes to the event bus
to print the scheduling decision, times the run to show the read-only calls
overlapped (wall-clock near one delay, not the sum of four), and checks that the
results line up with the input order.

No model, API key, or GPU is needed. Run:
    python demos/demo_side_effect_aware_dispatch.py
"""

import asyncio
import time
from typing import List

from fairlib import (
    Action,
    AgentEventBus,
    ToolExecutor,
    ToolRegistry,
)
from fairlib.core.events import ToolBatchScheduledEvent, ToolCallPostEvent
from fairlib.core.interfaces.tools import (
    AbstractTool,
    SideEffect,
    StringInput,
    TextResult,
    ToolOutput,
)

DELAY = 0.25  # seconds each tool spends "working"


class DemoTool(AbstractTool):
    """A tool that sleeps to simulate work and records when it ran."""

    input_schema = StringInput
    output_schema = TextResult

    def __init__(self, name: str, side_effect: SideEffect, log: List[str]) -> None:
        self.name = name
        self.description = f"A {side_effect.value} demo tool named {name}."
        self.side_effect = side_effect
        self._log = log

    async def acall(self, tool_input: StringInput) -> ToolOutput:
        self._log.append(f"start {self.name}")
        await asyncio.sleep(DELAY)
        self._log.append(f"end   {self.name}")
        return TextResult(result=f"{self.name} handled '{tool_input.input}'")


def _print_schedule(event: ToolBatchScheduledEvent) -> None:
    print(f"\nScheduled {event.batch_size} calls (max parallel = {event.max_parallel_tools}):")
    for i, group in enumerate(event.groups, start=1):
        how = "PARALLEL" if group.parallel else "sequential"
        print(f"  group {i}: {how:11} [{group.side_effect.value}]  {', '.join(group.tool_names)}")


async def main() -> None:
    log: List[str] = []
    registry = ToolRegistry()
    registry.register_tool(DemoTool("search_docs", SideEffect.READ_ONLY, log))
    registry.register_tool(DemoTool("read_config", SideEffect.READ_ONLY, log))
    registry.register_tool(DemoTool("list_files", SideEffect.READ_ONLY, log))
    registry.register_tool(DemoTool("write_report", SideEffect.MUTATING, log))

    bus = AgentEventBus()
    bus.subscribe(ToolBatchScheduledEvent, _print_schedule)
    bus.subscribe(
        ToolCallPostEvent,
        lambda e: print(f"  done: {e.tool_name} (succeeded={e.succeeded})"),
    )

    executor = ToolExecutor(registry, events=bus)

    # Three read-only calls (safe to overlap) followed by a mutating barrier.
    batch = [
        Action(tool_name="search_docs", tool_input="invoices"),
        Action(tool_name="read_config", tool_input="billing"),
        Action(tool_name="list_files", tool_input="."),
        Action(tool_name="write_report", tool_input="summary"),
    ]

    started = time.perf_counter()
    results = await executor.aexecute_batch(batch)
    elapsed = time.perf_counter() - started

    print("\nResults (in original call order):")
    for action, result in zip(batch, results):
        print(f"  {action.tool_name:13} -> {result.observation}")

    # The three read-only calls overlapped, so wall-clock is ~2 delays (one for
    # the parallel read group, one for the mutating barrier), not ~4.
    print(f"\nWall-clock: {elapsed:.2f}s for 4 calls of {DELAY:.2f}s each.")
    print(f"Sequential would have taken ~{4 * DELAY:.2f}s.")

    assert [r.tool_name for r in results] == [a.tool_name for a in batch], "order not preserved"
    assert all(r.succeeded for r in results), "a call failed unexpectedly"
    # The mutating call ran alone: its start follows the last read-only end.
    assert log.index("start write_report") > log.index("end   list_files"), "barrier overlapped reads"
    print("\nOK: order preserved, read-only calls overlapped, mutating call ran as a barrier.")


if __name__ == "__main__":
    asyncio.run(main())
