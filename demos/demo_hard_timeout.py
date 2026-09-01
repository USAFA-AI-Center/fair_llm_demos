"""Demo: a hard wall-clock timeout bounds a hung provider call.

Drives the real OllamaAdapter with a fake transport that never answers.
Part 1 calls the adapter directly; Part 2 puts the same adapter behind a
SimpleAgent. In both, the caller receives a typed DegradedResponse(TIMEOUT)
in roughly the configured deadline - not after the hang duration - and the
agent emits a DegradedResponseEvent before the error leaves arun().
"""

import asyncio
import time

from fairlib import (
    DegradedResponse,
    DegradedResponseEvent,
    Message,
    OllamaAdapter,
    SimpleAgent,
    SimpleReActPlanner,
    ToolExecutor,
    ToolRegistry,
    WorkingMemory,
)

HANG_SECONDS = 30.0
TIMEOUT_SECONDS = 0.5


class HungTransport:
    """Stands in for httpx.AsyncClient; the server never answers."""

    async def post(self, *args, **kwargs):
        await asyncio.sleep(HANG_SECONDS)


def _hung_adapter() -> OllamaAdapter:
    adapter = OllamaAdapter(model_name="demo-model", timeout=TIMEOUT_SECONDS)
    adapter.client = HungTransport()
    return adapter


def part1_adapter() -> None:
    print("=== Part 1: the adapter alone ===")
    adapter = _hung_adapter()
    caught = None
    start = time.monotonic()
    try:
        asyncio.run(adapter.ainvoke([Message(role="user", content="hello?")]))
    except DegradedResponse as exc:
        caught = exc
    elapsed = time.monotonic() - start

    if caught is None:
        print("No DegradedResponse was raised - the call completed unexpectedly.")
    else:
        print(f"Typed signal: kind={caught.kind.value}, retryable={caught.retryable}")
    print(
        f"Elapsed: {elapsed:.2f}s (timeout={TIMEOUT_SECONDS}s, hang={HANG_SECONDS}s)\n"
    )


def part2_agent() -> None:
    print("=== Part 2: the same adapter behind a SimpleAgent ===")
    llm = _hung_adapter()
    tool_registry = ToolRegistry()  # a toolless agent still needs an executor
    agent = SimpleAgent(
        llm=llm,
        planner=SimpleReActPlanner(llm, tool_registry),
        tool_executor=ToolExecutor(tool_registry),
        memory=WorkingMemory(),
        max_steps=3,
    )
    events = []
    agent.events.subscribe(DegradedResponseEvent, events.append)

    caught = None
    start = time.monotonic()
    try:
        asyncio.run(agent.arun("hello?"))
    except DegradedResponse as exc:
        caught = exc
    elapsed = time.monotonic() - start

    if caught is None:
        print("No DegradedResponse was raised - the run completed unexpectedly.")
    else:
        print(
            f"Typed signal out of arun(): kind={caught.kind.value}, retryable={caught.retryable}"
        )
    print(f"DegradedResponseEvents emitted on the agent's bus: {len(events)}")
    print(f"Elapsed: {elapsed:.2f}s (timeout={TIMEOUT_SECONDS}s, hang={HANG_SECONDS}s)")
    print("Hard timeout demo complete.")


def main() -> None:
    part1_adapter()
    part2_agent()


if __name__ == "__main__":
    main()
