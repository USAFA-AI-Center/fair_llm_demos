"""Demo: deterministic chaos-fault suite.

Runs seeded breaker-recovery and session-restore scenarios without GPU or
network access. Mirrors the acceptance demo for the chaos harness.

Run: PYTHONPATH=. python demos/demo_chaos_fault_suite.py
"""

import asyncio
import tempfile

from fairlib.core.base_agent import BaseAgent
from fairlib.core.event_bus import AgentEventBus
from fairlib.core.events import BreakerStateChangeEvent
from fairlib.core.message import Message
from fairlib.core.session import JsonSessionStore
from fairlib.modules.memory.base import WorkingMemory
from fairlib.testing.chaos import ChaosRunAbortError, get_scenario


class _CheckpointAgent(BaseAgent):
    async def arun(self, user_input):
        return str(user_input)


async def _breaker_demo() -> None:
    scenario = get_scenario("breaker_storm")
    schedule = scenario.schedule()
    print(f"Scenario {scenario.name} seed={scenario.seed} schedule={schedule}")

    bus = AgentEventBus()
    transitions: list[str] = []

    def _capture(event: BreakerStateChangeEvent) -> None:
        transitions.append(event.new_state)

    bus.subscribe(BreakerStateChangeEvent, _capture)
    scenario.bind_event_bus(bus)

    outcomes = await scenario.arun_breaker_storm(
        "demo-provider",
        failure_threshold=2,
    )
    for step, (kind, outcome) in enumerate(zip(schedule, outcomes[:-2])):
        print(f"  step {step} ({kind.value}): {outcome}")
    print(f"  {outcomes[-2]}")
    print(f"  {outcomes[-1]}")
    print(f"  breaker transitions observed: {transitions}")


def _checkpoint_demo() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = JsonSessionStore(tmp)
        agent = _CheckpointAgent()
        agent.memory = WorkingMemory(max_size=10)
        agent.memory.add_message(Message(role="user", content="bookmark-here"))
        agent.save_state(metadata={"label": "mid-run"})
        store.save_agent("demo-session", agent, metadata={"demo": "chaos"})

        print(f"  history at kill point: {len(agent.memory.history)} messages")

        recovered = _CheckpointAgent()
        recovered.memory = WorkingMemory(max_size=10)
        recovered._checkpoints = []
        record = store.restore_agent("demo-session", recovered)
        print(f"  restored history: {len(recovered.memory.history)} messages")
        print(f"  surviving content: {recovered.memory.history[0].content!r}")
        print(f"  session metadata: {record.metadata}")
        if recovered._checkpoints:
            print(f"  restored checkpoints: {len(recovered._checkpoints)}")


async def _scheduled_demo() -> None:
    scenario = get_scenario("timeout_focus")
    results = await scenario.run_scheduled()
    print(f"Scenario {scenario.name} scheduled {len(results)} steps")
    for result in results:
        print(f"  step {result.step} ({result.kind.value}): {result.outcome}")


async def main() -> None:
    print("=== Chaos fault suite demo ===")
    print("\n-- Breaker storm (schedule-driven) --")
    await _breaker_demo()
    print("\n-- Scheduled timeout focus --")
    await _scheduled_demo()
    print("\n-- Session kill / restore --")
    _checkpoint_demo()
    print("\n-- Mixed faults abort (partial results) --")
    mixed = get_scenario("mixed_faults")
    try:
        await mixed.run_scheduled()
    except ChaosRunAbortError as exc:
        print(f"  aborted at step {exc.step} with {len(exc.partial_results)} prior results")
    print("\nChaos fault suite demo complete.")


if __name__ == "__main__":
    asyncio.run(main())
