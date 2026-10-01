"""Demo: leased queue crash/resume with a real model and CostBudget.

Shows CheckpointWorkRunner claiming a job inside run_one, crashing mid-run
(unexpected exception -> release + JobRestartEvent), then a second runner
restoring the conversation from JsonSessionStore and finishing. The answer
the resumed run returns is read from the finished job record and from the
JobCompletedEvent the queue emits. Prints settled spend from the bound
CostBudget.

Requires FAIR_LLM_DEMO_MODEL (default qwen25-7b) and
pip install 'fair-llm[local]' for HuggingFaceAdapter / torch.

Run: PYTHONPATH=. python demos/demo_orchestration_layer.py
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

from fairlib import (
    CostBudget,
    CostRates,
    HuggingFaceAdapter,
    RoleDefinition,
    SafeCalculatorTool,
    SimpleAgent,
    SimpleReActPlanner,
    ToolCallPostEvent,
    ToolExecutor,
    ToolRegistry,
    WorkingMemory,
)
from fairlib.core.event_bus import AgentEventBus
from fairlib.core.events import (
    JobClaimedEvent,
    JobCompletedEvent,
    JobReleasedEvent,
    JobRestartEvent,
)
from fairlib.core.interfaces.orchestration import OrchestrationLimits
from fairlib.core.message import Message
from fairlib.core.session import JsonSessionStore
from fairlib.modules.orchestration import CheckpointWorkRunner, FileWorkQueue

MODEL_NAME = os.getenv("FAIR_LLM_DEMO_MODEL", "qwen25-7b")


class _SimulatedWorkerDeath(Exception):
    """Demo-only crash marker (not FairlibError, so the runner releases)."""


class _CrashOnceAgent(SimpleAgent):
    """First arun records a partial turn then raises; resume runs normally."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._crashed = False

    async def arun(
        self,
        user_input: Union[str, Message],
        *,
        resume: bool = False,
        **kwargs: Any,
    ) -> str:
        if resume or self._crashed:
            return await super().arun(user_input, resume=resume, **kwargs)
        text = (
            user_input.content if isinstance(user_input, Message) else str(user_input)
        )
        self.memory.add_message(Message(role="user", content=text))
        self.memory.add_message(
            Message(role="assistant", content="partial thought before crash")
        )
        self._crashed = True
        raise _SimulatedWorkerDeath("simulated worker death")


def _build_agent(budget: CostBudget, *, crash_once: bool = False) -> SimpleAgent:
    llm = HuggingFaceAdapter(MODEL_NAME, max_new_tokens=128)
    llm.bind_cost_budget(budget)
    registry = ToolRegistry()
    registry.register_tool(SafeCalculatorTool())
    executor = ToolExecutor(registry)
    planner = SimpleReActPlanner(llm, registry)
    planner.prompt_builder.role_definition = RoleDefinition(
        "You are a concise calculator assistant. Prefer the calculator tool."
    )
    cls = _CrashOnceAgent if crash_once else SimpleAgent
    return cls(
        llm=llm,
        planner=planner,
        tool_executor=executor,
        memory=WorkingMemory(),
        max_steps=6,
        budget=budget,
    )


def _on_tool_call(event: ToolCallPostEvent) -> None:
    status = "ok" if event.succeeded else "failed"
    print(
        f"  [{event.tool_name}] {event.tool_input!r} -> {status}: {event.observation}"
    )


async def main() -> None:
    print(f"Loading {MODEL_NAME} via HuggingFaceAdapter...")
    bus = AgentEventBus()
    claims: list[JobClaimedEvent] = []
    releases: list[JobReleasedEvent] = []
    restarts: list[JobRestartEvent] = []
    completions: list[JobCompletedEvent] = []
    bus.subscribe(JobClaimedEvent, claims.append)
    bus.subscribe(JobReleasedEvent, releases.append)
    bus.subscribe(JobRestartEvent, restarts.append)
    bus.subscribe(JobCompletedEvent, completions.append)

    budget = CostBudget(
        session_usd_ceiling=1.0,
        rates=CostRates(
            usd_per_prompt_token=0.000001, usd_per_completion_token=0.000002
        ),
        estimated_completion_tokens=64,
        events=bus,
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        limits = OrchestrationLimits(
            claim_lease_seconds=30.0,
            max_attempts=0,
            base_backoff_seconds=0.1,
            max_backoff_seconds=1.0,
        )
        queue = FileWorkQueue(root / "queue", events=bus, limits=limits)
        sessions = JsonSessionStore(root / "sessions")
        queue.enqueue({"goal": "Calculate 10 + 5."}, job_id="job-calc")
        queue.enqueue({"goal": "Calculate 3 * 4."}, job_id="job-extra")

        agent = _build_agent(budget, crash_once=True)
        runner = CheckpointWorkRunner(
            queue,
            worker_id="demo-worker",
            sessions=sessions,
            limits=limits,
        )

        print("run_one #1 (expect crash inside runner)...")
        first = await runner.run_one(agent)
        print(
            f"after crash: job={first.job_id if first else None} "
            f"status={first.status.value if first else None} "
            f"attempt={first.attempt if first else None}"
        )
        print(
            f"JobReleasedEvent count={len(releases)} "
            f"JobRestartEvent count={len(restarts)}"
        )
        if restarts:
            r0 = restarts[0]
            print(
                f"restart: attempt={r0.attempt} backoff={r0.backoff_seconds}s "
                f"reason={r0.reason}"
            )

        agent2 = _build_agent(budget, crash_once=False)
        runner2 = CheckpointWorkRunner(
            queue,
            worker_id="demo-worker-2",
            sessions=sessions,
            limits=limits,
        )
        agent2.events.subscribe(ToolCallPostEvent, _on_tool_call)
        finished = await runner2.run_one(agent2)
        print(
            f"resumed job={finished.job_id if finished else None} "
            f"status={finished.status.value if finished else None} "
            f"attempt={finished.attempt if finished else None}"
        )
        if finished is not None:
            # The answer arun returned is on the finished record, and the
            # queue's completion event carries the same value.
            print(f"resumed job answer (job record): {finished.result}")
            completed_answer: Optional[str] = (
                completions[-1].result if completions else None
            )
            print(f"resumed job answer (JobCompletedEvent): {completed_answer}")
            print(
                f"history_len={len(agent2.memory.history)} (the restored "
                "partial turn plus the resumed run)"
            )

        print(
            f"CostBudget session_spent_usd={budget.session_spent_usd:.6f} "
            f"JobClaimedEvent count={len(claims)}"
        )
        print("demo_orchestration_layer: ok (crash/resume + CostBudget)")


if __name__ == "__main__":
    asyncio.run(main())
