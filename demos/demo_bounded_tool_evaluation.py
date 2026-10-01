"""Demo: a tool runs bounded by declaration, and the model reacts to a tripped bound.

The adopter's path is three lines. A BasicSecurityManager is built with
compute_bounds=ComputeBounds(cpu_seconds=8, wall_seconds=10) and an event bus,
and handed to the ToolExecutor. AdvancedCalculusTool is registered as it is
shipped: it declares isolation = IsolationLevel.BOUNDED and takes no sandbox
arguments. From then on every dispatch of it runs in a bounded child, where
fairlib rebuilds the tool from its class, runs it, and returns its output.
Question 1 is an integral SymPy finishes at once: the child completes and the
agent answers under the bounds. Question 2 is an integral SymPy cannot finish
in that time: the child is killed at its bound, the tool boundary reports it
as a typed ResourceExhaustedError whose observation names the bound and says
the same call reaches it again, and the model reacts, with a different action
or a final answer that says the computation did not finish. Each run's
BoundedRunEvent and each dispatch's ToolCallPostEvent (with the tool_input
the model sent) print in one stream from the agent's bus, along with any
ToolInputRepairedEvent the planner raises when it reads the model's input
into the tool's field, so the outcome is read from typed events, not
inferred.

BOUNDED is a resource bound, not isolation: the child runs as the same user
with the same reach, so it is for fairlib's own compute, never model-written
code.

Requires the local extra (pip install 'fair-llm[local]') and a local model;
defaults to HuggingFaceAdapter("qwen25-14b"). Set FAIR_LLM_DEMO_MODEL to
override. The 14B is the default because question 2 needs a model that writes
a long nested command intact and then reports the tripped bound honestly: the
7B tends to misplace a quote in integral(1/(x**200 + 1), x, 0, 1), so the
child never reaches the hard integral, and then states a value it never
computed.
"""

from __future__ import annotations

import asyncio
import os

from fairlib import (
    AgentEventBus,
    BasicSecurityManager,
    BoundedRunEvent,
    ComputeBounds,
    FairlibError,
    HuggingFaceAdapter,
    LoopGuardTrippedEvent,
    MaxStepsExceeded,
    RoleDefinition,
    SimpleAgent,
    SimpleReActPlanner,
    ToolCallPostEvent,
    ToolExecutor,
    ToolInputRepairedEvent,
    ToolInvocationError,
    ToolRegistry,
    WorkingMemory,
)
from fairlib.modules.action.tools.advanced_calculus_tool import AdvancedCalculusTool

MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "qwen25-14b")

# The manager's opinion on the bounds. A child starts in well under a
# second, so 10 s of wall clock is generous for an integral SymPy can do and
# far too little for one it cannot; the CPU bound lands near the same time,
# so under load either may be the bound that shows.
BOUNDS = ComputeBounds(wall_seconds=10, cpu_seconds=8)

# A role definition, set the way the calculus demo sets its own. The tool's
# description shows whole commands such as integral(sin(x), x), each written
# into its one expression field; without this line a model can read such a
# command as a tool name, so the tool is never dispatched.
ROLE = RoleDefinition(
    "You are a symbolic mathematics assistant. You compute every derivative "
    "and integral with the calculus tool listed under Available Tools, never "
    "by hand. In the Action block, tool_name is that tool's listed name, "
    "exactly as listed; the calculus command itself, in the form the tool's "
    "description shows, is the tool_input. Every number you report comes "
    "from a tool observation: when the tool returns an error, correct the "
    "call and try again, and when it cannot finish, say that no value was "
    "computed rather than estimating one."
)

QUESTIONS = (
    "Compute the integral of sin(x) with respect to x.",
    "Compute the definite integral of 1/(x**200 + 1) from 0 to 1. "
    "If the tool cannot finish, say so and explain what you would try instead.",
)


def _describe(bounds: ComputeBounds | None) -> str:
    if bounds is None:
        return "none (in-process)"
    return (
        f"cpu_seconds={bounds.cpu_seconds} memory_bytes={bounds.memory_bytes} "
        f"wall_seconds={bounds.wall_seconds} output_bytes={bounds.output_bytes} "
        f"file_bytes={bounds.file_bytes} max_tasks={bounds.max_tasks}"
    )


def on_bounded_run(event: BoundedRunEvent) -> None:
    bound = event.bound.value if event.bound is not None else "none"
    achieved = event.achieved.value if event.achieved is not None else "none"
    print(
        f"[BoundedRunEvent] tool_name={event.tool_name} "
        f"outcome={event.outcome.value} bound={bound} "
        f"required={event.required.value} achieved={achieved} "
        f"bounds=({_describe(event.bounds)}) "
        f"duration={event.duration_seconds:.2f}s"
    )


def on_tool_call_post(event: ToolCallPostEvent) -> None:
    kind = "none"
    if isinstance(event.error, ToolInvocationError):
        kind = event.error.kind.value
    print(
        f"[ToolCallPostEvent] tool_name={event.tool_name} "
        f"tool_input={event.tool_input!r} "
        f"succeeded={event.succeeded} error_kind={kind} observation:\n"
        f"{event.observation}"
    )


def on_input_read(event: ToolInputRepairedEvent) -> None:
    print(
        f"[ToolInputRepairedEvent] tool_name={event.tool_name} "
        f"original={event.original_text!r} read_as={event.repaired_text}"
    )


def on_loop_guard(event: LoopGuardTrippedEvent) -> None:
    print(
        f"[LoopGuardTrippedEvent] guard={event.guard_type.value} "
        f"count={event.count} step={event.step}"
    )


async def main() -> None:
    print(f"--- bounded tool evaluation on {MODEL_NAME} ---")

    bus = AgentEventBus()
    # The adopter's path: a manager with bounds on the executor, the tool
    # registered as shipped. The manager decides at every dispatch that the
    # tool runs bounded (it declares IsolationLevel.BOUNDED), runs the child
    # and emits BoundedRunEvent on the bus it holds; the agent shares that
    # bus, so both event types print in one stream.
    manager = BasicSecurityManager(events=bus, compute_bounds=BOUNDS)
    tool = AdvancedCalculusTool()
    registry = ToolRegistry()
    registry.register_tool(tool)
    executor = ToolExecutor(registry, manager)
    print(
        f"tool declares:    isolation={tool.isolation.value if tool.isolation else None}"
    )
    print(f"manager bounds:   {_describe(BOUNDS)}")
    dispatch = manager.compute_bounds_for(tool)
    print(f"effective bounds: {_describe(dispatch.bounds if dispatch else None)}")
    print(f"required level:   {dispatch.required.value if dispatch else 'none'}")
    print("(0 = unlimited; effective = the manager's bounds clamped to limits.compute)")

    llm = HuggingFaceAdapter(MODEL_NAME, max_new_tokens=512)
    planner = SimpleReActPlanner(llm, registry)
    planner.prompt_builder.role_definition = ROLE
    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=executor,
        memory=WorkingMemory(),
        max_steps=6,
        events=bus,
    )
    agent.events.subscribe(BoundedRunEvent, on_bounded_run)
    agent.events.subscribe(ToolCallPostEvent, on_tool_call_post)
    agent.events.subscribe(ToolInputRepairedEvent, on_input_read)
    agent.events.subscribe(LoopGuardTrippedEvent, on_loop_guard)

    for number, question in enumerate(QUESTIONS, start=1):
        print(f"\n=== Question {number}: {question}")
        try:
            answer = await agent.arun(question)
            print(f"final answer {number}: {answer}")
        except MaxStepsExceeded as exc:
            print(f"outcome {number}: MaxStepsExceeded: {exc}")
        except FairlibError as exc:
            print(f"outcome {number}: {type(exc).__name__}: {exc}")

    print("\nbounded tool evaluation demo complete.")


if __name__ == "__main__":
    asyncio.run(main())
