# demo_structured_trace_export.py

"""
Structured trace export demonstration.

When an agent finishes a run, the caller normally receives only the final
answer string. That makes it hard to explain why the agent chose a tool or
what happened when something failed. Structured trace export records an
append-only stream of framework events and groups them by agent step.

You will see a real agent loop (local HuggingFace model plus calculator tool),
a call to BaseAgent.arun_with_trace (same behavior as arun, but it also returns
an AgentRunTrace), the recorded event stream, and then each agent step with
the events it grouped under it: the request the model received (its role
order), the model call that produced the step (with its token usage and stop
reason), the tool call and its observation. The
trace is also saved as JSON, which is the artifact a failed run is
reproduced from. Any agent with an event bus can use arun_with_trace.

Run: PYTHONPATH=. python demos/demo_structured_trace_export.py
Requires a GPU and local HuggingFace weights. Set FAIR_LLM_DEMO_MODEL to
override the default model.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any

from fairlib import (
    HuggingFaceAdapter,
    RoleDefinition,
    SafeCalculatorTool,
    SimpleAgent,
    SimpleReActPlanner,
    ToolExecutor,
    ToolRegistry,
    WorkingMemory,
)

MODEL_NAME = os.getenv("FAIR_LLM_DEMO_MODEL", "qwen25-7b")


def describe(event_type: str, payload: dict[str, Any]) -> str:
    """One readable line for a trace record, from the fields its event carries."""
    if event_type == "ModelInvocationEvent":
        usage = payload.get("usage") or {}
        return (
            f"model={payload.get('model_name')} outcome={payload.get('outcome')} "
            f"{payload.get('duration_ms', 0):.0f}ms "
            f"prompt_tokens={usage.get('prompt_tokens')} "
            f"completion_tokens={usage.get('completion_tokens')} "
            f"done_reason={usage.get('done_reason')}"
        )
    if event_type == "ModelRequestEvent":
        roles = ", ".join(payload.get("roles") or ())
        return f"roles as the provider received them: {roles}"
    if event_type == "AgentStepEvent":
        return (
            f"step {payload.get('step')} of max {payload.get('max_steps')}, "
            f"history_length={payload.get('history_length')}"
        )
    if event_type == "ToolCallPreEvent":
        return f"{payload.get('tool_name')} <- {payload.get('tool_input')!r}"
    if event_type == "ToolCallPostEvent":
        return (
            f"{payload.get('tool_name')} succeeded={payload.get('succeeded')} "
            f"->\n{payload.get('observation', '')}"
        )
    if event_type == "ToolBatchScheduledEvent":
        return (
            f"batch_size={payload.get('batch_size')} "
            f"max_parallel_tools={payload.get('max_parallel_tools')}"
        )
    keys = ", ".join(sorted(payload))
    return f"fields: {keys}"


async def main() -> None:
    print(f"Loading {MODEL_NAME} via HuggingFaceAdapter...")
    llm = HuggingFaceAdapter(MODEL_NAME, max_new_tokens=256)

    # --- Assemble a minimal calculator agent (same pieces as demo_single_agent_calculator) ---
    registry = ToolRegistry()
    registry.register_tool(SafeCalculatorTool())
    executor = ToolExecutor(registry)
    planner = SimpleReActPlanner(llm, registry)
    planner.prompt_builder.role_definition = RoleDefinition(
        "You are a helpful calculator assistant. Use the safe_calculator tool for "
        "arithmetic, then give a concise final answer."
    )
    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=executor,
        memory=WorkingMemory(),
        max_steps=6,
    )

    # --- Run with trace export enabled ---
    # TraceRecorder subscribes to the agent's event bus for the duration of the
    # run. On success or failure the finished trace is stored on agent.last_trace.
    question = "What is 18 + 27?"
    print(f"\nUser: {question}")
    trace = await agent.arun_with_trace(
        question,
        trace_metadata={"demo": "structured-trace-export"},
    )

    # to_dict is the trace contract every AbstractAgentRunTrace honours, so
    # the demo reads the trace through it rather than through one class.
    data = trace.to_dict()
    print("\n--- Trace summary ---")
    print("Final output:", data["output"])
    print("Run status:", data["status"])
    print(f"Recorded {len(data['events'])} events, in order:")
    for record in data["events"]:
        print(f"  #{record['sequence']:<3} {record['event_type']}")

    def show(record: dict[str, Any], indent: str) -> None:
        detail = describe(record["event_type"], record["payload"])
        print(f"{indent}#{record['sequence']:<3} {record['event_type']:<20} {detail}")

    print("\nGrouped steps (causal inspection):")
    placed = len(data["before_first_step"])
    if data["before_first_step"]:
        print("  Before the first step:")
        for record in data["before_first_step"]:
            show(record, "    ")
    for step in data["steps"]:
        print(f"  Step {step['step']}:")
        for record in step["events"]:
            placed += 1
            show(record, "    ")
    print(f"{placed} of {len(data['events'])} recorded events placed in the view.")

    with tempfile.TemporaryDirectory() as tmp:
        saved = trace.save(Path(tmp) / "run_trace.json")
        print(f"\nSaved the full trace as JSON: {saved.stat().st_size} bytes.")


if __name__ == "__main__":
    asyncio.run(main())
