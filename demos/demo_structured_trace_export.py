# demo_structured_trace_export.py

"""
Structured trace export demonstration.

When an agent finishes a run, the caller normally receives only the final
answer string. That makes it hard to explain why the agent chose a tool or
what happened when something failed. Structured trace export records an
append-only stream of framework events and groups them by agent step.

You will see a real agent loop (local HuggingFace model plus calculator tool),
a call to BaseAgent.arun_with_trace (same behavior as arun, but it also returns
an AgentRunTrace), and the trace event types and per-step groupings printed
for inspection. Any agent with an event bus can use arun_with_trace.

Run: PYTHONPATH=. python demos/demo_structured_trace_export.py
Requires a GPU and local HuggingFace weights. Set FAIR_LLM_DEMO_MODEL to
override the default model.
"""

import asyncio
import os
from pprint import pprint

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

    print("\n--- Trace summary ---")
    print("Final output:", trace.output)
    print("Run status:", trace.status)
    print("Recorded event types:", [event.event_type for event in trace.events])
    print("\nGrouped steps (causal inspection):")
    pprint(trace.to_dict()["steps"])


if __name__ == "__main__":
    asyncio.run(main())
