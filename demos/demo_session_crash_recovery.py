# demo_session_crash_recovery.py

"""
Per-session registry and crash recovery demonstration.

Checkpoints rewind in-process history, but a real crash wipes memory entirely.
JsonSessionStore persists messages and checkpoint metadata to disk.
SessionRegistry tracks live agents per session id and can hydrate a fresh
agent after a restart.

You will register a live calculator agent under a session id, run a turn, then
persist the session while it is still marked running. recover_interrupted_sessions
simulates a crash by marking stale runs. A fresh agent is built and recover
loads message history from disk. The demo continues the conversation on the
recovered agent.

Run: PYTHONPATH=. python demos/demo_session_crash_recovery.py
Requires a GPU. Set FAIR_LLM_DEMO_MODEL to override the default model.
"""

import asyncio
import os
import tempfile

from fairlib import (
    HuggingFaceAdapter,
    JsonSessionStore,
    RoleDefinition,
    SafeCalculatorTool,
    SessionRegistry,
    SessionStatus,
    SimpleAgent,
    SimpleReActPlanner,
    ToolExecutor,
    ToolRegistry,
    WorkingMemory,
)

MODEL_NAME = os.getenv("FAIR_LLM_DEMO_MODEL", "qwen25-7b")


def build_calculator_agent() -> SimpleAgent:
    """Assemble a small calculator agent used for both live and recovered runs."""
    llm = HuggingFaceAdapter(MODEL_NAME, max_new_tokens=256)
    registry = ToolRegistry()
    registry.register_tool(SafeCalculatorTool())
    executor = ToolExecutor(registry)
    planner = SimpleReActPlanner(llm, registry)
    planner.prompt_builder.role_definition = RoleDefinition(
        "You are a helpful calculator assistant. Use safe_calculator for math."
    )
    return SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=executor,
        memory=WorkingMemory(),
        max_steps=6,
    )


async def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        store = JsonSessionStore(directory)
        registry = SessionRegistry(store)

        # --- Live session: register agent and run one turn ---
        print(f"Loading {MODEL_NAME} for the live session...")
        live_agent = build_calculator_agent()
        registry.register("classroom-demo", live_agent)
        print("\n--- Live turn ---")
        live_result = await live_agent.arun("Calculate 7 * 6.")
        print("Live result:", live_result)

        # Persist while still marked running — simulates a crash mid-session.
        registry.persist(
            "classroom-demo",
            metadata={"course": "math142", "stage": "pre-crash"},
            status=SessionStatus.RUNNING,
        )
        interrupted = store.recover_interrupted_sessions()
        print("\nInterrupted sessions after restart:", [r.session_id for r in interrupted])

        # --- Recovery: hydrate a brand-new agent from disk ---
        print(f"\nLoading {MODEL_NAME} for the recovered session...")
        recovered_agent = build_calculator_agent()
        registry.recover("classroom-demo", recovered_agent)
        print("Recovered message count:", len(recovered_agent.memory.history))

        print("\n--- Continued turn on recovered agent ---")
        recovered_result = await recovered_agent.arun("Continue with 9 + 1.")
        print("Recovered result:", recovered_result)

        registry.unregister("classroom-demo")
        print("Final stored status:", store.load("classroom-demo").status.value)


if __name__ == "__main__":
    asyncio.run(main())
