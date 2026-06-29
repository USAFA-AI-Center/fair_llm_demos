# demo_checkpoint_resume.py

"""
Checkpoint and resume demonstration.

Agents keep conversation history in memory, but until now there was no standard
way to bookmark a position and rewind. AgentCheckpoint stores only a history
length (not message copies). Rewinding truncates memory.history back to that
position.

You will see a calculator and weather agent driven by a local HuggingFace model.
Turn 1 completes and save_state writes a bookmark to disk. Turn 2 grows the
history. load_state rewinds to the bookmark. Turn 3 continues from the
rewound position with a new question.

Run: PYTHONPATH=. python demos/demo_checkpoint_resume.py
Requires a GPU. Set FAIR_LLM_DEMO_MODEL to override the default model.
"""

import asyncio
import os
import tempfile
from pathlib import Path

from fairlib import (
    HuggingFaceAdapter,
    RoleDefinition,
    SafeCalculatorTool,
    SimpleAgent,
    SimpleReActPlanner,
    ToolExecutor,
    ToolRegistry,
    WeatherTool,
    WorkingMemory,
)

MODEL_NAME = os.getenv("FAIR_LLM_DEMO_MODEL", "qwen25-7b")


def print_history(agent) -> None:
    print("Message count:", len(agent.memory.history))
    for message in agent.memory.history:
        preview = message.content.replace("\n", " ")[:80]
        print(f"  [{message.role}] {preview}")


async def main() -> None:
    print(f"Loading {MODEL_NAME} via HuggingFaceAdapter...")
    llm = HuggingFaceAdapter(MODEL_NAME, max_new_tokens=256)

    registry = ToolRegistry()
    registry.register_tool(SafeCalculatorTool())
    registry.register_tool(WeatherTool())
    executor = ToolExecutor(registry)
    planner = SimpleReActPlanner(llm, registry)
    planner.prompt_builder.role_definition = RoleDefinition(
        "You are a helpful assistant with a calculator and a weather tool. "
        "Use the right tool for each request, then answer concisely."
    )
    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=executor,
        memory=WorkingMemory(),
        max_steps=8,
    )

    # --- Turn 1: establish a baseline conversation ---
    print("\n--- Turn 1: calculator ---")
    first = await agent.arun("Calculate 10 + 5.")
    print("Agent:", first)
    print_history(agent)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "agent-checkpoint.json"

        # --- Bookmark history position after turn 1 ---
        checkpoint = agent.save_state(path, metadata={"after": "first-turn"})
        print("\nSaved checkpoint history_length:", checkpoint.history_length)

        # --- Turn 2: history grows beyond the bookmark ---
        print("\n--- Turn 2: weather (history grows) ---")
        second = await agent.arun("What is the weather in Denver?")
        print("Agent:", second)
        print_history(agent)

        # --- Rewind: truncate history back to the bookmark ---
        print("\n--- Rewind via load_state() ---")
        restored = agent.load_state(path)
        print("Restored checkpoint:", restored.to_dict())
        print_history(agent)

        # --- Turn 3: continue from rewound position ---
        print("\n--- Turn 3: new question after rewind ---")
        third = await agent.arun("Now calculate 20 / 4.")
        print("Agent:", third)
        print_history(agent)


if __name__ == "__main__":
    asyncio.run(main())
