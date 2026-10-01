# demo_model_comparison.py
"""
Compare several models on the same task through the Model Abstraction Layer.

Every model here is reached through AbstractChatModel, so the agent code is
identical for all of them: one factory builds the same tool-free ReAct agent
around each model, every agent gets the same prompt, and the answers print
side by side. Only the line that constructs each adapter names a provider.

The comparison always runs two local instruct models of different sizes
(Qwen2.5 7B and 14B through the HuggingFaceAdapter). When GEMINI_API_KEY is
exported, a hosted Gemini model joins as a third column, which shows the same
agent running unchanged across providers.

Run:
    PYTHONPATH=. python demos/demo_model_comparison.py
Requires a GPU for the local models; GEMINI_API_KEY is optional.
"""

import asyncio
import os
import time
from typing import Dict

from fairlib import (
    GeminiAdapter,
    HuggingFaceAdapter,
    ReActPlanner,
    RoleDefinition,
    SimpleAgent,
    ToolExecutor,
    ToolRegistry,
    WorkingMemory,
)
from fairlib.core.errors import FairlibError
from fairlib.core.interfaces.llm import AbstractChatModel

LOCAL_MODELS = ("qwen25-7b", "qwen25-14b")
GEMINI_MODEL = os.environ.get("FAIR_LLM_DEMO_GEMINI", "gemini-3.6-flash")


def create_comparison_agent(
    llm: AbstractChatModel, role_description: str
) -> SimpleAgent:
    """Build a tool-free agent; only the model differs between agents."""
    tool_registry = ToolRegistry()
    executor = ToolExecutor(tool_registry)
    memory = WorkingMemory()
    # With no tools the ReActPlanner's instructions still ask for a
    # final_answer turn, so the answer comes back through the normal loop.
    planner = ReActPlanner(llm, tool_registry)

    # The role reaches the model through the planner's prompt builder, the
    # seam every planner renders its system prompt from.
    planner.prompt_builder.role_definition = RoleDefinition(role_description)
    return SimpleAgent(llm, planner, executor, memory)


async def timed_run(agent: SimpleAgent, prompt: str) -> tuple[object, float]:
    """Run one agent and return its answer (or typed failure) and seconds taken."""
    started = time.perf_counter()
    try:
        answer: object = await agent.arun(prompt)
    except FairlibError as exc:
        # A typed failure is itself a comparison result worth displaying.
        answer = exc
    return answer, time.perf_counter() - started


async def main() -> None:
    """Build one agent per model, give them one prompt, print the answers."""
    print("Initializing models...")
    models: Dict[str, AbstractChatModel] = {}
    for alias in LOCAL_MODELS:
        print(f"  local  {alias} (HuggingFaceAdapter)")
        models[alias] = HuggingFaceAdapter(alias, max_new_tokens=512)
    if os.environ.get("GEMINI_API_KEY"):
        print(f"  hosted {GEMINI_MODEL} (GeminiAdapter)")
        models[GEMINI_MODEL] = GeminiAdapter(model_name=GEMINI_MODEL, timeout=60)
    else:
        print("  GEMINI_API_KEY is not set; the hosted column is skipped.")

    # The role sets personality only. The planner owns the response format:
    # its rendered instructions already tell the model to deliver the poem
    # through a final_answer turn.
    role = (
        "You are a creative poet. You have no tools; when asked for a poem, "
        "deliver the finished poem itself as your final answer."
    )
    agents = {
        name: create_comparison_agent(model, role) for name, model in models.items()
    }

    prompt = "Write a short, four-line poem about a lighthouse in a storm."
    print(f"\nSame prompt for every agent:\n  {prompt}\n")

    # All agents run concurrently; each one's failure stays its own result.
    runs = await asyncio.gather(*(timed_run(a, prompt) for a in agents.values()))

    print("--- Model comparison ---")
    for name, (answer, seconds) in zip(agents, runs):
        print(f"\n=== {name} ({type(models[name]).__name__}, {seconds:.1f}s) ===")
        if isinstance(answer, FairlibError):
            print(f"FAILED ({type(answer).__name__}): {answer}")
        else:
            # The poem is printed exactly as the model wrote it. A stray
            # backslash or a literal \n at a line end is the model's own
            # escaping inside its answer, not something the framework added.
            print(answer)


if __name__ == "__main__":
    asyncio.run(main())
