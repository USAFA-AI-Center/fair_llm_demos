# demo_multi_agent.py
"""
Multi-agent collaboration on the workers-as-tools path: a manager model
delegating to a research and analysis team.

A real local model (loaded through the HuggingFaceAdapter) plays the
manager. The manager is not a special orchestrator: it is a plain
SimpleAgent built by build_worker_manager, whose tools are worker agents
wrapped in WorkerAgentTool. The manager model learns what each worker is
for from the tool descriptions in its rendered tool catalog, delegates
subtasks as ordinary typed tool calls, and synthesizes the final answer
itself.

The scenario: a request that needs both real-time information gathering
(web research) and mathematical computation (analysis). No single worker
can solve it alone; the manager splits the work and combines the results.

What it shows:
  - WorkerAgentTool adapts any BaseAgent into a typed tool; the tool
    description is what the manager model reads to choose a worker.
  - build_worker_manager is pure wiring: a batch-capable planner plus a
    ToolExecutor over a registry of worker tools. Delegation is an
    ordinary tool-call turn, so the side-effect-aware executor applies.
  - Each worker declares its own SideEffect. The analyst only computes,
    so it is READ_ONLY: independent READ_ONLY delegations in one turn
    run concurrently. The researcher keeps the conservative EXTERNAL
    default because its web search reaches the network, so it runs as a
    sequential barrier.
  - Workers are ordinary stateless SimpleAgents with their own planners
    and tools, the same agents you would build standalone.

Note that this particular query is dependent (find the price, then divide
the budget by it), so a sensible manager delegates in sequence here; the
concurrency win shows up on queries whose subtasks are independent.

Requirements: a local HuggingFace model (transformers) plus Google CSE
credentials in settings.yml for the web search tool. The manager and
workers share one loaded model, and a real model's delegation choices are
stochastic; re-run or use a stronger instruct model if a run wanders.
"""
import asyncio

from fairlib import (
    HuggingFaceAdapter,
    ReActPlanner,
    SafeCalculatorTool,
    SimpleAgent,
    ToolExecutor,
    ToolRegistry,
    WebSearcherTool,
    WorkerAgentTool,
    WorkingMemory,
    build_worker_manager,
    settings,
)
from fairlib.core.interfaces.tools import SideEffect


def create_worker(llm, tools):
    """Build an ordinary stateless worker agent around its own tools.

    This is the same construction a standalone agent uses; nothing about a
    worker is manager-specific until WorkerAgentTool wraps it. What each
    worker is for is stated in the WorkerAgentTool description, which the
    manager model reads from its rendered tool catalog.
    """
    tool_registry = ToolRegistry()
    for tool in tools:
        tool_registry.register_tool(tool)

    planner = ReActPlanner(llm, tool_registry)
    executor = ToolExecutor(tool_registry)
    memory = WorkingMemory()

    # Stateless: each delegation is planned fresh, not against the
    # accumulated history of earlier delegations.
    return SimpleAgent(llm, planner, executor, memory, stateless=True)


async def main():
    """Set up and run the multi-agent team."""
    # The web search tool needs Google CSE credentials; without them the
    # researcher cannot do its job, so bail out early.
    if not settings.search_engine.google_cse_search_api or not settings.search_engine.google_cse_search_engine_id:
        print("A google search engine API key as well as search engine ID needs to be set to run this demo. Exiting...")
        return

    # --- Step 1: Initialize the shared model ---
    print("Initializing fairlib.core.components...")
    llm = HuggingFaceAdapter("dolphin3-qwen25-3b")

    # --- Step 2: Create Specialized Worker Agents ---
    # Each worker is a standard ReAct agent with a limited set of tools.
    print("Building the agent team...")

    web_search_config = {
        "google_api_key": settings.search_engine.google_cse_search_api,
        "google_search_engine_id": settings.search_engine.google_cse_search_engine_id,
        "cache_ttl": settings.search_engine.web_search_cache_ttl,
        "cache_max_size": settings.search_engine.web_search_cache_max_size,
        "max_results": settings.search_engine.web_search_max_results,
    }

    researcher = create_worker(llm, [WebSearcherTool(config=web_search_config)])
    analyst = create_worker(llm, [SafeCalculatorTool()])

    # --- Step 3: Wrap each worker as a typed tool ---
    # The description is what the manager model sees; it replaces the old
    # role_description roster. The researcher keeps the conservative
    # EXTERNAL default because its web search reaches the network, so its
    # delegations run as sequential barriers. The analyst only computes,
    # so READ_ONLY is the author's assertion that lets independent analyst
    # delegations in one turn run concurrently.
    worker_tools = [
        WorkerAgentTool(
            researcher,
            name="researcher",
            description=(
                "Delegate a research subtask, phrased as a complete question, "
                "to an agent that uses a web search tool to find current, "
                "real-time information like prices, news, and facts."
            ),
        ),
        WorkerAgentTool(
            analyst,
            name="analyst",
            description=(
                "Delegate a math subtask, phrased as a complete question, "
                "to an analyst agent that performs calculations with a safe "
                "calculator."
            ),
            side_effect=SideEffect.READ_ONLY,
        ),
    ]

    # --- Step 4: Build the Manager ---
    # A plain SimpleAgent over the worker tools; the model sees the
    # workers through the rendered tool catalog and decides for itself
    # what to delegate and when to answer.
    manager = build_worker_manager(llm, worker_tools)

    # --- Step 5: Define a Complex User Query ---
    # Unsolvable by any single worker: the researcher finds the price and
    # the analyst performs the calculation.
    user_query = "My budget is $5,000. Please find the current price of Bitcoin and then calculate exactly how many Bitcoins I can afford to buy."

    # --- Step 6: Run the Agent Team ---
    final_answer = await manager.arun(user_query)

    # --- Step 7: Display the Final Result ---
    print("\n--- FINAL Synthesized Answer ---")
    print(final_answer)


if __name__ == "__main__":
    asyncio.run(main())
