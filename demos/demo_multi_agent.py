import asyncio
import os

"""
This script serves as a hands-on tutorial and demonstration of the framework's
most advanced feature: Hierarchical Multi-Agent Collaboration.

We will construct a "team" of AI agents with a clear reporting structure:
1.  A "Manager" Agent: Its job is to understand a complex user request and
    delegate sub-tasks to the appropriate worker.
2.  "Worker" Agents: Each worker has a specialized role and a specific tool,
    allowing it to excel at one type of task.

The scenario: A user wants to perform a task that requires both real-time
information gathering (research) and mathematical computation (analysis). No single
agent can solve this alone, but by collaborating, the team can deliver a
comprehensive solution.

🎓 EDUCATIONAL NOTE:
This demo automatically detects if Google API credentials are available.
- Without credentials: Uses MockWebSearcherTool
- With credentials: Uses real WebSearcherTool for live data
To set up real search, see: google_api_setup.md
"""

# --- Step 1: Import all necessary components ---
from fairlib import (
    settings,
    OpenAIAdapter,
    ToolRegistry,
    SafeCalculatorTool,
    WebSearcherTool,
    ToolExecutor,
    WorkingMemory,
    ReActPlanner,
    SimpleAgent,
    ManagerPlanner,
    HierarchicalAgentRunner
)

from demo_tools.mock_web_searcher import MockWebSearcherTool

from dotenv import load_dotenv
load_dotenv()

# LOAD API KEYS AND SETTNGS FROM ENV VARS
settings.api_keys.openai_api_key = os.getenv("OPENAI_API_KEY")
settings.api_keys.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
settings.search_engine.google_cse_search_api = os.getenv("GOOGLE_CSE_SEARCH_API")
settings.search_engine.google_cse_search_engine_id = os.getenv("GOOGLE_CSE_SEARCH_ENGINE_ID")

def create_agent(llm, tools, role_description):
    """
    A helper factory function to simplify the creation of worker agents.
    Each agent gets its own tool registry, planner, executor, and memory.
    """
    tool_registry = ToolRegistry()
    for tool in tools:
        tool_registry.register_tool(tool)
    
    planner = ReActPlanner(llm, tool_registry)
    executor = ToolExecutor(tool_registry)
    memory = WorkingMemory()
    
    # create a stateless agent
    agent = SimpleAgent(llm, planner, executor, memory, stateless=True)

    # This custom attribute helps the manager understand the worker's purpose.
    agent.role_description = role_description
    return agent


def get_web_searcher_tool(cse_search_api, cse_engine_id):
    if cse_search_api and cse_engine_id:
        web_search_config = {
            "google_api_key": cse_search_api,
            "google_search_engine_id": cse_engine_id,
            "cache_ttl": settings.search_engine.web_search_cache_ttl,
            "cache_max_size": settings.search_engine.web_search_cache_max_size,
            "max_results": settings.search_engine.web_search_max_results,
        }
        return WebSearcherTool(config=web_search_config)
    else:
        return MockWebSearcherTool()

async def main():
    """
    The main function to set up and run the multi-agent system.
    """
    
    print("=" * 60)
    print("🤖 Multi-Agent Collaboration Demo")
    print("=" * 60)
    
    # --- Step 2: Initialize Core Components ---
    print("\n📚 Initializing fairlib.core.components...")
    llm = OpenAIAdapter(
        api_key=settings.api_keys.openai_api_key,
        model_name=settings.models["openai_gpt4"].model_name
    )

    # --- Step 3: Create Specialized Worker Agents ---
    print("👥 Building the agent team...")
    
    # The get_web_searcher_tool function automatically chooses the right implementation
    search_tool = get_web_searcher_tool(settings.search_engine.google_cse_search_api, settings.search_engine.google_cse_search_engine_id)
    
    # The Researcher: Its only tool is the WebSearcher
    researcher = create_agent(
        llm, 
        [search_tool],
        "A research agent that uses a web search tool to find current, real-time information like prices, news, and facts."
    )
    print("   ✓ Researcher agent created")

    # The Analyst: Its only tool is the SafeCalculator
    analyst = create_agent(
        llm,
        [SafeCalculatorTool()],
        "An analyst agent that performs mathematical calculations using a safe calculator."
    )
    print("   ✓ Analyst agent created")

    # We organize the workers in a dictionary so the manager can find them by name.
    workers = {"Researcher": researcher, "Analyst": analyst}

    # --- Step 4: Create the Manager Agent ---
    manager_memory = WorkingMemory()
    manager_planner = ManagerPlanner(llm, workers)
    manager_agent = SimpleAgent(llm, manager_planner, None, manager_memory)
    print("   ✓ Manager agent created")

    # --- Step 5: Initialize the Hierarchical Runner ---
    team_runner = HierarchicalAgentRunner(manager_agent, workers)
    print("\n🚀 Agent team ready!\n")
    
    # --- Step 6: Define a Complex User Query ---
    print("=" * 60)
    print("📋 USER QUERY:")
    user_query = "My budget is $5,000. Please find the current price of Bitcoin and then calculate exactly how many Bitcoins I can afford to buy."
    print(f"   '{user_query}'")
    print("=" * 60)
    
    # --- Step 7: Run the Agent Team ---
    print("\n🔄 Starting multi-agent collaboration...\n")
    print("-" * 40)
    
    final_answer = await team_runner.arun(user_query)
    
    # --- Step 8: Display the Final Result ---
    print("-" * 40)
    print("\n✅ FINAL SYNTHESIZED ANSWER:")
    print("=" * 60)
    print(final_answer)
    print("=" * 60)


if __name__ == "__main__":
    # Run the asynchronous main function.
    asyncio.run(main())