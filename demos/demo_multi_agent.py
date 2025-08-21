# multi_agent_demo.py
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

settings.api_keys.openai_api_key = os.getenv("OPENAI_API_KEY")
settings.api_keys.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# TODO:: this kind of function should be a utility available to all demo files
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


async def main():
    """
    The main function to set up and run the multi-agent system.
    """
    # --- Step 2: Initialize Core Components ---
    # We load our settings using the validated Pydantic configuration loader.
    # This ensures all necessary API keys and model settings are present.
    print("Initializing fairlib.core.components...")
    llm = OpenAIAdapter(
        api_key=settings.api_keys.openai_api_key,
        model_name=settings.models["openai_gpt4"].model_name
    )

    # --- Step 3: Create Specialized Worker Agents ---
    # Here, we build our team of specialists. Each worker is a standard
    # ReAct agent but is given a very specific role and a limited set of tools.

    print("Building the agent team...")
    
    # The Researcher: Its only tool is the WebSearcher. Its role is clearly defined.
    researcher = create_agent(
        llm,
        [WebSearcherTool()],
        "A research agent that uses a web search tool to find current, real-time information like prices, news, and facts."
    )

    # The Analyst: Its only tool is the SafeCalculator. It's designed for math.
    analyst = create_agent(
        llm,
        [SafeCalculatorTool()],
        "An analyst agent that performs mathematical calculations using a safe calculator."
    )

    # We organize the workers in a dictionary so the manager can find them by name.
    workers = {"Researcher": researcher, "Analyst": analyst}

    # --- Step 4: Create the Manager Agent ---
    # The manager is a special type of agent. It doesn't have regular tools.
    # Instead, its "tool" is the ability to delegate tasks to its workers.
    # We equip it with the special `ManagerPlanner`.
    manager_memory = WorkingMemory()
    manager_planner = ManagerPlanner(llm, workers)
    
    # Note: The manager's ToolExecutor is `None` because it should never execute
    # a tool directly. Its planner will only produce 'delegate' or 'final_answer' actions.
    manager_agent = SimpleAgent(llm, manager_planner, None, manager_memory) 

    # --- Step 5: Initialize the Hierarchical Runner ---
    # The runner is the orchestrator that connects the manager to the workers
    # and manages the overall flow of the conversation.
    team_runner = HierarchicalAgentRunner(manager_agent, workers)
    
    # --- Step 6: Define a Complex User Query ---
    # This query is designed to be unsolvable by any single worker.
    # It requires the Researcher to find the price and the Analyst to perform the calculation.
    user_query = "My budget is $5,000. Please find the current price of Bitcoin and then calculate exactly how many Bitcoins I can afford to buy."
    
    # --- Step 7: Run the Agent Team ---
    # We call the `arun` method on the runner, which kicks off the entire
    # collaborative process. The runner will print the internal thoughts
    # and actions of the agents as it works.
    final_answer = await team_runner.arun(user_query)
    
    # --- Step 8: Display the Final Result ---
    print("\n✅ --- FINAL Synthesized Answer ---")
    print(final_answer)


if __name__ == "__main__":
    # Run the asynchronous main function.
    asyncio.run(main())
