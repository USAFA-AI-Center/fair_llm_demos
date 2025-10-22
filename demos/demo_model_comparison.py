# demo_model_comparison.py
import asyncio
from typing import Dict

"""
This module provides a tutorial on comparing the outputs of different Large
Language Models (LLMs) for the same task, showcasing the power of the framework's
Model Abstraction Layer (MAL).
"""

# --- Step 1: Import all necessary components ---
from fairlib import (
    HuggingFaceAdapter,
    SimpleAgent,
    WorkingMemory,
    ReActPlanner,
    ToolRegistry,
    ToolExecutor
)

from fairlib.core.interfaces.llm import AbstractChatModel # Keep interface for type hinting

# --- Step 2: Create a simple factory to build agents ---
# This helps keep our code clean when creating multiple identical agents.
def create_comparison_agent(llm: AbstractChatModel, role_description: str) -> SimpleAgent:
    """Creates a basic agent with no tools for text generation comparison."""
    # An agent with no tools will rely entirely on its LLM for responses.
    tool_registry = ToolRegistry()
    executor = ToolExecutor(tool_registry)
    memory = WorkingMemory()
    # Even with no tools, the ReActPlanner effectively prompts the LLM to give a direct answer.
    planner = ReActPlanner(llm, tool_registry)
    
    agent = SimpleAgent(llm, planner, executor, memory)
    agent.role_description = role_description
    return agent


async def main():
    """The main function to set up and run the model comparison."""
    
    # --- Step 3: Dynamically Initialize LLMs from Settings ---
    # This section demonstrates the plug-and-play nature of the MAL.
    # We will try to initialize every model the user has configured
    # in their `settings.yml` file.
    print("Initializing configured models from settings...")
    
    models: Dict[str, AbstractChatModel] = {}
    
    # initialize models for comparison
    models["dolphin3-qwen25-3b"] = HuggingFaceAdapter("dolphin3-qwen25-3b")
    models["dolphin3-qwen25-0.5b"] = HuggingFaceAdapter("dolphin3-qwen25-0.5b")

    if not models:
        print("\n❌ No valid models were initialized. Please check your API keys and configuration in `config/settings.yml`.")
        return

    # --- Step 4: Create an Identical Agent for Each Model ---
    print("\nCreating an agent for each initialized model...")
    # descriptive role to give to each agent
    role = """You are a creative poet who responds directly with your poetry.

            CRITICAL INSTRUCTIONS:
            - You do NOT have access to any tools, functions, or external systems
            - You must respond ONLY with plain text - never attempt to call functions or use tools
            - Write your poems directly in your response
            - Do NOT use JSON, function calls, or tool invocations of any kind
            - Simply write the poem as natural text in your reply

            When asked to write a poem, immediately begin writing the poem itself. 
            Do not describe what tools you would use or mention any functions. Just write the poem."""
    
    agents = {
        name: create_comparison_agent(model, role) for name, model in models.items()
    }

    # --- Step 5: Define a Subjective Prompt ---
    # A creative task is best for seeing differences in model "personality".
    prompt = "Write a short, four-line poem about a lighthouse in a storm."
    print(f"\n--- Giving all agents the same prompt: ---\n'{prompt}'\n")

    # --- Step 6: Run All Agents in Parallel ---
    # `asyncio.gather` is a an efficient way to run multiple async tasks concurrently.
    tasks = [agent.arun(prompt) for agent in agents.values()]
    responses = await asyncio.gather(*tasks)
    
    results = dict(zip(agents.keys(), responses))

    # --- Step 7: Display the Side-by-Side Comparison ---
    print("--- 📊 Model Comparison Results ---")
    for model_name, response in results.items():
        print("\n=====================================")
        print(f"   Model: {model_name}")
        print("=====================================")
        print(response)
        print("-------------------------------------")

if __name__ == "__main__":
    # To get the most out of this demo, ensure you have API keys for
    # both OpenAI and Anthropic in your `config/settings.yml` file.
    asyncio.run(main())
