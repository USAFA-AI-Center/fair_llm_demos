# demo_model_comparison.py
import asyncio
from typing import Dict
import os

"""
This module provides a tutorial on comparing the outputs of different Large
Language Models (LLMs) for the same task, showcasing the power of the framework's
Model Abstraction Layer (MAL).
"""

# --- Step 1: Import all necessary components ---
from fairlib import (
    settings,
    OpenAIAdapter,
    AnthropicAdapter,
    SimpleAgent,
    WorkingMemory,
    ReActPlanner,
    ToolRegistry,
    ToolExecutor
)
from fairlib.core.interfaces.llm import AbstractChatModel # Keep interface for type hinting

from dotenv import load_dotenv
load_dotenv()

settings.api_keys.openai_api_key = os.getenv("OPENAI_API_KEY")
settings.api_keys.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

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
    
    # Try to initialize OpenAI model
    if settings.api_keys.openai_api_key:
        openai_config = settings.models.get("openai_gpt4") # Or another configured OpenAI model
        if openai_config:
            models["OpenAI"] = OpenAIAdapter(
                api_key=settings.api_keys.openai_api_key,
                model_name=openai_config.model_name
            )
            print("✅ OpenAI model initialized.")
        else:
            print("- OpenAI model settings not found in config.")

    # Try to initialize Anthropic model
    if settings.api_keys.anthropic_api_key:
        anthropic_config = settings.models.get("anthropic_claude") # Or another configured Anthropic model
        if anthropic_config:
            # This assumes an 'AnthropicAdapter' exists in your 'mal' directory
            # similar to the OpenAI one.
            models["Anthropic"] = AnthropicAdapter(
                api_key=settings.api_keys.anthropic_api_key,
                model_name=anthropic_config.model_name
            )
            print("✅ Anthropic model initialized.")
        else:
            print("- Anthropic model settings not found in config.")
            
    if not models:
        print("\n❌ No valid models were initialized. Please check your API keys and configuration in `config/settings.yml`.")
        return

    # --- Step 4: Create an Identical Agent for Each Model ---
    print("\nCreating an agent for each initialized model...")
    role = "a creative poet"
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
