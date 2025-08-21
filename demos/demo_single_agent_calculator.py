# single_agent_calculator_demo.py
import asyncio
import yaml
from pathlib import Path

"""
This script serves as a foundational tutorial for the FAIR-LLM framework.

It demonstrates how to assemble and run a single, autonomous agent. This is the
"Hello, World!" of the framework, showing the essential components and their
interactions in the simplest possible configuration.

We will build an agent that has one tool: a calculator. We will then interact
with it in a simple loop. For a more advanced example demonstrating multi-agent
collaboration, see `demo_advanced_calculator_calculus.py`.
"""

# --- Step 1: Import the necessary framework components ---
# We import everything needed to build a single agent from the ground up.
from fairlib import (
    HuggingFaceAdapter,
    ToolRegistry,
    SafeCalculatorTool,
    ToolExecutor,
    WorkingMemory,
    ReActPlanner,
    SimpleAgent, 
    SimpleReActPlanner,
    RoleDefinition
)

from fairlib.modules.mal import huggingface_adapter

async def main():
    """
    The main function to assemble and run our single agent.
    """
    print("Initializing a single agent for demonstration...")

    # --- Step 2: Assemble the Agent's "Anatomy" ---
    # An agent is composed of several key parts, which we instantiate here.

    # a) The "Brain": The Language Model
    # We use our settings file to configure the LLM adapter.

    #llm = OpenAIAdapter(
    #    api_key=settings.api_keys.openai_api_key,
    #    model_name=settings.models["openai_gpt4"].model_name
    #)

    llm = HuggingFaceAdapter("dolphin3-qwen25-3b")

    # b) The "Toolbelt": The Tool Registry and Tools
    # The registry holds all the tools the agent can use.
    tool_registry = ToolRegistry()
    
    # We create an instance of our safe calculator and register it.
    calculator_tool = SafeCalculatorTool()
    tool_registry.register_tool(calculator_tool)
    
    print(f"Agent's tools: {[tool.name for tool in tool_registry.get_all_tools().values()]}")

    # c) The "Hands": The Tool Executor
    # This component is responsible for actually running the tool that the agent decides to use.
    executor = ToolExecutor(tool_registry)

    # d) The "Memory": The Agent's Short-Term Memory
    # We use WorkingMemory to keep track of the current conversation.
    memory = WorkingMemory()

    # e) The "Mind": The Planner
    # The ReActPlanner is responsible for the agent's reasoning process. It takes
    # the user's request and the conversation history and decides what to do next.
    
    # For use with the more complex lanuage models
    #planner = ReActPlanner(llm, tool_registry)
    
    # For use with simple, local models
    planner = SimpleReActPlanner(llm, tool_registry)

    # modify the default role a bit:
    planner.prompt_builder.role_definition = \
    RoleDefinition(
        "You are an expert mathematical calculator. Your job it is to perform mathematical calculations.\n"
        "You reason step-by-step to determine the best course of action. If a user's request requires "
        "multiple steps or tools, you must break it down and execute them sequentially. You must follow the strict formatting rules that follow..."
    )

    # --- Step 3: Create the Agent ---
    # Now we assemble all the pieces into a complete agent. The SimpleAgent
    # class ties everything together into a functional unit.
    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=executor,
        memory=memory,
        max_steps=10  # We give it a limit to prevent it from running forever.
    )
    print("Agent successfully created. You can now chat with the agent.")
    print("Try asking it a math problem, like 'What is 45 * 11?' or 'What is the result of 1024 divided by 256?'. Type 'exit' to quit.")

    # --- Step 4: Run the Interaction Loop ---
    # This loop allows you to have a continuous conversation with the agent.
    while True:
        try:
            user_input = input("\n👤 You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("🤖 Agent: Goodbye!")
                break
            
            # This is the main call. The agent takes the input and runs its
            # entire Reason-Act loop to come up with a response.
            agent_response = await agent.arun(user_input)
            print(f"LLM Raw Output:\n{agent_response}")
            print(f"🤖 Agent: {agent_response}")

        except KeyboardInterrupt:
            print("\n🤖 Agent: Exiting...")
            break


if __name__ == "__main__":
    asyncio.run(main())
