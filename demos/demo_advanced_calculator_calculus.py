# demo_advanced_calculator_calculus.py
"""
This script demonstrates how to assemble a single intelligent agent that can reason,
respond, and use tools in a mathematically rich environment.

The agent supports:
    1. Basic arithmetic calculations using SafeCalculatorTool
    2. Symbolic calculus operations (derivatives and integrals) using AdvancedCalculusTool

This serves as a practical tutorial for combining multiple tools under the FAIR-LLM framework.
"""

import asyncio

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

# --- Step 2: Import the additiojnal tools we want this agent to use ---
# NOTE: SafeCalculatorTool is a built-in tool while AdvancedCalculusTool
# is a tool we built to extend beyond our basic built-in tools.
from fairlib.modules.action.tools.advanced_calculus_tool import AdvancedCalculusTool
from fairlib.utils.math_expression_parser import parse_math_expression


async def main():
    """
    Main entry point for the demo agent.
    This sets up the brain, memory, planner, tools, and interaction loop.
    """
    print("Initializing the Advanced Calculator + Calculus Agent...")

    # === (a) Brain: Language Model ===
    # Uses dolphin3-qwen25-3b for reasoning and decision making
    llm = HuggingFaceAdapter("dolphin3-qwen25-3b")

    # === (b) Toolbelt: Register both calculator and calculus tools ===
    tool_registry = ToolRegistry()

    calculator_tool = SafeCalculatorTool()
    calculus_tool = AdvancedCalculusTool()

    # Register tools with the registry
    tool_registry.register_tool(calculator_tool)
    tool_registry.register_tool(calculus_tool)

    print(
        f"Registered tools: {[tool.name for tool in tool_registry.get_all_tools().values()]}"
    )

    # === (c) Hands: Tool Executor ===
    executor = ToolExecutor(tool_registry)

    # === (d) Memory: Conversation Context ===
    memory = WorkingMemory()

    # === (e) Mind: Reasoning Engine ===
    # planner = ReActPlanner(llm, tool_registry)
    # For use with simple, local models

    planner = SimpleReActPlanner(llm, tool_registry)

    # modify the default role a bit:
    planner.prompt_builder.role_definition = RoleDefinition(
        "You are an advanced expert mathematical calculator whose job it is to perform calculations.\n"
        "You must reason step-by-step to determine the best course of action. If a user's request requires "
        "multiple steps or tools, you must break it down and execute them sequentially."
    )

    # === (f) Assemble the Agent ===
    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=executor,
        memory=memory,
        max_steps=10,  # Limit reasoning loops to prevent runaway execution
    )

    print("Agent is ready to work.")
    print("You can enter either plain math commands or symbolic math expressions.")
    print("Examples of supported queries:")
    print("  - 'What is (50 + 25) / 5?'                (basic arithmetic)")
    print("  - 'derivative(x**3 + sin(x), x)'          (functional form)")
    print("  - '∫(x**3 + sin(x)) dx'                   (symbolic indefinite integral)")
    print("  - 'integral(1/(1 + x**2), x, 0, 1)'       (definite integral, functional)")
    print("  - '∫0 to 1 (1 / (1 + x**2)) dx'           (symbolic definite integral)")
    print("  - 'd/dx x**2 + sin(x)'                    (symbolic derivative)")
    print("\nType 'exit' or 'quit' to end the session.")

    # === (g) Interaction Loop ===
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Agent: Goodbye!")
                break

            # Run the agent's full Reason+Act cycle
            parsed_input = parse_math_expression(user_input)
            agent_response = await agent.arun(parsed_input)
            print(f"Agent: {agent_response}")

        except (EOFError, KeyboardInterrupt):
            # End of piped input or Ctrl-C: the session is over.
            print("\nAgent: Session ended.")
            break
        except Exception as e:
            print(f"Agent error: {e}")


# Entrypoint for script execution
if __name__ == "__main__":
    asyncio.run(main())
