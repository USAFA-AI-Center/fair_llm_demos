# demo_mcp_single_agent.py
"""
================================================================================
                    MCP Single Agent Demo
================================================================================

This demo shows how to use Model Context Protocol (MCP) to extend an agent's
capabilities with external tools. MCP allows your agent to use tools from
external servers - like filesystem access, database queries, or any custom
tool service.

In this example, we create an agent that combines:
1. Local tools (SafeCalculator) - built into fair_llm
2. MCP tools (filesystem access) - from a Python MCP server we spawn

This demonstrates the power of MCP: your agent can seamlessly use tools from
multiple sources without any changes to the agent code itself.

PREREQUISITES:
- Install the MCP package: pip install mcp

USAGE:
    python demo_mcp_single_agent.py

The agent will have access to both the calculator AND filesystem tools,
allowing it to answer questions like:
- "What is 25 * 4?" (uses local calculator)
- "List the files in the current directory" (uses MCP filesystem)
- "Read the README.md file and tell me what this project is about" (uses MCP)
"""
import asyncio
import os
import sys

from fairlib import (
    HuggingFaceAdapter,
    ToolRegistry,
    SafeCalculatorTool,
    ToolExecutor,
    WorkingMemory,
    SimpleAgent,
    SimpleReActPlanner,
    MCPServerConfig,
)

# Import MCP helper function
from fairlib.modules.mcp import create_mcp_enhanced_registry


# Path to our Python-based MCP filesystem server
MCP_SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "mcp_filesystem_server.py")


async def main():
    """
    Demonstrates a single agent with both local and MCP tools.
    """
    print("=" * 70)
    print("           MCP Single Agent Demo")
    print("=" * 70)
    print()

    # --- Step 1: Initialize the LLM ---
    # Using local HuggingFace model for testing without API keys
    print("Initializing LLM (local HuggingFace model)...")
    print("  (First run may take a moment to download the model)")
    llm = HuggingFaceAdapter("dolphin3-qwen25-3b")

    # --- Step 2: Create local tool registry with built-in tools ---
    print("\nSetting up local tools...")
    local_registry = ToolRegistry()
    local_registry.register_tool(SafeCalculatorTool())

    # --- Step 3: Configure MCP servers ---
    # We use a Python-based MCP server for filesystem access
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    mcp_configs = [
        MCPServerConfig(
            name="filesystem",
            transport="stdio",
            command=sys.executable,  # Use the current Python interpreter
            args=[MCP_SERVER_SCRIPT, project_dir],
            timeout=30,
            enabled=True
        )
    ]

    # --- Step 4: Create combined registry with local + MCP tools ---
    print("Connecting to MCP servers...")
    try:
        registry = await create_mcp_enhanced_registry(
            local_registry,
            mcp_configs=mcp_configs,
            tool_prefix="mcp"
        )
        print(f"Successfully connected! Available tools:")
        for name in registry.get_all_tools().keys():
            print(f"  - {name}")
    except Exception as e:
        print(f"Warning: Could not connect to MCP server: {e}")
        print("Falling back to local tools only...")
        print("(Install MCP: pip install mcp)")
        registry = local_registry

    # --- Step 5: Create the agent components ---
    print("\nAssembling the agent...")
    executor = ToolExecutor(registry)
    memory = WorkingMemory()
    # SimpleReActPlanner works well with local models
    planner = SimpleReActPlanner(llm, registry)

    # --- Step 6: Create the agent ---
    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=executor,
        memory=memory,
        max_steps=10
    )

    # --- Step 7: Interactive loop ---
    print("\n" + "=" * 70)
    print("Agent ready! You can now interact with it.")
    print("=" * 70)
    print("\nTry these example queries:")
    print("  - 'What is 256 * 4?'")
    print("  - 'List the files in the project directory'")
    print("  - 'Read the README.md and summarize what this project does'")
    print("\nType 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            print("\nAgent is thinking...")
            response = await agent.arun(user_input)
            print(f"\nAgent: {response}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
