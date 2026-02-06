# demo_agent_config_export_load.py
"""
This demo shows how to save a FAIR-LLM agent's complete configuration
to JSON and load it back as a fully functional agent. This enables:

- Saving agent setups for reuse
- Sharing configurations with teammates
- Preparing agents for prompt optimization with fair_prompt_optimizer
- Version controlling agent configurations
"""
import asyncio
import logging
from pathlib import Path

from fairlib import (
    HuggingFaceAdapter,
    ToolRegistry,
    SafeCalculatorTool,
    ToolExecutor,
    WorkingMemory,
    SimpleReActPlanner,
    SimpleAgent,
    RoleDefinition,
    FormatInstruction,
    Example,
)

# Import utility fucntions from the config_manager
from fairlib.utils.config_manager import (
    save_agent_config,
    load_agent,
    load_agent_config,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_NAME = "dolphin3-qwen25-3b"
OUTPUT_DIR = Path("outputs")


def build_calculator_agent(llm) -> SimpleAgent:
    """
    Build a calculator agent with complete prompt configuration.
    """
    print("\n" + "=" * 60)
    print("  BUILDING AGENT")
    print("=" * 60)
    
    tool_registry = ToolRegistry()
    tool_registry.register_tool(SafeCalculatorTool())
    
    planner = SimpleReActPlanner(llm, tool_registry)
    
    planner.prompt_builder.role_definition = RoleDefinition(
        "You are a helpful calculator assistant. Your job is to solve "
        "mathematical problems accurately using the calculator tool. "
        "Always use the tool for calculations - never compute mentally. "
        "Provide clear, concise answers."
    )
    
    planner.prompt_builder.format_instructions.append(
        FormatInstruction(
            "When solving math problems:\n"
            "1. Identify the mathematical operation needed\n"
            "2. Use the safe_calculator tool with the expression\n"
            "3. Report the exact numerical result"
        )
    )
    
    planner.prompt_builder.examples.append(
        Example(
            "User: What is 15 plus 27?\n"
            "Thought: I need to add 15 and 27. I'll use the calculator.\n"
            "Action: {\"tool_name\": \"safe_calculator\", \"tool_input\": \"15 + 27\"}\n"
            "Observation: 42\n"
            "Thought: The calculator returned 42.\n"
            "Action: {\"tool_name\": \"final_answer\", \"tool_input\": \"42\"}"
        )
    )
    
    planner.prompt_builder.examples.append(
        Example(
            "User: Calculate 8 times 9\n"
            "Thought: I need to multiply 8 by 9.\n"
            "Action: {\"tool_name\": \"safe_calculator\", \"tool_input\": \"8 * 9\"}\n"
            "Observation: 72\n"
            "Thought: The result is 72.\n"
            "Action: {\"tool_name\": \"final_answer\", \"tool_input\": \"72\"}"
        )
    )

    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=ToolExecutor(tool_registry),
        memory=WorkingMemory(),
        max_steps=5
    )

    return agent


async def test_agent(agent: SimpleAgent, label: str):
    """Run a quick test on the agent."""
    
    print(f"\n{'─' * 60}")
    print(f"  Testing: {label}")
    print('─' * 60)
    
    test_queries = [
        "What is 25 times 4?",
        "Calculate 150 divided by 6",
    ]
    
    for query in test_queries:
        print(f"\n👤 Query: {query}")
        try:
            agent.memory.clear()
            response = await agent.arun(query)
            print(f"🤖 Response: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")


async def main():
    print("=" * 70)
    print("   FAIR-LLM: Agent Configuration Save/Load Demo")
    print("=" * 70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config_path = OUTPUT_DIR / "calculator_agent.json"
    
    print("\n" + "=" * 60)
    print("  INITIALIZING LLM")
    print("=" * 60)
    
    llm = HuggingFaceAdapter(MODEL_NAME)

    original_agent = build_calculator_agent(llm)
    await test_agent(original_agent, "Original Agent")
    
    print("\n" + "=" * 60)
    print("  SAVING AGENT CONFIGURATION")
    print("=" * 60)
    
    config = save_agent_config(original_agent, str(config_path))
    
    print(f"\n📄 Saved to: {config_path}")
    print("\n📋 Configuration contents:")
    print(f"   • Role: {config['role_definition'][:50]}...")
    print(f"   • Tools: {config['agent']['tools']}")
    print(f"   • Examples: {len(config['examples'])}")
    print(f"   • Max steps: {config['agent']['max_steps']}")
    print(f"   • Model: {config['model']['model_name']}")
    
    print("\n" + "=" * 60)
    print("  LOADING AGENT FROM CONFIGURATION")
    print("=" * 60)
    
    loaded_agent = load_agent(str(config_path), llm)
    await test_agent(loaded_agent, "Loaded Agent")
    
    print("\n" + "=" * 60)
    print("  INSPECTING CONFIGURATION")
    print("=" * 60)
    
    config_dict = load_agent_config(str(config_path))
    
    print(f"\nMetadata:")
    print(f"   Optimized: {config_dict['metadata']['optimized']}")
    print(f"   Exported at: {config_dict['metadata']['exported_at']}")
    print(f"   Source: {config_dict['metadata']['source']}")

if __name__ == "__main__":
    asyncio.run(main())