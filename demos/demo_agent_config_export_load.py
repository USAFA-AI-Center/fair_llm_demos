# demo_agent_config_export_load.py
"""
This demo shows how to save a FAIR-LLM agent's complete configuration
to JSON and load it back as a fully functional agent. This enables:

- Saving agent setups for reuse
- Sharing configurations with teammates
- Preparing agents for prompt optimization with fair_prompt_optimizer
- Version controlling agent configurations

Prompt content is written in Python: build_calculator_prompts constructs a
PromptBuilder and sets its fields, and the builder is handed to the planner
at construction (ReActPlanner(llm, registry, prompt_builder=...)) -
the recommended shape for applications. The prompt-store JSON format
appears where serialization earns its place: exporting configs, and the
save -> edit -> hot-swap loop below.

It ends with the two live-swap surfaces, both mid-session with no agent
reconstruction:

- Prompt swap: save the running planner's prompts to a prompt-store file,
  edit the file, and hot-swap it back through the planner.prompt_builder
  setter. The same mechanism behind every load-new-prompts flow (optimized
  configs from fair_prompt_optimizer, A/B prompt variants, per-model
  overlays).
- Toolset swap: replace the whole registry through the planner.tool_registry
  setter, paired with a matching executor. The same move an application
  makes to gate toolsets per user or session phase, and the one an MCP
  re-discovery makes when it rebuilds the registry.
"""
import asyncio
import json
import logging
from pathlib import Path

from fairlib import (
    HuggingFaceAdapter,
    ToolRegistry,
    SafeCalculatorTool,
    ToolExecutor,
    WeatherTool,
    WorkingMemory,
    ReActPlanner,
    SimpleAgent,
    RoleDefinition,
    FormatInstruction,
    Example,
)

# The agent factory: build agents from config files, save them back
from fairlib.modules.agent.factory import (
    save_agent_config,
    load_agent,
    load_agent_config,
)

# The prompt store: PromptBuilder content as a JSON file.
from fairlib.core.prompts import PromptBuilder, load_builder, save_builder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_NAME = "dolphin3-qwen25-3b"
OUTPUT_DIR = Path("outputs")


def build_calculator_prompts() -> PromptBuilder:
    """Create the calculator prompt content: a fresh PromptBuilder, every
    field set in Python. This is the builder the planner is constructed
    with; the planner merges its mandatory format rules on top at build
    time, so the parser contract holds whatever this content says.
    """
    builder = PromptBuilder()

    builder.role_definition = RoleDefinition(
        "You are a helpful calculator assistant. Your job is to solve "
        "mathematical problems accurately using the calculator tool. "
        "Always use the tool for calculations - never compute mentally. "
        "Provide clear, concise answers."
    )

    builder.format_instructions.append(
        FormatInstruction(
            "When solving math problems:\n"
            "1. Identify the mathematical operation needed\n"
            "2. Use the safe_calculator tool with the expression\n"
            "3. Report the exact numerical result"
        )
    )

    builder.examples.append(
        Example(
            "User: What is 15 plus 27?\n"
            "{\"thought\": \"I need to add 15 and 27 with the calculator.\", "
            "\"action\": {\"tool_name\": \"safe_calculator\", \"tool_input\": \"15 + 27\"}}\n"
            "Observation: 42\n"
            "{\"thought\": \"The calculator returned 42, so I can answer.\", "
            "\"action\": {\"tool_name\": \"final_answer\", \"tool_input\": \"42\"}}"
        )
    )

    builder.examples.append(
        Example(
            "User: Calculate 8 times 9\n"
            "{\"thought\": \"I need to multiply 8 by 9.\", "
            "\"action\": {\"tool_name\": \"safe_calculator\", \"tool_input\": \"8 * 9\"}}\n"
            "Observation: 72\n"
            "{\"thought\": \"The result is 72, so I can answer.\", "
            "\"action\": {\"tool_name\": \"final_answer\", \"tool_input\": \"72\"}}"
        )
    )

    return builder


def build_calculator_agent(llm, prompt_builder: PromptBuilder) -> SimpleAgent:
    """
    Build a calculator agent around the given prompt content.

    The builder is injected at planner construction - the recommended shape
    for applications.
    """
    print("\n" + "=" * 60)
    print("BUILDING AGENT")
    print("=" * 60)

    tool_registry = ToolRegistry()
    tool_registry.register_tool(SafeCalculatorTool())

    planner = ReActPlanner(llm, tool_registry, prompt_builder=prompt_builder)

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
    print(f"Testing: {label}")
    print('─' * 60)
    
    test_queries = [
        "What is 25 times 4?",
        "Calculate 150 divided by 6",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            agent.memory.clear()
            response = await agent.arun(query)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")


async def main():
    print("=" * 70)
    print("FAIR-LLM: Agent Configuration Save/Load Demo")
    print("=" * 70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config_path = OUTPUT_DIR / "calculator_agent.json"
    
    print("\n" + "=" * 60)
    print("INITIALIZING LLM")
    print("=" * 60)
    
    llm = HuggingFaceAdapter(MODEL_NAME)

    original_agent = build_calculator_agent(llm, build_calculator_prompts())
    await test_agent(original_agent, "Original Agent")
    
    print("\n" + "=" * 60)
    print("SAVING AGENT CONFIGURATION")
    print("=" * 60)
    
    config = save_agent_config(original_agent, str(config_path))
    
    print(f"\nSaved to: {config_path}")
    print("\nConfiguration contents:")
    print(f"• Role: {config['prompts']['role_definition'][:50]}...")
    print(f"• Tools: {config['agent']['tools']}")
    print(f"• Examples: {len(config['prompts']['examples'])}")
    print(f"• Max steps: {config['agent']['max_steps']}")
    print(f"• Model: {config['model']['model_name']}")
    
    print("\n" + "=" * 60)
    print("LOADING AGENT FROM CONFIGURATION")
    print("=" * 60)
    
    loaded_agent = load_agent(str(config_path), llm)
    await test_agent(loaded_agent, "Loaded Agent")
    
    print("\n" + "=" * 60)
    print("INSPECTING CONFIGURATION")
    print("=" * 60)
    
    config_dict = load_agent_config(str(config_path))

    print("\nMetadata:")
    print(f"Exported at: {config_dict['metadata']['exported_at']}")
    print(f"Source: {config_dict['metadata']['source']}")

    await demonstrate_live_prompt_swap(loaded_agent)
    await demonstrate_live_registry_swap(loaded_agent)


async def demonstrate_live_prompt_swap(agent: SimpleAgent):
    """Save the live prompts to a file, edit the file, hot-swap it back.

    The planner.prompt_builder SETTER is the supported surface for runtime
    prompt changes: assignment replaces the content and invalidates the
    planner's prepared-prompt cache, so the very next plan call renders from
    the new prompts. The parser keeps working regardless - the mandatory
    format rules are merged on every build and cannot be customized away.
    """
    print("\n" + "=" * 60)
    print("LIVE PROMPT SWAP (prompt store)")
    print("=" * 60)

    prompts_path = OUTPUT_DIR / "calculator_prompts.json"

    # 1. Save the running planner's prompt content to a prompt-store file.
    save_builder(agent.planner.prompt_builder, prompts_path)
    print(f"\nPrompts saved to: {prompts_path}")

    # 2. Edit the file - the same edit a user makes by hand in an editor.
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    prompts["role_definition"] = (
        "You are a terse calculator assistant. Use the calculator tool for "
        "every computation and answer with the bare number only - no "
        "sentences, no punctuation, just the numeric result."
    )
    prompts_path.write_text(
        json.dumps(prompts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Edited role_definition in the file (terse persona).")

    # 3. Hot-swap: assignment through the setter picks the file back up and
    #    invalidates the prepared-prompt cache. Same agent, same memory, new
    #    prompt on the next call.
    agent.planner.prompt_builder = load_builder(prompts_path)
    print("Swapped into the running agent via planner.prompt_builder = ...")

    await test_agent(agent, "After live prompt swap (terse persona)")


async def demonstrate_live_registry_swap(agent: SimpleAgent):
    """Swap the agent's whole toolset at runtime through tool_registry.

    The planner.tool_registry SETTER is the surface for changing what the
    agent can DO, mirroring the prompt_builder setter for what it SAYS:
    assignment invalidates the prepared-prompt cache, so the very next plan
    call renders the new tool catalog. An application uses this to gate
    toolsets per user, per environment, or per session phase; an MCP
    re-discovery makes the same move when it rebuilds the registry.

    The catalog and dispatch must agree: the planner renders what the model
    may call, the executor dispatches what actually runs. Swapping a
    registry therefore pairs the planner assignment with an executor built
    on the same registry object.
    """
    print("\n" + "=" * 60)
    print("LIVE TOOLSET SWAP (tool_registry setter)")
    print("=" * 60)

    catalog_before = agent.planner.render_system_prompt()
    print(f"\n'weather' in catalog before swap: {'weather' in catalog_before}")

    registry_v2 = ToolRegistry()
    registry_v2.register_tool(SafeCalculatorTool())
    registry_v2.register_tool(WeatherTool())

    agent.planner.tool_registry = registry_v2
    agent.tool_executor = ToolExecutor(registry_v2)
    print("Swapped in a calculator+weather registry via planner.tool_registry = ...")

    catalog_after = agent.planner.render_system_prompt()
    print(f"'weather' in catalog after swap:  {'weather' in catalog_after}")

    agent.memory.clear()
    response = await agent.arun("What is the weather in Denver?")
    print("\nQuery: What is the weather in Denver?")
    print(f"Response: {response}")


if __name__ == "__main__":
    asyncio.run(main())