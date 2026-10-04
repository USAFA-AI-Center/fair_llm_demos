# demo_named_prompt_instructions.py
"""Named prompt instructions, and worked examples checked against the planner.

A PromptBuilder's format instructions and examples can carry a name. A named
entry is replaced or removed by that name, so an application customizes one
instruction without rewriting the prompt, and the name travels with the text
through an agent config file. This demo runs a calculator agent against a
real local model through the loop an application uses:

1. Build the prompt with two named instructions and a named example, and ask
   a question.
2. Save the agent config, edit ONE named instruction in the file, and load it
   back into the running agent with load_prompts_into_agent. The other entry
   is untouched. The edited instruction changes the answer style from a
   sentence with digits to the number spelled out in words, so the same
   question gets a visibly different answer.
3. Add an example written in the wrong format for this planner. The planner
   refuses it with a typed ConfigurationError when it prepares the prompt,
   naming the example, instead of letting the model imitate a shape the
   parser rejects on every call.
4. Remove the bad example by name and ask again.

Set FAIR_LLM_DEMO_MODEL to choose the local model.
"""

import asyncio
import atexit
import json
import os
import shutil
import tempfile
from pathlib import Path

from fairlib import (
    HuggingFaceAdapter,
    PromptBuilder,
    ReActPlanner,
    RoleDefinition,
    SafeCalculatorTool,
    SimpleAgent,
    ToolExecutor,
    ToolRegistry,
    WorkingMemory,
    load_prompts_into_agent,
    save_agent_config,
)
from fairlib.core.errors import ConfigurationError

MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "qwen25-7b")
# Every file the demo writes lands in a scratch directory removed at exit.
SCRATCH_DIR = Path(tempfile.mkdtemp(prefix="fair_named_prompts_demo_"))
atexit.register(shutil.rmtree, SCRATCH_DIR, ignore_errors=True)

# A worked example in the JSON shape ReActPlanner parses: two model turns,
# with the observation between them.
ADD_EXAMPLE = (
    "User: What is 15 plus 27?\n"
    '{"thought": "I need to add 15 and 27 with the calculator.", '
    '"action": {"tool_name": "safe_calculator", '
    '"tool_input": {"expression": "15 + 27"}}}\n'
    "Observation: 42\n"
    '{"thought": "The calculator returned 42, so I can answer.", '
    '"action": {"tool_name": "final_answer", '
    '"tool_input": {"text": "15 plus 27 is 42."}}}'
)

# The same idea in SimpleReActPlanner's key-value shape, its input as bare
# text: a shape this planner's parser refuses.
WRONG_FORMAT_EXAMPLE = (
    "User: What is 2 plus 2?\n"
    "Thought: I should add them.\n"
    "Action:\n"
    "tool_name: safe_calculator\n"
    "tool_input: 2 + 2"
)


def build_prompt() -> PromptBuilder:
    """The calculator prompt, with every instruction and example named."""
    builder = PromptBuilder()
    builder.role_definition = RoleDefinition(
        "You are a calculator assistant. Use the safe_calculator tool for "
        "every computation - never compute mentally."
    )
    builder.set_format_instruction(
        "method", "Call safe_calculator once per computation, then answer."
    )
    builder.set_format_instruction(
        "answer_style",
        "State the result in one short sentence that restates the question, "
        "such as: 15 plus 27 is 42.",
    )
    builder.set_example("add", ADD_EXAMPLE)
    return builder


def build_agent(llm) -> SimpleAgent:
    """A calculator agent whose planner starts from build_prompt()."""
    registry = ToolRegistry()
    registry.register_tool(SafeCalculatorTool())
    planner = ReActPlanner(llm, registry, prompt_builder=build_prompt())
    return SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=ToolExecutor(registry),
        memory=WorkingMemory(),
        max_steps=5,
    )


def show_instructions(agent: SimpleAgent) -> None:
    """Print the builder's format instructions with their names."""
    for instruction in agent.planner.prompt_builder.format_instructions:
        print(f"  [{instruction.name}] {instruction.text}")


async def ask(agent: SimpleAgent, question: str) -> None:
    """Ask one question on a fresh conversation and print the answer."""
    agent.memory.clear()
    answer = await agent.arun(question)
    print(f"  Q: {question}\n  A: {answer}")


async def main() -> None:
    print("=" * 70)
    print("FAIR-LLM: Named prompt instructions and example validation")
    print("=" * 70)
    llm = HuggingFaceAdapter(MODEL_NAME)
    agent = build_agent(llm)

    print("\n1. The prompt's named instructions")
    show_instructions(agent)
    await ask(agent, "What is 12 times 7?")

    print("\n2. Override one instruction by name, through the config file")
    path = SCRATCH_DIR / "calculator.json"
    save_agent_config(agent, str(path))
    config = json.loads(path.read_text(encoding="utf-8"))
    print(f"  config version: {config['version']}")
    for entry in config["prompts"]["content"]["format_instructions"]:
        if isinstance(entry, dict) and entry["name"] == "answer_style":
            entry["text"] = (
                "Write the final answer as the result spelled out in English "
                "words only, with no digits, such as: forty-two."
            )
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    load_prompts_into_agent(str(path), agent)
    print("  edited only 'answer_style' in the file and loaded it back:")
    show_instructions(agent)
    await ask(agent, "What is 12 times 7?")

    print("\n3. An example written for another planner's format")
    agent.planner.prompt_builder.set_example("wrong_format", WRONG_FORMAT_EXAMPLE)
    agent.planner.invalidate_prompt_cache()
    try:
        await ask(agent, "What is 9 plus 10?")
    except ConfigurationError as exc:
        print(f"  refused when the prompt was prepared: {exc}")

    print("\n4. Remove it by name and ask again")
    agent.planner.prompt_builder.remove_example("wrong_format")
    agent.planner.invalidate_prompt_cache()
    await ask(agent, "What is 9 plus 10?")


if __name__ == "__main__":
    asyncio.run(main())
