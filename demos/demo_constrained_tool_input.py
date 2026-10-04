# demo_constrained_tool_input.py
"""
Constrained decoding: the model writes the action object the schema allows.

A tool declares its input as a Pydantic model. The planner joins every
registered tool's input schema into one action schema (a thought plus one
action whose tool_name and tool_input follow a tool's own schema) and,
when the model declares the response_schema generation option, sends that
schema with the call. The provider then decodes under it: Ollama through
its format field, a Hugging Face model through an xgrammar logits
processor. The reply is one JSON object of that schema, so the tool_input
the executor validates already has the declared keys and value types.

For each of two local routes the demo runs one planner step by hand and
prints:

  - schema carried: whether the call carried the action schema, read from
    ModelRequestEvent.response_schema_digest (the adapter states it);
  - raw reply: the reply text the planner parsed (FinalAnswer.raw_text or
    ToolCallBatch.raw_text), one JSON object;
  - accepted: the executor validating that input and running the tool.

It then runs the agent to its answer. A contrast run builds the same
planner with constrained_decoding=False: the call carries no schema and
the model writes the taught key-value text shape, whose tool_input the
planner reads strictly as one JSON object (anything else is refused and
retried) and the executor validates the same way.

Requirements: an Ollama server at localhost:11434 serving FAIR_LLM_DEMO_OLLAMA
(default qwen2.5:14b), and a Hugging Face model FAIR_LLM_DEMO_MODEL (default
qwen25-7b) with the grammar extra installed (pip install "fair-llm[grammar]").

Run: PYTHONPATH=. python demos/demo_constrained_tool_input.py
"""

import asyncio
import os
import sys
from typing import List, Literal, Tuple

from pydantic import BaseModel, ConfigDict, Field

from fairlib import (
    AbstractChatModel,
    AgentEventBus,
    FairlibError,
    FinalAnswer,
    HuggingFaceAdapter,
    Message,
    ModelRequestEvent,
    OllamaAdapter,
    RoleDefinition,
    SimpleAgent,
    SimpleReActPlanner,
    ToolCallPreEvent,
    ToolExecutor,
    ToolRegistry,
    WorkingMemory,
)
from fairlib.core.interfaces.tools import (
    AbstractTool,
    SideEffect,
    TextResult,
    ToolOutput,
)

OLLAMA_MODEL = os.environ.get("FAIR_LLM_DEMO_OLLAMA", "qwen2.5:14b")
HF_MODEL = os.environ.get("FAIR_LLM_DEMO_MODEL", "qwen25-7b")

QUESTION = (
    "What is 98.6 degrees Fahrenheit in Celsius? Use the convert_temperature "
    "tool, then give the result in one sentence."
)


class ConvertTemperatureInput(BaseModel):
    """A temperature and the scale it is written in."""

    model_config = ConfigDict(extra="forbid")

    value: float = Field(description="The temperature to convert.")
    unit: Literal["celsius", "fahrenheit"] = Field(
        description="The scale the value is written in."
    )


class ConvertTemperatureTool(AbstractTool):
    name = "convert_temperature"
    description = "Convert a temperature between Celsius and Fahrenheit."
    input_schema = ConvertTemperatureInput
    output_schema = TextResult
    side_effect = SideEffect.READ_ONLY

    async def acall(self, tool_input: ConvertTemperatureInput) -> ToolOutput:
        if tool_input.unit == "fahrenheit":
            converted = (tool_input.value - 32) * 5 / 9
            return TextResult(result=f"{tool_input.value} F is {converted:.1f} C.")
        converted = tool_input.value * 9 / 5 + 32
        return TextResult(result=f"{tool_input.value} C is {converted:.1f} F.")


def build_routes() -> List[Tuple[str, AbstractChatModel]]:
    """The only place a provider is named; everything below is blind to it."""
    ollama = OllamaAdapter(
        model_name=OLLAMA_MODEL, timeout=300, options={"temperature": 0.0}
    )
    print(f"Loading {HF_MODEL} via HuggingFaceAdapter...")
    hf = HuggingFaceAdapter(HF_MODEL, max_new_tokens=256, do_sample=False)
    return [("ollama", ollama), ("huggingface", hf)]


def build_parts(
    llm: AbstractChatModel, constrained: bool
) -> Tuple[SimpleReActPlanner, ToolExecutor]:
    registry = ToolRegistry()
    registry.register_tool(ConvertTemperatureTool())
    planner = SimpleReActPlanner(llm, registry, constrained_decoding=constrained)
    planner.prompt_builder.role_definition = RoleDefinition(
        "You are a careful assistant. Use the convert_temperature tool for "
        "conversions, then give the result as your final answer."
    )
    return planner, ToolExecutor(registry)


async def one_step(llm: AbstractChatModel, constrained: bool) -> None:
    """One planner step by hand: what was sent, what came back, what ran."""
    planner, executor = build_parts(llm, constrained)
    bus = AgentEventBus()
    requests: List[ModelRequestEvent] = []
    bus.subscribe(ModelRequestEvent, requests.append)
    llm.bind_event_bus(bus)
    try:
        step = await planner.aplan(
            [Message(role="user", content=QUESTION)], QUESTION, step=0
        )
    finally:
        llm.unbind_event_bus(bus)
    carried = bool(requests) and requests[-1].response_schema_digest is not None
    print(f"  schema carried: {carried}")
    print(f"  raw reply: {step.raw_text}")
    if isinstance(step, FinalAnswer):
        print(f"  the model answered without a tool call: {step.text}")
        return
    for action in step.actions:
        try:
            observation = await executor.aexecute(action.tool_name, action.tool_input)
            print(
                f"  accepted: {action.tool_name} {action.tool_input!r} -> {observation}"
            )
        except FairlibError as exc:
            print(f"  refused: {type(exc).__name__}: {exc}")


async def full_run(llm: AbstractChatModel, constrained: bool) -> None:
    """The same planner inside the agent loop, to the answer."""
    planner, executor = build_parts(llm, constrained)
    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=executor,
        memory=WorkingMemory(),
        max_steps=4,
    )
    requests: List[ModelRequestEvent] = []
    calls: List[ToolCallPreEvent] = []
    agent.events.subscribe(ModelRequestEvent, requests.append)
    agent.events.subscribe(ToolCallPreEvent, calls.append)
    answer = await agent.arun(QUESTION)
    with_schema = sum(1 for r in requests if r.response_schema_digest is not None)
    print(f"  model calls in the run: {len(requests)}, with the schema: {with_schema}")
    for call in calls:
        print(f"  tool call: {call.tool_name} {call.tool_input!r}")
    print(f"  final answer: {answer}")


async def main() -> int:
    try:
        routes = build_routes()
    except FairlibError as exc:
        print(f"A route could not be built: {type(exc).__name__}: {exc}")
        return 1
    print(f"\nQuestion on every route: {QUESTION}")
    for name, llm in routes:
        declared = "response_schema" in llm.get_model_capabilities().generation_options
        print(f"\n=== {name}: constrained (declares response_schema: {declared}) ===")
        await one_step(llm, constrained=True)
        await full_run(llm, constrained=True)

    name, llm = routes[-1]
    print(f"\n=== {name}: contrast run, constrained_decoding=False ===")
    await one_step(llm, constrained=False)
    await full_run(llm, constrained=False)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
