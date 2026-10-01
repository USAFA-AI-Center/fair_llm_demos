# demo_tool_conformance.py
"""
Run the tool conformance suite over a tool you wrote, then put that tool in
front of a real model.

A tool is the unit a capstone extends fairlib with, and the conformance
suite is the green/red proof that it honors the AbstractTool contract: the
declared attributes, the catalog entry rendered from its input schema, a
typed output, failures on the typed channel, a side-effect class that
matches what the tool does, a body that cooperates with the event loop, and
an observation the executor can bound. The suite never raises; it returns a
report you read.

The demo builds a small unit-conversion tool the way a student would and
declares one ToolConformanceCase for it that sets every field of the case:
the sample input and the fragment its observation must contain, a failing
input the tool must refuse on the typed channel, a watched directory the
READ_ONLY tool must leave untouched, a reset hook run before every
behavioral call, a wall-clock bound per call, a label for the report
header, and the security manager the tool's dispatch meets (the same
BasicSecurityManager the agent's executor holds below). It prints the
report, then runs the same suite over a broken twin that returns an error
string instead of raising ToolInvocationError, so the failure a report
shows is a real one. Finally the conforming tool is registered with a
SimpleAgent on a local model and asked a question that needs it, and the
tool's own call log shows the agent used it.

Set FAIR_LLM_DEMO_MODEL to pick the local model (a settings.yml alias or a
Hugging Face model id).
"""

import asyncio
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from fairlib import (
    AbstractTool,
    BasicSecurityManager,
    HuggingFaceAdapter,
    RoleDefinition,
    SideEffect,
    SimpleAgent,
    SimpleReActPlanner,
    TextResult,
    ToolConformanceCase,
    ToolExecutor,
    ToolInvocationError,
    ToolOutput,
    ToolRegistry,
    WorkingMemory,
    check_tool_conformance,
)

MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "qwen25-7b")

_FACTORS_TO_METERS = {
    "m": 1.0,
    "km": 1000.0,
    "mi": 1609.344,
    "ft": 0.3048,
    "nmi": 1852.0,
}

# A tiny usage log the tool appends to on every call. The case's reset hook
# clears it before each behavioral call, the way a mutating tool's hook
# would put its sample's target back; after the agent run it shows the
# conversions the agent actually asked for.
_CONVERSION_LOG: list = []

# The security manager the tool's dispatch meets, in the suite and in the
# agent alike. The tool declares no isolation level, so it runs in-process
# with this manager bound, and the executor screens each input through it.
SECURITY_MANAGER = BasicSecurityManager()


def reset_conversion_log() -> None:
    _CONVERSION_LOG.clear()


class ConvertInput(BaseModel):
    """A length to convert between units."""

    value: float = Field(description="The quantity to convert.")
    from_unit: str = Field(description="The unit of value: m, km, mi, ft, or nmi.")
    to_unit: str = Field(description="The unit to convert into: m, km, mi, ft, or nmi.")


class UnitConverterTool(AbstractTool):
    """Convert a length between metric, imperial, and nautical units."""

    name = "convert_length"
    description = "Convert a length between m, km, mi, ft, and nmi."
    input_schema = ConvertInput
    output_schema = TextResult
    side_effect = SideEffect.READ_ONLY

    async def acall(self, tool_input: ConvertInput) -> ToolOutput:
        try:
            meters = tool_input.value * _FACTORS_TO_METERS[tool_input.from_unit]
            converted = meters / _FACTORS_TO_METERS[tool_input.to_unit]
        except KeyError as exc:
            raise ToolInvocationError(
                f"Unknown unit {exc.args[0]!r}; use one of m, km, mi, ft, nmi.",
                tool_name=self.name,
            ) from exc
        _CONVERSION_LOG.append(f"{tool_input.from_unit}->{tool_input.to_unit}")
        return TextResult(
            result=(
                f"{tool_input.value} {tool_input.from_unit} is {converted:.4f} "
                f"{tool_input.to_unit} ({meters:.1f} m; factors from the SI table)"
            )
        )


class ErrorStringConverterTool(UnitConverterTool):
    """The same tool with the classic mistake: an error string instead of a typed error."""

    name = "convert_length_broken"

    async def acall(self, tool_input: ConvertInput) -> ToolOutput:
        if tool_input.from_unit not in _FACTORS_TO_METERS:
            return TextResult(result=f"Error: unknown unit {tool_input.from_unit}")
        return await super().acall(tool_input)


def run_suite(tool: AbstractTool, label: str) -> None:
    # Every field of ToolConformanceCase, set: the sample and the fragment
    # its observation must contain, a failing input the tool must refuse on
    # the typed channel, a directory this READ_ONLY tool must leave
    # untouched, the reset hook, a wall-clock bound on each call, the
    # label the report header carries, and the security manager the
    # tool's dispatch meets.
    with tempfile.TemporaryDirectory(prefix="conformance_watched_") as watched:
        case = ToolConformanceCase(
            tool,
            sample_input=ConvertInput(value=26.2, from_unit="mi", to_unit="km"),
            expected_fragment="42.1648 km",
            failing_input=ConvertInput(value=1, from_unit="furlong", to_unit="km"),
            watched_root=Path(watched),
            reset=reset_conversion_log,
            timeout_seconds=5.0,
            label=label,
            security_manager=SECURITY_MANAGER,
        )
        report = check_tool_conformance(case)
    print(report.render())
    if report.skipped:
        print(
            "(skipped means the suite could not observe that property on this "
            "sample; a skipped line is never a pass)"
        )
    print()


async def ask_the_agent() -> None:
    llm = HuggingFaceAdapter(MODEL_NAME)
    registry = ToolRegistry()
    registry.register_tool(UnitConverterTool())
    planner = SimpleReActPlanner(llm, registry)
    planner.prompt_builder.role_definition = RoleDefinition(
        "You are a navigation assistant. You cannot convert units yourself: for "
        "any unit conversion you MUST call the convert_length tool first, then "
        "give the figure it returned as your final answer."
    )
    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=ToolExecutor(registry, security_manager=SECURITY_MANAGER),
        memory=WorkingMemory(),
        max_steps=6,
    )
    question = "How many kilometers is a 26.2 mile marathon?"
    print(f"You: {question}")
    print(f"Agent: {await agent.arun(question)}")
    print(f"Conversions the tool logged for the agent: {_CONVERSION_LOG}")


def main() -> None:
    print("=" * 70)
    print("FAIR-LLM: Tool Conformance Suite Demo")
    print("=" * 70)
    print("\n--- A conforming tool ---")
    run_suite(UnitConverterTool(), label="unit_converter (capstone build)")
    print("--- The same tool returning an error string instead of raising ---")
    run_suite(ErrorStringConverterTool(), label="unit_converter (error-string twin)")
    print("--- The conforming tool inside a real agent ---")
    reset_conversion_log()
    asyncio.run(ask_the_agent())


if __name__ == "__main__":
    main()
