# demo_tool_catalog_and_validation.py

"""
Auto-generated tool catalog and input validation before dispatch.

A tool declares its input as a Pydantic schema. From that one declaration three
things follow automatically, with no hand-written tool signatures to drift:

  1. The catalog the model reads is generated from the schema - every field, its
     type, whether it is required, and its description. The renderer unwraps the
     hard cases too: a nested model is expanded into its fields, a list reads as
     "array of <element>", and Literal/Enum fields enumerate their legal values.
     This demo prints that catalog so you can see it is the schema, rendered.
  2. The executor validates the model's tool_input against the same schema before
     the tool runs. A call that does not fit is rejected as a typed
     ToolInputValidationError, before any tool code executes, and the error
     carries the expected-input shape (schema_hint) so the failure is actionable.
  3. When that rejection happens inside the agent loop, the loop echoes the
     expected-input shape back into the observation, so the model self-corrects
     on the next step instead of retrying the same malformed call.

The agent then runs end to end against a real local model on a tool whose input
is deliberately non-trivial (a nested object, an enum, and a list of literals).
The prompt the model reads and the schema the executor enforces come from the
same source, so they cannot disagree. The loop transcript is printed at the end:
if the model fumbles the nested shape on the first try, you will see the
"Expected input for ..." correction (feature 3) and the recovered call; if it
gets it right first time, the rich catalog did its job up front.

Run:
    python demos/demo_tool_catalog_and_validation.py
"""

import asyncio
import enum
from typing import List, Literal

from pydantic import BaseModel, Field

from fairlib import (
    HuggingFaceAdapter,
    SimpleAgent,
    SimpleReActPlanner,
    ToolExecutor,
    ToolInputValidationError,
    ToolRegistry,
    UnverifiedCompletionEvent,
    WorkingMemory,
)
from fairlib.core.interfaces.tools import AbstractTool, SideEffect, TextResult, ToolOutput
from fairlib.core.prompts import PromptBuilder

MODEL_NAME = "dolphin3-qwen25-3b"


# --- A deliberately non-trivial schema: nested model + enum + list of literals ---
# This is what exercises the catalog renderer's hard cases. Each piece below
# shows up in the generated catalog without a line of hand-written prompt.


class Location(BaseModel):
    """Where an event happens - a nested object inside the tool input."""

    city: str = Field(description="City name, e.g. Denver.")
    country: str = Field(description="Two-letter country code, e.g. US.")


class Priority(enum.Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class ScheduleEventInput(BaseModel):
    """The catalog the model reads is generated from this one declaration."""

    title: str = Field(description="Short event title.")
    location: Location = Field(description="Where the event happens.")
    priority: Priority = Field(default=Priority.NORMAL, description="How important the event is.")
    tags: List[Literal["work", "personal", "urgent"]] = Field(
        default_factory=list, description="Zero or more labels for the event."
    )


class ScheduleEventTool(AbstractTool):
    name = "schedule_event"
    description = "Schedule a calendar event at a location with a priority and tags."
    input_schema = ScheduleEventInput
    output_schema = TextResult
    side_effect = SideEffect.READ_ONLY

    async def acall(self, tool_input: ScheduleEventInput) -> ToolOutput:
        loc = tool_input.location
        tags = ", ".join(tool_input.tags) if tool_input.tags else "none"
        return TextResult(
            result=(
                f"Scheduled '{tool_input.title}' in {loc.city}, {loc.country} "
                f"at {tool_input.priority.value} priority (tags: {tags})."
            )
        )


QUESTION = (
    "Schedule a high-priority event titled 'Sprint Review' in Denver, US, "
    "tagged work and urgent. Use the schedule_event tool, then confirm what you booked."
)


def show_generated_catalog(registry: ToolRegistry) -> None:
    """Feature 1: print the tool catalog the model will receive, from the schema."""
    builder = PromptBuilder()
    builder.add_tool_registry(registry)
    print("=== Feature 1: Auto-generated catalog (nested model, enum, list of literals) ===")
    print(builder.render_tool_catalog())
    print()


async def show_validation_rejection(registry: ToolRegistry) -> None:
    """Feature 2: a malformed call rejected before the tool runs, with the hint it carries."""
    executor = ToolExecutor(registry)
    # 'location' must be an object; passing a bare string is a hard schema mismatch.
    bad_input = {"title": "Sprint Review", "location": "Denver", "priority": "high"}
    print("=== Feature 2: Validation before dispatch (typed error carries the schema) ===")
    print(f"Calling schedule_event with a bad location: {bad_input}")
    try:
        await executor.aexecute("schedule_event", bad_input)
    except ToolInputValidationError as exc:
        print(f"Rejected before the tool ran -> {type(exc).__name__}: {exc}")
        print("schema_hint carried on the error (what the loop echoes back):")
        print(exc.schema_hint)
    print()


def print_history(agent: SimpleAgent) -> None:
    """Surface the loop transcript so any in-loop self-correction is visible."""
    print("\n=== Loop transcript (Feature 3 fires here if the model fumbles first) ===")
    for message in agent.memory.history:
        preview = message.content.replace("\n", " | ")[:160]
        print(f"  [{message.role}] {preview}")


async def main() -> None:
    registry = ToolRegistry()
    registry.register_tool(ScheduleEventTool())

    show_generated_catalog(registry)
    await show_validation_rejection(registry)

    print(f"Loading {MODEL_NAME} via the HuggingFaceAdapter (first run downloads weights)...")
    llm = HuggingFaceAdapter(MODEL_NAME, max_new_tokens=512)

    executor = ToolExecutor(registry)
    planner = SimpleReActPlanner(llm, registry)

    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=executor,
        memory=WorkingMemory(),
        max_steps=8,
    )

    # Robustness signal: if the model declares success after every tool attempt
    # failed (a weak model that cannot satisfy the schema), the loop emits this
    # so the caller can flag the answer rather than trust it. A capable model
    # never trips it.
    def _on_unverified(event: UnverifiedCompletionEvent) -> None:
        print(
            f"\n[!] UnverifiedCompletionEvent: the agent declared completion after "
            f"{event.tool_attempts} tool attempt(s), 0 succeeded - treat the answer "
            f"as unverified."
        )

    agent.events.subscribe(UnverifiedCompletionEvent, _on_unverified)

    print("=" * 70)
    print(f"Question:\n  {QUESTION}\n")
    print("Running the agent (it learns the tool's fields from the generated catalog)...\n")
    answer = await agent.arun(QUESTION)
    print("\nAgent answer:")
    print(answer)
    print_history(agent)
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
