# demo_structured_output.py
"""Structured output: an agent whose final answer must be valid JSON.

Many applications need a form filled in, not prose: parse a resume into
skills, turn an email into a calendar event, pull a rating out of a review.
The pattern here is a Pydantic schema plus the agent's validator surface:

    response = await agent.arun(text, validator=conforms_to_schema, max_retries=3)

The agent runs once, then hands its final answer to the validator. A reply
that does not parse as the schema is rejected with the validation error as
feedback, and only the wrap-up is rewritten; after max_retries the framework
raises a typed ValidatorRejectedError instead of returning a broken reply.

Requires a local model; defaults to HuggingFaceAdapter("qwen25-7b").
Set FAIR_LLM_DEMO_MODEL to override.
"""

import asyncio
import json
import os
from typing import List

from pydantic import BaseModel, Field, ValidationError

from fairlib import (
    HuggingFaceAdapter,
    MaxStepsExceeded,
    RoleDefinition,
    SimpleAgent,
    SimpleReActPlanner,
    ToolExecutor,
    ToolRegistry,
    ValidatorRejectedError,
    Verdict,
    WorkingMemory,
)

MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "qwen25-7b")


# --- Step 1: the schema is the form the agent has to fill in -----------------
class UserProfile(BaseModel):
    name: str = Field(..., description="The full name of the user.")
    age: int = Field(..., description="The age of the user.")
    city: str = Field(..., description="The city where the user resides.")
    interests: List[str] = Field(..., description="The user's interests or hobbies.")
    is_student: bool = Field(
        ..., description="Whether the user is currently a student."
    )


def _json_object(text: str) -> str:
    """The first {...} block in a reply, so a stray sentence or fence around it is tolerated."""
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text.strip()


# --- Step 2: the validator is the contract the final answer must meet --------
async def conforms_to_schema(response: str) -> Verdict:
    """Approve a reply that parses as UserProfile; otherwise coach the rewrite."""
    try:
        UserProfile.model_validate_json(_json_object(response))
        return Verdict.approve()
    except ValidationError as exc:
        print(f"  [validator] rejected draft: {response.strip()[:60]!r}")
        return Verdict.reject(
            "Your reply must be ONLY a JSON object matching the schema. "
            f"Validation error: {exc.errors()[0]['msg']} at "
            f"{'.'.join(str(p) for p in exc.errors()[0]['loc'])}."
        )


async def main() -> None:
    print("Initializing components...")
    llm = HuggingFaceAdapter(MODEL_NAME, max_new_tokens=256)

    # --- Step 3: build the agent; the schema goes in the role definition -----
    tool_registry = ToolRegistry()  # no tools: this agent only reads and writes
    planner = SimpleReActPlanner(llm, tool_registry)
    template = json.dumps(
        {
            "name": "Full Name",
            "age": 0,
            "city": "City",
            "interests": ["one", "two"],
            "is_student": False,
        }
    )
    planner.prompt_builder.role_definition = RoleDefinition(
        "You are a data extraction engine. Read the text the user gives you "
        "and answer with ONLY a JSON object of this exact shape, with the "
        f"values filled in from the text: {template}"
    )
    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=ToolExecutor(tool_registry),
        memory=WorkingMemory(),
        max_steps=3,
    )

    unstructured_text = (
        "My name is Jane Doe and I live in San Francisco. I am 28 years old and "
        "a full-time student. In my free time I enjoy painting, playing the "
        "guitar, and long-distance running."
    )
    print("\n--- Input Text ---")
    print(unstructured_text)

    # --- Step 4: run with the validator; the framework owns the retries ------
    try:
        response = await agent.arun(
            unstructured_text, validator=conforms_to_schema, max_retries=3
        )
    except ValidatorRejectedError as exc:
        print(
            f"\nNo valid profile after {exc.attempt_count} attempts. "
            f"Last feedback: {exc.last_feedback}"
        )
        return
    except MaxStepsExceeded as exc:
        # A small model sometimes never reaches a final answer within the
        # step budget; that is the agent's typed signal, not a crash.
        print(f"\nThe agent ran out of steps before answering: {exc}")
        return

    profile = UserProfile.model_validate_json(_json_object(response))
    print("\n--- Final Structured Data ---")
    print(profile.model_dump_json(indent=2))
    print(f"\nExtracted name: {profile.name}")
    print(f"Extracted interests: {profile.interests}")


if __name__ == "__main__":
    asyncio.run(main())
