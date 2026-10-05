# demo_response_pool_template_cycling.py

"""
Response pool and template cycling demonstration.

Tutor and fallback flows often need varied wording without repeating the same
phrase every time. ResponsePool cycles deterministically through a list of
templates and can persist its cursor across restarts.

Part A shows direct ResponsePool cycling with no LLM. Part B wires the same
pool into a small inline tool inside a real HuggingFace-driven agent and asks
for two redirects in a row: each tool call is printed with the redirect the
pool rendered, so you can watch the template advance from one call to the
next and carry on from where Part A left off. A LoopGuardTrippedEvent
subscriber prints a line if the agent ever repeats the same call, so one
call per question is visible, not assumed, and a run that stops without a
final answer prints its typed error (MaxStepsExceeded or
LoopGuardStoppedError) rather than a traceback. The demo also shows state
and load_state so the cursor survives a restart.

Run: PYTHONPATH=. python demos/demo_response_pool_template_cycling.py
Requires a GPU for Part B. Set FAIR_LLM_DEMO_MODEL to override the default model.
"""

import asyncio
import os

from pydantic import BaseModel, Field

from fairlib import (
    HuggingFaceAdapter,
    LoopGuardStoppedError,
    LoopGuardTrippedEvent,
    MaxStepsExceeded,
    ResponsePool,
    RoleDefinition,
    SimpleAgent,
    SimpleReActPlanner,
    ToolCallPostEvent,
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

# Qwen2.5-14B-Instruct by default. The 7B model reads each redirect as a step
# the student already completed and ends by solving the problem itself instead
# of relaying the redirect; the 14B model relays it.
MODEL_NAME = os.getenv("FAIR_LLM_DEMO_MODEL", "qwen25-14b")

# The redirect templates. The pool cycles through them in order.
TEMPLATES = [
    "Let's try a smaller step: {hint}",
    "Another way to approach it: {hint}",
    "Redirecting to the core idea: {hint}",
]


class HintInput(BaseModel):
    """A short hint topic the tutor should weave into the redirect."""

    hint: str = Field(description="The core idea to redirect the student toward.")


class TutorRedirectTool(AbstractTool):
    """Returns the next deterministic tutor redirect from a ResponsePool."""

    name = "tutor_redirect"
    description = (
        "Returns the one tutor redirect sentence to send to a stuck student "
        "for the hint topic you pass. Call it once per student question."
    )
    input_schema = HintInput
    output_schema = TextResult
    side_effect = SideEffect.READ_ONLY

    def __init__(self, pool: ResponsePool) -> None:
        self.pool = pool

    async def acall(self, tool_input: HintInput) -> ToolOutput:
        return TextResult(result=self.pool.render(hint=tool_input.hint))


def on_tool_call(event: ToolCallPostEvent) -> None:
    """Print one tool call: the tool, the hint the agent chose, and the redirect."""
    status = "ok" if event.succeeded else "failed"
    print(
        f"  [{event.tool_name}] {event.tool_input!r} -> {status}: {event.observation}"
    )


def on_loop_guard(event: LoopGuardTrippedEvent) -> None:
    """Print a loop-guard trip: the agent repeated one call, or kept failing."""
    print(
        f"  [loop guard] {event.guard_type.value} at step {event.step}: "
        f"{event.count} in a row (threshold {event.threshold})"
    )


def demo_pool_primitive() -> ResponsePool:
    """Part A: show deterministic cycling without any model."""
    print("=== Part A: ResponsePool primitive (no LLM) ===")
    pool = ResponsePool(TEMPLATES)
    for topic in ("factor first", "draw the graph", "check units"):
        print("Redirect:", pool.render(hint=topic))
    return pool


async def demo_pool_in_agent(pool: ResponsePool) -> None:
    """Part B: the same pool inside a real agent tool loop."""
    print("\n=== Part B: ResponsePool inside a real agent loop ===")
    print(f"Loading {MODEL_NAME} via HuggingFaceAdapter...")
    # Greedy decoding (do_sample=False): a sampled run sometimes relays the
    # hint alone or copies memory's observation label into the answer, where
    # the greedy one relays the redirect sentence as the tool returned it.
    llm = HuggingFaceAdapter(MODEL_NAME, max_new_tokens=256, do_sample=False)

    registry = ToolRegistry()
    registry.register_tool(TutorRedirectTool(pool))
    executor = ToolExecutor(registry)
    planner = SimpleReActPlanner(llm, registry)
    # The role names the two steps of a request (call the tool once, then
    # answer with its result word for word), so the model answers after one
    # tool call.
    planner.prompt_builder.role_definition = RoleDefinition(
        "You relay tutor redirects for a math teacher. For each request, take "
        "exactly two steps. Step 1: call tutor_redirect once with a short hint "
        "topic of a few words (the next small step the student should try). "
        "Step 2: give your final answer, whose text is the tool's result copied "
        "word for word. The tool's result is always the finished redirect, so "
        "never call the tool a second time for the same request, and never "
        "solve the student's problem."
    )
    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=executor,
        memory=WorkingMemory(),
        max_steps=6,
        # Two identical calls in a row trip the guard, so a repeat shows up.
        repeat_signature_threshold=2,
    )
    # Each tool call prints the redirect the pool rendered for it.
    agent.events.subscribe(ToolCallPostEvent, on_tool_call)
    agent.events.subscribe(LoopGuardTrippedEvent, on_loop_guard)

    prompts = (
        "A student is stuck factoring x**2 + 5*x + 6. What redirect should I send them?",
        "A student cannot start solving 2*x + 3 = 11. What redirect should I send them?",
    )
    for prompt in prompts:
        print(f"\nUser: {prompt}")
        print(f"  (next template: {pool.peek()!r})")
        try:
            answer = await agent.arun(prompt)
        except (MaxStepsExceeded, LoopGuardStoppedError) as exc:
            # A run that never reaches a final answer ends with a typed error;
            # the redirects the pool rendered before it are printed above.
            print(f"Agent stopped: {type(exc).__name__}: {exc}")
            continue
        print("Agent:", answer)


def demo_pool_persistence(pool: ResponsePool) -> None:
    """Show cursor save/load for deterministic behavior across restarts."""
    print("\n=== Persistence: state() / load_state() ===")
    saved = pool.state()
    print("Saved cursor index:", saved.index)
    restored = ResponsePool(TEMPLATES)
    restored.load_state({"index": saved.index})
    print("Live pool next:     ", pool.peek())
    print("Restored pool next: ", restored.peek())


async def main() -> None:
    pool = demo_pool_primitive()
    await demo_pool_in_agent(pool)
    demo_pool_persistence(pool)


if __name__ == "__main__":
    asyncio.run(main())
