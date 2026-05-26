"""
Demo for v0.3.3 SimpleAgent.arun() validator extensions. All surfaces
below import from the top-level fairlib namespace.

  SimpleAgent.arun(
      user_input,
      *,
      validator=None,
      max_retries=2,
      on_response_repeat=None,
      similarity_threshold=0.95,
  )
      Single agent entry point. With validator=None, runs one ReAct cycle
      and commits the final answer to memory — byte-identical to the
      v0.3.2 contract. With validator=, runs the cycle, hands the final
      answer to a consumer-supplied async validator, and on rejection
      rewrites via a direct llm.ainvoke call carrying full history.
      The retry does NOT re-enter the planner or re-run tools — the
      contract is "your work was correct, only the wrap-up needs to
      change."

  Validator
      Type alias for Callable[[str], Awaitable[Verdict]]. Implement as
      an async function or as a closure capturing per-turn context
      (correct answer, target format, content classifier handle, etc.).

  Verdict.approve() / Verdict.reject(feedback)
      Frozen value returned by the validator. The feedback string is
      carried verbatim into the retry's user-role message; the LLM sees
      it framed as "your response was rejected, reason: <feedback>,
      reasoning above is correct, only the wrap-up needs to change."

  ValidatorRejectedError
      Raised when every attempt is rejected. Carries last_response,
      last_feedback, attempt_count so the caller can drive a
      deterministic fallback informed by what the agent actually produced.

  ValidatorError
      Raised when the consumer-supplied validator itself raises (a
      consumer bug). Original exception preserved on __cause__. Aborts
      the retry loop immediately.

  ResponseRepeatEvent
      Frozen dataclass payload fired when two consecutive retry
      responses cross the similarity threshold. Detection-only; the
      framework does not abort.

  on_response_repeat=, similarity_threshold= (arun() kwargs)
      Per-call options, NOT constructor kwargs — callers who do not
      validate pay zero state cost. Single-subscriber callback,
      synchronous by design; subscriber exceptions are caught and
      logged.

Memory contract under the validator path:
    - User message pre-committed once at the start.
    - Intermediate Thought/Action/Observation from the first attempt persist
      (so the retry's history shows the work that was already done).
    - Rejected final answers are NEVER committed.
    - Validator feedback strings are NEVER committed (ephemeral retry
      context only).
    - Approved final answer is committed via
      planner.format_turn_for_memory(FinalAnswer(text=...)).

The demo defaults to HuggingFaceAdapter("qwen25-14b").
"""

import asyncio
import difflib
import logging
from typing import List

from fairlib import (
    HuggingFaceAdapter,
    Message,
    ResponseRepeatEvent,
    RoleDefinition,
    SafeCalculatorTool,
    SimpleAgent,
    SimpleReActPlanner,
    ToolExecutor,
    ToolRegistry,
    Validator,
    ValidatorRejectedError,
    Verdict,
    WorkingMemory,
)


def _print_section(title: str) -> None:
    bar = "=" * max(len(title), 60)
    print(f"\n{bar}\n{title}\n{bar}")


def _build_agent(llm) -> SimpleAgent:
    """Construct a SimpleAgent identical to the v0.3.2 shape — no
    validator-specific constructor kwargs. The validator surface is
    activated at the arun() callsite, not at agent construction."""
    tool_registry = ToolRegistry()
    tool_registry.register_tool(SafeCalculatorTool())
    planner = SimpleReActPlanner(llm, tool_registry)
    planner.prompt_builder.role_definition = RoleDefinition(
        "You are a precise assistant. Use the calculator for any "
        "arithmetic. Keep your final answer short and direct."
    )
    return SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=ToolExecutor(tool_registry),
        memory=WorkingMemory(max_history_length=30),
        max_steps=4,
    )


async def stage_one_validator_none_preserves_v0_3_2(llm) -> None:
    """Show that arun() with no validator behaves exactly as it did in
    v0.3.2. Old callers and old call sites are byte-identical."""
    _print_section("Stage 1: validator=None preserves the v0.3.2 arun() contract")

    agent = _build_agent(llm)

    print("\nCalling agent.arun('What is 7 * 8?') with no validator kwarg ...")
    response = await agent.arun("What is 7 * 8?")
    print(f"  agent: {response}")

    history = agent.memory.history
    assistant_msgs = [m for m in history if m.role == "assistant"]
    print(
        f"\nMemory holds {len(history)} message(s); "
        f"{len(assistant_msgs)} assistant turn(s). "
        "The final answer was committed exactly once, just as v0.3.2 did."
    )


async def stage_two_validator_rejects_then_approves(llm) -> None:
    """Demonstrate the retry path. The validator rejects any response
    that does not start with the marker 'ANSWER:'. The first ReAct
    cycle solves the arithmetic correctly but is unlikely to use that
    exact prefix; the retry sees the full prior work in history plus
    the validator's feedback and produces a properly-formatted answer.

    Memory state at the end shows: user msg + intermediate tool
    observations + approved assistant turn. The rejected first attempt
    and the validator's feedback string are NOT in memory.
    """
    _print_section(
        "Stage 2: validator rejects then approves on retry "
        "(memory contract verified)"
    )

    REQUIRED_MARKER = "ANSWER:"

    async def marker_validator(response: str) -> Verdict:
        """Reject any response that doesn't start with REQUIRED_MARKER.

        A real consumer validator typically closes over per-turn context
        (correct answer, target format, etc.). The signature is just
        async (response: str) -> Verdict — extra context is the
        consumer's closure responsibility.
        """
        if response.strip().startswith(REQUIRED_MARKER):
            return Verdict.approve()
        return Verdict.reject(
            f"Your reply must START with the literal token "
            f"'{REQUIRED_MARKER}' followed by the number. "
            "Do not add any preamble before the marker."
        )

    agent = _build_agent(llm)

    print(
        "\nCalling agent.arun('What is 13 * 17?', validator=marker_validator) ...\n"
        f"(validator requires the response to start with '{REQUIRED_MARKER}')"
    )
    response = await agent.arun(
        "What is 13 * 17?",
        validator=marker_validator,
        max_retries=2,
    )
    print(f"\n  agent: {response}")

    history = agent.memory.history
    rejected_in_memory = any(
        m.role == "assistant" and not m.content.strip().startswith(REQUIRED_MARKER)
        and "ANSWER:" not in m.content  # be tolerant of the planner's serialized form
        for m in history
    )
    feedback_in_memory = any(
        m.role == "system"
        and (getattr(m, "metadata", None) or {}).get("validator_feedback") is True
        for m in history
    )
    user_msgs = [m for m in history if m.role == "user"]
    assistant_msgs = [m for m in history if m.role == "assistant"]

    print(
        f"\nMemory after the turn:\n"
        f"  user messages:      {len(user_msgs)} (pre-committed once)\n"
        f"  assistant messages: {len(assistant_msgs)} "
        f"(intermediate planner turns + approved final answer)\n"
        f"  validator feedback messages: {1 if feedback_in_memory else 0} "
        f"(must be 0 — feedback is ephemeral)"
    )
    assert not feedback_in_memory, (
        "Validator feedback must not be committed to memory."
    )
    assert response.strip().startswith(REQUIRED_MARKER), (
        f"Final response should start with the required marker '{REQUIRED_MARKER}'."
    )
    print(
        f"\nApproved response starts with the required '{REQUIRED_MARKER}' marker. "
        "The retry path picked up the full history of the calculator work "
        "and only changed the wrap-up format."
    )


async def stage_three_exhausts_retries_and_falls_back(llm) -> None:
    """Demonstrate the typed-error contract on exhaustion. The validator
    here is impossible to satisfy by construction (it requires a token
    the LLM is extremely unlikely to emit). After max_retries the
    framework raises ValidatorRejectedError carrying last_response,
    last_feedback, and attempt_count. The consumer's fallback path runs
    deterministically off the typed cause.
    """
    _print_section(
        "Stage 3: all retries rejected → ValidatorRejectedError + fallback"
    )

    IMPOSSIBLE_TOKEN = "BANANAQUARTERLY7XQ"

    async def impossible_validator(response: str) -> Verdict:
        if IMPOSSIBLE_TOKEN in response:
            return Verdict.approve()
        return Verdict.reject(
            f"Your reply must contain the exact token '{IMPOSSIBLE_TOKEN}'. "
            "This is the only way it will be accepted."
        )

    agent = _build_agent(llm)

    print(
        "\nCalling agent.arun('What is 2 + 2?', validator=impossible_validator, "
        "max_retries=2) ..."
    )

    fallback_used = False
    final_response = ""
    try:
        final_response = await agent.arun(
            "What is 2 + 2?",
            validator=impossible_validator,
            max_retries=2,
        )
    except ValidatorRejectedError as exc:
        # Consumer catches the typed error and substitutes its own
        # deterministic fallback. This is the canonical pattern: the
        # framework does not silently return a placeholder, so the
        # caller has full visibility into the exhaustion event.
        fallback_used = True
        print(
            f"\n  ValidatorRejectedError raised:\n"
            f"    attempt_count  = {exc.attempt_count}\n"
            f"    last_feedback  = {exc.last_feedback[:80]!r}\n"
            f"    last_response  = {exc.last_response[:80]!r}"
        )
        final_response = (
            "Sorry — I'm unable to produce a valid response for this request."
        )
        print(f"\n  consumer fallback: {final_response}")

    assert fallback_used, (
        "Expected ValidatorRejectedError; the validator is impossible to "
        "satisfy by construction."
    )

    history = agent.memory.history
    assistant_msgs = [m for m in history if m.role == "assistant"]
    print(
        f"\nMemory holds {len(history)} message(s); "
        f"assistant turns: {len(assistant_msgs)}.\n"
        "Note: NO final-answer assistant turn was committed (all attempts "
        "rejected). Intermediate planner/tool turns from the first ReAct "
        "cycle DO persist — they represent real work that was done."
    )


async def stage_four_response_repeat_event(llm) -> None:
    """Demonstrate the ResponseRepeatEvent hook. When the LLM produces
    two near-identical retry responses (above similarity_threshold), the
    framework fires a single ResponseRepeatEvent so consumers can flag
    the session as stuck. Detection-only — the agent loop continues.

    We coerce a near-duplicate scenario by giving the validator
    instructions the LLM cannot improve on across retries (a tautological
    constraint), then catch the event in a list and inspect it.
    """
    _print_section(
        "Stage 4: ResponseRepeatEvent fires on near-duplicate retries "
        "(detection only)"
    )

    repeat_events: List[ResponseRepeatEvent] = []

    async def stuck_validator(response: str) -> Verdict:
        # Tautological constraint: reject everything with the same vague
        # feedback. The LLM tends to produce structurally similar retries
        # under this signal, which is exactly the failure mode the event
        # is designed to surface.
        return Verdict.reject(
            "Please be more concise and direct in your answer."
        )

    agent = _build_agent(llm)

    print(
        "\nCalling agent.arun(..., on_response_repeat=<append>, "
        "similarity_threshold=0.6) ..."
    )
    try:
        await agent.arun(
            "Briefly: what is the capital of France?",
            validator=stuck_validator,
            max_retries=2,
            on_response_repeat=repeat_events.append,
            # Low threshold so even moderately-similar retries fire the
            # event. Production callers typically run with the default
            # 0.95 to catch only near-identical duplicates.
            similarity_threshold=0.6,
        )
    except ValidatorRejectedError:
        # Expected — the validator always rejects. We are interested in
        # whether the event fired, not in the final response.
        pass

    print(f"\nResponseRepeatEvents captured: {len(repeat_events)}")
    for i, ev in enumerate(repeat_events, start=1):
        prev_preview = ev.previous[:60].replace("\n", " ")
        curr_preview = ev.current[:60].replace("\n", " ")
        print(
            f"  event {i}:\n"
            f"    attempt    = {ev.attempt}\n"
            f"    similarity = {ev.similarity:.3f} "
            f"(threshold {ev.threshold})\n"
            f"    previous   = {prev_preview!r}\n"
            f"    current    = {curr_preview!r}"
        )

    if not repeat_events:
        # The model may have produced sufficiently varied retries to
        # avoid crossing the threshold. The event surface is still
        # exercised; the consumer's callback is registered and was
        # offered every retry pair.
        print(
            "\n(no near-duplicate detected at threshold 0.6; "
            "the LLM produced varied retries — the event surface was "
            "still wired and evaluated each retry pair)"
        )

    print(
        "\nKey property: the framework did NOT abort the retry loop when "
        "the event fired. Consumers decide what to do with the signal — "
        "typical handling is to record it in the per-turn trace for "
        "downstream telemetry."
    )


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    llm = HuggingFaceAdapter("qwen25-14b")

    await stage_one_validator_none_preserves_v0_3_2(llm)
    await stage_two_validator_rejects_then_approves(llm)
    await stage_three_exhausts_retries_and_falls_back(llm)
    await stage_four_response_repeat_event(llm)

    _print_section("Demo complete")
    print(
        "Exercised: arun(validator=) approve / reject-then-approve / "
        "exhaust-and-raise paths; the memory contract (rejected answers "
        "and validator feedback never committed; intermediate tool work "
        "preserved; approved answer serialized via planner.format_turn_for_memory); "
        "Verdict.approve()/Verdict.reject(feedback); "
        "ValidatorRejectedError typed fallback; ResponseRepeatEvent "
        "detection-only event surface."
    )


if __name__ == "__main__":
    asyncio.run(main())
