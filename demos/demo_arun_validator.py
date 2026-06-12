"""
This demo builds a miniature application on the v0.3.3 validator surface: a
math help desk whose answers are guaranteed to arrive in a fixed format.

The problem this solves for an implementer: an agent can do the work
correctly and still phrase the final answer in a way your application
cannot use - the wrong format for your parser, a policy violation, a
missing citation. You do not want to re-run the whole reasoning loop over
a wording problem, and you do not want format-fixing retry loops scattered
through consumer code.

arun(validator=...) is the framework primitive for exactly this:

    response = await agent.arun(question, validator=policy, max_retries=2)

The agent runs its ReAct cycle once, then hands the candidate answer to
your async validator. Verdict.approve() releases it. Verdict.reject(
feedback) sends ONLY the wrap-up back for a rewrite - the tool work and
reasoning are kept, and your feedback string is shown to the model as the
reason. If every attempt is rejected, the framework raises a typed
ValidatorRejectedError carrying what happened, so your fallback path is a
deliberate decision instead of a mystery string.

The help desk below enforces one house rule: every reply must start with
"ANSWER:" so the (imaginary) frontend can parse it. Watch the rejection
feedback steer the retry, and try to trip the fallback yourself.

Requires a local model; defaults to HuggingFaceAdapter("qwen25-14b").
"""

import asyncio

from fairlib import (
    HuggingFaceAdapter,
    ResponseRepeatEvent,
    RoleDefinition,
    SafeCalculatorTool,
    SimpleAgent,
    SimpleReActPlanner,
    ToolExecutor,
    ToolRegistry,
    ValidatorRejectedError,
    Verdict,
    WorkingMemory,
)

REQUIRED_PREFIX = "ANSWER:"


async def house_format_policy(response: str) -> Verdict:
    """The help desk's response contract, expressed as a validator.

    A validator is just an async callable: str in, Verdict out. Real
    deployments typically close over per-turn context (the expected
    answer, a content classifier, a schema) - the closure is the
    consumer's business; the framework only sees the Verdict.

    The feedback string on reject() matters: it is shown to the model
    verbatim as the reason for the rewrite, so write it the way you
    would coach a person.
    """
    if response.strip().startswith(REQUIRED_PREFIX):
        return Verdict.approve()
    print(f"  [policy] rejected draft: {response.strip()[:70]!r}")
    return Verdict.reject(
        f"Your reply must START with the literal token '{REQUIRED_PREFIX}' "
        "followed by the result. No preamble before the token."
    )


def notice_stuck_rewrites(event: ResponseRepeatEvent) -> None:
    """Operational signal: consecutive rewrites that are near-identical.

    When the model keeps producing the same rejected wording, the retry
    budget is being spent without progress. The framework fires this
    detection-only event so the application can surface it - here we
    print; production code would mark the session for review.
    """
    print(
        f"  [policy] rewrite {event.attempt} is {event.similarity:.0%} "
        f"similar to the previous one - the model may be stuck"
    )


async def main() -> None:
    print("Assembling the math help desk (format-guaranteed responses)...\n")

    llm = HuggingFaceAdapter("qwen25-14b")

    tool_registry = ToolRegistry()
    tool_registry.register_tool(SafeCalculatorTool())
    planner = SimpleReActPlanner(llm, tool_registry)
    planner.prompt_builder.role_definition = RoleDefinition(
        "You are a precise assistant. Use the calculator for any "
        "arithmetic. Keep your final answer short and direct, and begin "
        f"it with the literal token '{REQUIRED_PREFIX}'."
    )
    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=ToolExecutor(tool_registry),
        memory=WorkingMemory(max_size=30),
        max_steps=4,
    )

    # The repeat detector lives on the agent's own event bus - no extra
    # wiring; every SimpleAgent carries one.
    agent.events.subscribe(ResponseRepeatEvent, notice_stuck_rewrites)

    # --- One scripted question so you can see the contract in action ------
    # The model will usually solve the arithmetic but open with prose
    # ("The result is..."), which the policy rejects; the retry arrives
    # with the feedback and the required prefix. The calculator work is
    # NOT redone - only the wrap-up changes.

    question = "What is 13 * 17?"
    print(f"You: {question}")
    try:
        response = await agent.arun(
            question,
            validator=house_format_policy,
            max_retries=2,
        )
        print(f"Help desk: {response}\n")
    except ValidatorRejectedError as exc:
        print(
            f"Help desk (fallback): I could not produce a well-formed "
            f"answer after {exc.attempt_count} attempts.\n"
        )

    print(
        "Every response you receive in this session has passed the "
        f"'{REQUIRED_PREFIX}' policy - or you get the fallback, never a "
        "malformed reply.\n"
    )

    # --- Your turn ---------------------------------------------------------
    print("Ask your own questions. Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        try:
            response = await agent.arun(
                user_input,
                validator=house_format_policy,
                max_retries=2,
            )
            print(f"Help desk: {response}\n")
        except ValidatorRejectedError as exc:
            # The typed error carries everything needed for a deliberate
            # fallback: what the last attempt said, why it was rejected,
            # and how many attempts were spent. The framework never
            # substitutes a placeholder on its own - the application owns
            # what the user sees.
            print(
                f"Help desk (fallback): I could not format that answer "
                f"correctly (tried {exc.attempt_count} times; last "
                f"feedback: {exc.last_feedback[:60]!r}). Please rephrase "
                f"the question.\n"
            )
        except Exception as exc:
            print(f"Help desk could not finish: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    asyncio.run(main())
