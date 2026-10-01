"""
This demo builds a miniature application on fairlib's memory surfaces: a
trip-planning assistant that never forgets your non-negotiables.

The problem this solves for an implementer: a long conversation eventually
overflows the context window, so SummarizingMemory compresses old turns
into a summary. Compression is lossy - and some messages must never be
lossy. A budget ceiling. An accessibility requirement. The user's name.
If those land in the summary blob, the agent starts violating them and the
user notices immediately.

The fix is two coordinated primitives, both flexed here:

    Message(importance="pinned")
        The caller's "this must survive verbatim" signal. Pinned messages
        are excluded from summarization and reinserted in their original
        positions. agent.arun() accepts a full Message, so the user turn
        itself can be pinned at the callsite.

    MemorySummarizedEvent via SummarizingMemory(events=bus)
        Every compaction announces itself on the event bus: what was
        dropped, what was kept, the summary text. The assistant below uses
        it to tell the user their constraints were carried forward -
        no guessing, no inspecting memory internals.

The session starts with a short scripted budgeting conversation (so
compaction actually happens) and prints the totals the stated prices imply
next to the agent's answer, then hands the keyboard to you.

Requires a local model; defaults to HuggingFaceAdapter("qwen25-14b").
"""

import asyncio

from fairlib import (
    AgentEventBus,
    HuggingFaceAdapter,
    MemorySummarizedEvent,
    Message,
    RoleDefinition,
    SafeCalculatorTool,
    SimpleAgent,
    SimpleReActPlanner,
    SummarizingMemory,
    ToolCallPostEvent,
    ToolExecutor,
    ToolRegistry,
)


def announce_compaction(event: MemorySummarizedEvent) -> None:
    """Turn the compaction event into a user-facing reassurance.

    This is the consumer pattern the event exists for: the application
    decides what a summarization means to its user. Here we surface it;
    a quieter application might log it; a web UI might badge it.
    """
    pinned_kept = sum(1 for m in event.kept if m.importance == "pinned")
    summary = event.summary.content.replace("\n", " ")
    # reason is a SummarizationReason enum; .value is the friendly string.
    print(
        f"\n  [memory] Conversation compacted ({event.reason.value}): "
        f"{len(event.dropped)} older messages folded into a summary, "
        f"{len(event.kept)} kept verbatim - including your "
        f"{pinned_kept} pinned constraint(s).\n"
        f"  [memory] Summary now reads: {summary!r}\n"
    )


def show_tool_call(event: ToolCallPostEvent) -> None:
    """Print each calculator call, so the arithmetic behind an answer is visible."""
    print(f"  [{event.tool_name}] {event.tool_input!r} -> {event.observation}")


def show_memory(memory: SummarizingMemory) -> None:
    """Print the live history with pinned messages marked."""
    print(f"\nMemory currently holds {len(memory.history)} messages:")
    for i, m in enumerate(memory.history):
        marker = "[pinned]" if m.importance == "pinned" else "        "
        preview = m.content[:84].replace("\n", " ")
        print(f"  {i:2d}. {marker} {m.role:>9}: {preview}")
    print()


async def main() -> None:
    print("Assembling a trip-planning agent with pinned-constraint memory...\n")

    llm = HuggingFaceAdapter("qwen25-14b")

    bus = AgentEventBus()
    bus.subscribe(MemorySummarizedEvent, announce_compaction)
    bus.subscribe(ToolCallPostEvent, show_tool_call)

    # max_history_length is deliberately small so you can watch
    # summarization happen within one short planning session. It still sits
    # well above what a compaction keeps (the pinned messages, the summary and
    # the last messages_to_keep_at_end), so each compaction buys several
    # messages of headroom instead of firing again on the very next step.
    # Six kept messages cover a whole budgeting turn (the request, two
    # calculator calls with their results, and the answer), so a compaction
    # that lands mid-turn summarizes only earlier turns, never the one the
    # agent is still working on.
    memory = SummarizingMemory(
        llm=llm,
        max_history_length=16,
        messages_to_keep_at_end=6,
        events=bus,
    )

    tool_registry = ToolRegistry()
    tool_registry.register_tool(SafeCalculatorTool())
    planner = SimpleReActPlanner(llm, tool_registry)
    planner.prompt_builder.role_definition = RoleDefinition(
        "You are a budget-aware trip planner. The only costs on this trip are "
        "the prices the user states: never add a cost for a stop or item the "
        "user did not price, and never propose priced extras of your own. A "
        "message that gives no new stop cost needs no arithmetic: acknowledge "
        "it in one sentence as your final answer, without the calculator. Use "
        "the calculator for any arithmetic. For each new stop, make one "
        "calculator call that adds the stop's cost to the running total from "
        "your previous answer; its result is the new running total, so never "
        "add the same cost again. Then compute the remaining budget from the "
        "budget constraint, and end your answer with the line 'Running total: "
        "<spent> USD spent, <remaining> USD remaining.' Respect every "
        "HARD CONSTRAINT you have been given. Keep answers short and always "
        "finish your turn with a clear final answer."
    )

    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=ToolExecutor(tool_registry, events=bus),
        memory=memory,
        # Room for a calculator call or two per turn even when a compaction
        # lands mid-turn.
        max_steps=8,
        events=bus,
    )

    # --- Scripted opening: a realistic planning session -------------------
    # The two constraints are sent as pinned Messages. Everything else is a
    # plain string turn, free to be summarized away. By the final scripted
    # turn the history exceeds max_history_length, compaction fires, and
    # the constraints are still in memory verbatim - which is why the
    # agent can still do the budget math correctly afterward.

    budget = 1500
    stop_costs = {"aquarium": 2 * 30, "coffee": 2 * 12, "museum": 2 * 25}
    opening_turns = [
        "Hi! I am planning a 3-stop weekend trip in San Francisco for two. I "
        "will tell you each stop and its price; keep a running total of only "
        "the costs I give you.",
        Message(
            role="user",
            content=f"HARD CONSTRAINT: my total budget is {budget} USD for two people.",
            importance="pinned",
        ),
        Message(
            role="user",
            content="HARD CONSTRAINT: every stop must be wheelchair accessible.",
            importance="pinned",
        ),
        "Stop 1: an aquarium. Admission is 30 USD per person, so 60 USD for "
        "the two of us. What have we spent so far?",
        "Stop 2: a coffee shop nearby. Two drinks at 12 USD each, so 24 USD. "
        "Update the running total.",
        "Stop 3: a museum at 25 USD per person, so 50 USD for the two of us. "
        "Now total all three stops and tell me how much of the budget remains.",
    ]

    for turn in opening_turns:
        text = turn.content if isinstance(turn, Message) else turn
        pinned = isinstance(turn, Message) and turn.importance == "pinned"
        tag = "  (pinned)" if pinned else ""
        print(f"You{tag}: {text}")
        try:
            response = await agent.arun(turn)
            print(f"Agent: {response}\n")
        except Exception as exc:
            print(f"Agent could not finish: {type(exc).__name__}: {exc}\n")

    spent = sum(stop_costs.values())
    print(
        f"The stated prices imply: spent {spent} USD "
        f"({' + '.join(str(c) for c in stop_costs.values())}), "
        f"remaining {budget - spent} USD. Compare with the agent's last answer."
    )

    show_memory(memory)
    print(
        "Note what survived: the conversation has been compacted at least "
        "once, yet both HARD CONSTRAINT messages are still there verbatim, "
        "at full fidelity, while ordinary turns became a summary."
    )

    # --- Your turn ---------------------------------------------------------
    print("\nThe session is now yours. Keep planning, or stress the memory.")
    print("Commands:  pin: <text>  - send a turn as a pinned constraint")
    print("           memory       - show history with pinned markers")
    print("           exit         - quit\n")

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
        if user_input.lower() == "memory":
            show_memory(memory)
            continue

        if user_input.lower().startswith("pin:"):
            turn = Message(
                role="user",
                content=user_input[4:].strip(),
                importance="pinned",
            )
            print("  (this turn is pinned - it will survive every compaction)")
        else:
            turn = user_input

        try:
            response = await agent.arun(turn)
            print(f"Agent: {response}\n")
        except Exception as exc:
            print(f"Agent could not finish: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    asyncio.run(main())
