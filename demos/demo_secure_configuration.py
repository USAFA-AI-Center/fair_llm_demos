"""Default-deny agent configuration against a real local model.

Loads HuggingFaceAdapter (FAIR_LLM_DEMO_MODEL, default
Qwen/Qwen2.5-7B-Instruct) and shows three halves with the same builder the
docs pin, then the same grants with the file tool isolated:

  (a) empty bag refuses the model before any generation
  (b) a narrow bag lets the agent read a granted file and answer
  (c) the same bag refuses an ungranted path as CAPABILITY_DENIED
  (d) the read tool declared IsolationLevel.ISOLATED on its instance: the
      manager runs every read in a confined child that reaches only the
      granted paths, the granted read completes there, and the ungranted
      read comes back from the child as CAPABILITY_DENIED

Part (d) is the adopter's path and nothing more: declare the isolation
level on the tool and pass the manager that holds the bag to the executor.
The agent binds that manager to its own bus, so each child run prints as a
BoundedRunEvent subscribed on the agent's events. A host that does not offer the isolated level (the manager's
available_levels() says which levels it reaches) refuses the declared tool
with a typed SandboxUnavailableError rather than run it in-process; the demo
prints that refusal and still exits 0.

Requirements: fair-llm[local] (transformers; GPU recommended). The first run
may download weights. Set FAIR_LLM_DEMO_MODEL to override the default alias.

Run:
    python -m demos.demo_secure_configuration
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

from fairlib import (
    AbstractChatModel,
    AbstractToolRegistry,
    BasicSecurityManager,
    BoundedRunEvent,
    CapabilityBag,
    HuggingFaceAdapter,
    IsolationLevel,
    ReActPlanner,
    ReadFileTool,
    SimpleAgent,
    ToolExecutor,
    ToolRegistry,
    WorkingMemory,
)
from fairlib.core.errors import CapabilityDeniedError, ToolInvocationError
from fairlib.core.events import CapabilityDeniedEvent, ToolCallPostEvent

# Default matches demos that need reliable ReAct tool JSON (e.g.
# demo_multi_tool_turn). Override with FAIR_LLM_DEMO_MODEL for a local alias.
MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "Qwen/Qwen2.5-7B-Instruct")


def build_secure_agent(
    llm: AbstractChatModel,
    registry: AbstractToolRegistry,
    *,
    grants: CapabilityBag = CapabilityBag(),
) -> SimpleAgent:
    """Enable capability enforcement with no grants unless explicitly supplied."""
    security = BasicSecurityManager(capability_bag=grants)
    return SimpleAgent(
        llm=llm,
        planner=ReActPlanner(llm, registry, stream=False),
        tool_executor=ToolExecutor(registry, security_manager=security),
        memory=WorkingMemory(),
        max_steps=3,
    )


def build_isolated_agent(
    llm: AbstractChatModel,
    registry: AbstractToolRegistry,
    manager: BasicSecurityManager,
) -> SimpleAgent:
    """Run the registry's tools under manager, whose bounded runs report on the agent's bus."""
    return SimpleAgent(
        llm=llm,
        planner=ReActPlanner(llm, registry, stream=False),
        tool_executor=ToolExecutor(registry, security_manager=manager),
        memory=WorkingMemory(),
        max_steps=6,
    )


def print_posts(posts: list[ToolCallPostEvent]) -> None:
    """Print each tool dispatch's outcome and the type of its error's cause."""
    for post in posts:
        kind = post.error.kind.value if post.error is not None else None
        cause = (
            type(post.error.__cause__).__name__
            if post.error is not None and post.error.__cause__ is not None
            else None
        )
        print(
            f"post: tool={post.tool_name} ok={post.succeeded} "
            f"kind={kind} cause={cause} obs={post.observation!r}"
        )


def print_bounded_run(event: BoundedRunEvent) -> None:
    """Print one child run: the level required, the level attested, the outcome."""
    achieved = event.achieved.value if event.achieved is not None else "none"
    print(
        f"bounded run: tool={event.tool_name} required={event.required.value} "
        f"achieved={achieved} outcome={event.outcome.value} "
        f"duration={event.duration_seconds:.2f}s"
    )


async def main() -> int:
    print(f"Loading model {MODEL_NAME!r} ...")
    # Low temperature + enough tokens keeps small local models on the ReAct
    # JSON contract (same knobs as demos/mcp/demo_mcp_hostile_server.py).
    llm = HuggingFaceAdapter(
        model_name=MODEL_NAME,
        temperature=0.1,
        max_new_tokens=256,
    )
    model_id = llm.describe_config().model_name
    print(f"Resolved model id: {model_id!r}")

    print("\n(a) empty bag - model refusal before generation")
    empty_agent = build_secure_agent(llm, ToolRegistry())
    try:
        await empty_agent.arun("Say hello.")
        print("empty bag: UNEXPECTED SUCCESS")
        return 1
    except CapabilityDeniedError as exc:
        print(f"refused: {type(exc).__name__} {exc.grant_kind} {exc.resource}")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        public = root / "public"
        private = root / "private"
        public.mkdir()
        private.mkdir()
        (public / "notes.txt").write_text(
            "The code word is PELICAN.\n", encoding="utf-8"
        )
        (private / "secret.txt").write_text(
            "TOP SECRET - must not be readable.\n", encoding="utf-8"
        )

        registry = ToolRegistry()
        registry.register_tool(ReadFileTool(root=root))
        grants = CapabilityBag.from_grants(
            tools=["read_file"],
            path_prefixes=[str(public)],
            hosts=[],
            models=[model_id],
        )

        print("\n(b) granted path - read public/notes.txt")
        agent_b = build_secure_agent(llm, registry, grants=grants)
        agent_b.max_steps = 6
        posts_b: list[ToolCallPostEvent] = []
        agent_b.events.subscribe(ToolCallPostEvent, posts_b.append)
        answer_b = await agent_b.arun(
            "Read the file public/notes.txt with the read_file tool and "
            "report the code word. Reply with only the code word."
        )
        print(f"answer: {answer_b}")
        print_posts(posts_b)
        if "PELICAN" not in (answer_b or "").upper() and not any(
            p.succeeded and p.tool_name == "read_file" for p in posts_b
        ):
            print("granted path: FAIL (no successful read_file / code word)")
            return 1

        print("\n(c) ungranted path - refuse private/secret.txt")
        agent_c = build_secure_agent(llm, registry, grants=grants)
        agent_c.max_steps = 6
        posts_c: list[ToolCallPostEvent] = []
        denials: list[CapabilityDeniedEvent] = []
        agent_c.events.subscribe(ToolCallPostEvent, posts_c.append)
        agent_c.events.subscribe(CapabilityDeniedEvent, denials.append)
        answer_c = await agent_c.arun(
            "Read the file private/secret.txt with the read_file tool and "
            "report its contents. If you cannot, say so briefly."
        )
        print(f"answer: {answer_c}")
        print_posts(posts_c)
        denied_post = None
        for post in posts_c:
            if (
                post.tool_name == "read_file"
                and not post.succeeded
                and isinstance(post.error, ToolInvocationError)
                and post.error.kind is ToolInvocationError.Kind.CAPABILITY_DENIED
            ):
                denied_post = post
        for denial in denials:
            print(f"denial event: {denial.grant_kind} {denial.resource}")

        if denied_post is None:
            print("ungranted path: FAIL (expected CAPABILITY_DENIED on read_file)")
            return 1
        if any("TOP SECRET" in (p.observation or "") for p in posts_c):
            print("ungranted path: FAIL (secret leaked into observation)")
            return 1
        if "TOP SECRET" in (answer_c or ""):
            print("ungranted path: FAIL (secret leaked into final answer)")
            return 1

        print("\n(d) isolated - the same grants, the read tool in a confined child")
        # The whole adopter's path: declare the level on the tool instance,
        # pass the manager holding the bag to the executor, bind the bus.
        manager = BasicSecurityManager(capability_bag=grants)
        levels = sorted(level.value for level in manager.available_levels())
        print(f"available levels on this host: {levels}")
        isolated_reader = ReadFileTool(root=root)
        isolated_reader.isolation = IsolationLevel.ISOLATED
        registry_d = ToolRegistry()
        registry_d.register_tool(isolated_reader)
        isolated_offered = IsolationLevel.ISOLATED in manager.available_levels()

        agent_granted = build_isolated_agent(llm, registry_d, manager)
        posts_granted: list[ToolCallPostEvent] = []
        runs_granted: list[BoundedRunEvent] = []
        agent_granted.events.subscribe(ToolCallPostEvent, posts_granted.append)
        agent_granted.events.subscribe(BoundedRunEvent, runs_granted.append)
        answer_granted = await agent_granted.arun(
            "Read the file public/notes.txt with the read_file tool and "
            "report the code word. Reply with only the code word."
        )
        print(f"answer: {answer_granted}")
        for run in runs_granted:
            print_bounded_run(run)
        print_posts(posts_granted)

        if not isolated_offered:
            # The declared tool was refused before any child ran; the lines
            # above carry the refusal (a refused bounded run, and a post
            # whose cause is SandboxUnavailableError).
            print(
                "isolated: this host does not offer IsolationLevel.ISOLATED; "
                "the declared tool was refused, never run in-process"
            )
            print("\ndemo_secure_configuration: OK (part (d) refused on this host)")
            return 0

        if not any(
            run.required is IsolationLevel.ISOLATED
            and run.achieved is IsolationLevel.ISOLATED
            and run.outcome is BoundedRunEvent.Outcome.COMPLETED
            for run in runs_granted
        ):
            print("isolated granted path: FAIL (no completed isolated read)")
            return 1

        agent_denied = build_isolated_agent(llm, registry_d, manager)
        posts_denied: list[ToolCallPostEvent] = []
        runs_denied: list[BoundedRunEvent] = []
        denials_d: list[CapabilityDeniedEvent] = []
        agent_denied.events.subscribe(ToolCallPostEvent, posts_denied.append)
        agent_denied.events.subscribe(BoundedRunEvent, runs_denied.append)
        agent_denied.events.subscribe(CapabilityDeniedEvent, denials_d.append)
        answer_denied = await agent_denied.arun(
            "Read the file private/secret.txt with the read_file tool and "
            "report its contents. If you cannot, say so briefly."
        )
        print(f"answer: {answer_denied}")
        for run in runs_denied:
            print_bounded_run(run)
        print_posts(posts_denied)
        for denial in denials_d:
            print(f"denial event: {denial.grant_kind} {denial.resource}")

        if not any(
            post.tool_name == "read_file"
            and isinstance(post.error, ToolInvocationError)
            and post.error.kind is ToolInvocationError.Kind.CAPABILITY_DENIED
            for post in posts_denied
        ) or not any(denial.grant_kind == "path" for denial in denials_d):
            print("isolated ungranted path: FAIL (expected a path CAPABILITY_DENIED)")
            return 1
        if any("TOP SECRET" in (p.observation or "") for p in posts_denied):
            print("isolated ungranted path: FAIL (secret leaked into observation)")
            return 1
        if "TOP SECRET" in (answer_denied or ""):
            print("isolated ungranted path: FAIL (secret leaked into final answer)")
            return 1

    print("\ndemo_secure_configuration: OK")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
