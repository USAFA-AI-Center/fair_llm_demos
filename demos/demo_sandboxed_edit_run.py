# demo_sandboxed_edit_run.py

"""
Sandboxed read-edit-run loop with the mutating standard tools.

This demo extends the agentic-search demo from a read-only navigator into an
agent that can change the project and run it. It is given the read tools
(list_dir, glob, grep, read_file) plus the mutating ones (edit_file, write_file)
and the shell tool, all confined to one directory, and asked to fix a bug: a
check script fails, and the agent must find the cause, edit the source, and run
the check through the shell to confirm the fix.

Two containment boundaries are on display. Every file tool is confined to the
granted root, so an edit cannot land outside the project. The shell tool runs
every command in an isolated child process: the executor's security manager
(BasicSecurityManager) carries a capability bag that grants the tools, the
agent's model and exactly the project root, and the child is confined by the
kernel to what that bag grants. Each shell call runs under a timeout, after
which the command and everything it started are ended. The demo prints every
tool call with its input as it returns, a BoundedRunEvent for every shell
call naming the level the child reached, and at the end the shell output of
the last check the agent ran.
"""

import asyncio
import tempfile
from pathlib import Path

from fairlib import (
    BasicSecurityManager,
    BoundedRunEvent,
    CapabilityBag,
    EditFileTool,
    GlobTool,
    GrepTool,
    HuggingFaceAdapter,
    ListDirTool,
    ReadFileTool,
    RoleDefinition,
    ShellTool,
    SimpleAgent,
    SimpleReActPlanner,
    ToolCallPostEvent,
    ToolExecutor,
    ToolRegistry,
    WorkingMemory,
    WriteFileTool,
    settings,
)
from fairlib.core.config_schemas import ShellLimits

MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"

# Every shell call that passes no timeout of its own runs under this one.
SHELL_TIMEOUT_SECONDS = 60

# What the shell's isolated child can and cannot do with the bag built below.
CONFINEMENT_CLAIM = (
    "Commands run in a confined child. It writes only the paths its capability "
    "bag grants (here, just the project root) and its own scratch directory; it "
    "reads only those plus the interpreter, the system libraries, fairlib's code "
    "and the declared programs; it opens no network or listening socket (only a "
    "connected Unix stream pair); it can still read any path's metadata and see "
    "which processes exist."
)


# A tiny project whose check script fails because add() returns only its first
# argument. The agent is not told where the bug is; it has to find calculator.py,
# fix the return line, and run the check to confirm. The fix is a single
# unique-substring edit.
FIXTURE_FILES = {
    "README.md": (
        "# Calc\n\nA toy math package. Run `python3 run_checks.py` to verify it.\n"
    ),
    "app/__init__.py": "",
    "app/calculator.py": (
        "def add(a, b):\n"
        "    # Should return the sum of its two arguments.\n"
        "    return a\n"
    ),
    "run_checks.py": (
        "from app.calculator import add\n\n"
        "result = add(2, 3)\n"
        "assert result == 5, f'add(2, 3) returned {result}, expected 5'\n"
        "print('All checks passed.')\n"
    ),
}

QUESTION = (
    "Running 'python3 run_checks.py' in this project fails. Find the cause, fix "
    "the source file responsible, and then run the check again with the shell "
    "tool to confirm it passes. Report what the bug was and the final check output."
)


def build_fixture(root: Path) -> None:
    """Write the sample project tree under root."""
    for relative, content in FIXTURE_FILES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def on_bounded_run(event: BoundedRunEvent) -> None:
    """Print one child run: which tool ran, at which level, and how it ended."""
    achieved = event.achieved.value if event.achieved is not None else "none"
    print(
        f"[BoundedRunEvent] tool={event.tool_name} "
        f"required={event.required.value} achieved={achieved} "
        f"outcome={event.outcome.value} duration={event.duration_seconds:.2f}s"
    )


def on_tool_call(event: ToolCallPostEvent) -> None:
    """Print one tool call: the tool, its input, and what it returned, whole."""
    status = "ok" if event.succeeded else "failed"
    print(
        f"  [{event.tool_name}] {event.tool_input!r} -> {status}:\n{event.observation}"
    )


async def main():
    print(
        f"Loading {MODEL_NAME} via the HuggingFaceAdapter (first run downloads weights)..."
    )
    llm = HuggingFaceAdapter(MODEL_NAME, max_new_tokens=512)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        build_fixture(root)

        # One capability bag for the whole executor: the tools by capability,
        # the agent's own model, and exactly the project root. The shell's
        # isolated child writes every path the bag grants, so the bag holds
        # only the root.
        model_id = llm.describe_config().model_name
        bag = CapabilityBag.from_grants(
            tools=["shell.exec", "filesystem.read", "filesystem.write"],
            path_prefixes=[str(root)],
            models=[model_id],
        )
        security = BasicSecurityManager.from_settings(settings)
        print(
            "Isolation levels this host reaches: "
            + ", ".join(sorted(level.value for level in security.available_levels()))
        )
        # The live settings object is the configured seam the shell's isolated
        # child reads, so the timeout is set there, validated as ShellLimits.
        settings.limits.shell = ShellLimits.model_validate(
            {
                **settings.limits.shell.model_dump(),
                "default_timeout_seconds": SHELL_TIMEOUT_SECONDS,
            }
        )
        print(f"Each shell call runs under a {SHELL_TIMEOUT_SECONDS} s timeout.")
        print(CONFINEMENT_CLAIM)

        registry = ToolRegistry()
        tools = (
            ListDirTool(root),
            GlobTool(root),
            GrepTool(root),
            ReadFileTool(root),
            EditFileTool(root),
            WriteFileTool(root),
            ShellTool(root),
        )
        for tool in tools:
            registry.register_tool(tool)
        print(f"Tools: {list(registry.get_all_tools())}")
        print(f"Project root: {root}\n")

        executor = ToolExecutor(registry, security.with_capability_bag(bag))

        planner = SimpleReActPlanner(llm, registry)
        planner.prompt_builder.role_definition = RoleDefinition(
            "You are a software-fixing agent working inside one project directory. "
            "You answer by navigating and changing files, then running the project "
            "to verify. Work one step at a time: grep or read to locate the bug, "
            "edit_file to fix it (old_string must be unique), then use the shell "
            "tool to run the check. Do not guess paths you have not seen."
        )

        agent = SimpleAgent(
            llm=llm,
            planner=planner,
            tool_executor=executor,
            memory=WorkingMemory(),
            max_steps=30,
        )
        agent.events.subscribe(BoundedRunEvent, on_bounded_run)
        agent.events.subscribe(ToolCallPostEvent, on_tool_call)
        # The shell calls, kept so the last check's output can be shown whole.
        shell_calls: list = []

        def keep_shell_call(event: ToolCallPostEvent) -> None:
            if event.tool_name == "shell":
                shell_calls.append(event)

        agent.events.subscribe(ToolCallPostEvent, keep_shell_call)

        print(f"Task:\n  {QUESTION}\n")
        print("Running sandboxed edit-run loop...\n")
        answer = await agent.arun(QUESTION)
        print("=" * 60)
        print("Agent answer:")
        print(answer)
        print("=" * 60)

        if shell_calls:
            last = shell_calls[-1]
            print(f"\nLast shell call: {last.tool_input!r}")
            print("Its output, as the tool returned it:")
            print(last.observation)
        else:
            print("\nThe agent never ran the shell tool.")

        # Show the ground truth so a human watching can confirm the agent really
        # changed the file on disk, not just claimed to.
        print("\nFinal app/calculator.py on disk:")
        print((root / "app" / "calculator.py").read_text())


if __name__ == "__main__":
    asyncio.run(main())
