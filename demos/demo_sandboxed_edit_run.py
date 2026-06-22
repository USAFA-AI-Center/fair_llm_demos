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
every command through an injected security manager (here BasicSecurityManager)
before it executes, with the root as its working directory.
"""

import asyncio
import tempfile
from pathlib import Path

from fairlib import (
    HuggingFaceAdapter,
    ToolRegistry,
    ToolExecutor,
    WorkingMemory,
    SimpleAgent,
    SimpleReActPlanner,
    RoleDefinition,
    GlobTool,
    GrepTool,
    ListDirTool,
    ReadFileTool,
    EditFileTool,
    WriteFileTool,
    ShellTool,
)
from fairlib.modules.security.basic_security_manager import BasicSecurityManager

MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"


# A tiny project whose check script fails because add() subtracts. The agent is
# not told where the bug is; it has to find calculator.py, fix the operator, and
# run the check to confirm. The fix is a single unique-substring edit.
FIXTURE_FILES = {
    "README.md": (
        "# Calc\n\n"
        "A toy math package. Run `python3 run_checks.py` to verify it.\n"
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


async def main():
    print(f"Loading {MODEL_NAME} via the HuggingFaceAdapter (first run downloads weights)...")
    llm = HuggingFaceAdapter(MODEL_NAME, max_new_tokens=512)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_fixture(root)

        # The shell tool is the only one that takes a security manager; it
        # screens every command before running it. All tools, file and shell,
        # are confined to the granted root.
        security = BasicSecurityManager()
        registry = ToolRegistry()
        tools = (
            ListDirTool(root),
            GlobTool(root),
            GrepTool(root),
            ReadFileTool(root),
            EditFileTool(root),
            WriteFileTool(root),
            ShellTool(root, security),
        )
        for tool in tools:
            registry.register_tool(tool)
        print(f"Tools: {list(registry.get_all_tools())}")
        print(f"Project root: {root}\n")

        executor = ToolExecutor(registry)

        planner = SimpleReActPlanner(llm, registry)
        planner.prompt_builder.role_definition = RoleDefinition(
            "You are a software-fixing agent working inside one project directory. "
            "You answer by navigating and changing files, then running the project "
            "to verify. Work one step at a time: grep or read to locate the bug, "
            "edit_file to fix it (old_string must be unique), then use the shell "
            "tool to run the check. Do not guess paths you have not seen. Pass each "
            "tool a single bare value as tool_input, never a 'field: value' pair."
        )

        agent = SimpleAgent(
            llm=llm,
            planner=planner,
            tool_executor=executor,
            memory=WorkingMemory(),
            max_steps=15,
        )

        print(f"Task:\n  {QUESTION}\n")
        print("Running sandboxed edit-run loop...\n")
        answer = await agent.arun(QUESTION)
        print("=" * 60)
        print("Agent answer:")
        print(answer)
        print("=" * 60)

        # Show the ground truth so a human watching can confirm the agent really
        # changed the file on disk, not just claimed to.
        print("\nFinal app/calculator.py on disk:")
        print((root / "app" / "calculator.py").read_text())


if __name__ == "__main__":
    asyncio.run(main())
