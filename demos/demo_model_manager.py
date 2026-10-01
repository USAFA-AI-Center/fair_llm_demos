# demo_model_manager.py
"""
One small app with three model roles, written once - then the deployment changes
and the application does not.

Without ModelManager, every lab app grows a provider ladder: import every
adapter, know each constructor's spelling, hardcode model names, and keep a
module-level cache so a local model is not loaded twice. Moving from a laptop
to the lab GPU box means editing that code. This demo deletes the ladder.

Roles are tutor, student, and judge. The application asks for roles; settings
YAML picks the provider and model. The same tutoring_review function runs
against a laptop file (all Ollama), a lab file (student moves to HuggingFace),
and a typo file that refuses before any model is contacted.

Local models come from FAIR_LLM_DEMO_MODEL (Ollama) and FAIR_LLM_DEMO_HF_MODEL
(HuggingFace), same as the other demos. No arguments required.

Invoke:

    python demos/demo_model_manager.py
"""

from __future__ import annotations

import asyncio
import difflib
import os
import sys
import tempfile
from pathlib import Path

from fairlib import (
    Message,
    ModelManager,
    load_settings,
)
from fairlib.core.errors import ConfigurationError
from fairlib.core.interfaces.llm import AbstractChatModel

OLLAMA_MODEL = os.environ.get("FAIR_LLM_DEMO_MODEL", "qwen3-vl-instruct-16k")
HF_MODEL = os.environ.get("FAIR_LLM_DEMO_HF_MODEL", "qwen25-7b")

ROLES = ("tutor", "student", "judge")


def _settings_yaml(
    *, student_provider: str, student_model: str, judge_kwargs: str = ""
) -> str:
    """AppSettings YAML: api_keys, default_model, and three role rows."""
    judge_block = (
        f"  judge:\n"
        f"    provider: ollama\n"
        f'    model_name: "{OLLAMA_MODEL}"\n'
        f"    temperature: 0.0\n"
        f"    max_tokens: 120\n"
        f"{judge_kwargs}"
    )
    return (
        "api_keys: {}\n"
        "default_model: tutor\n"
        "models:\n"
        f"  tutor:\n"
        f"    provider: ollama\n"
        f'    model_name: "{OLLAMA_MODEL}"\n'
        f"    temperature: 0.4\n"
        f"    max_tokens: 200\n"
        f"  student:\n"
        f"    provider: {student_provider}\n"
        f'    model_name: "{student_model}"\n'
        f"    temperature: 0.9\n"
        f"    max_tokens: 120\n"
        f"{judge_block}"
    )


def _write_settings(directory: Path, name: str, body: str) -> Path:
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path


def _print_diff(before: Path, after: Path) -> None:
    before_lines = before.read_text(encoding="utf-8").splitlines()
    after_lines = after.read_text(encoding="utf-8").splitlines()
    for line in difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile=before.name,
        tofile=after.name,
        lineterm="",
    ):
        print(line)


def _one_line(text: str) -> str:
    return " ".join(text.split())


async def ask(model: AbstractChatModel, system: str, user: str) -> str:
    """One chat turn: system plus user, return the assistant text."""
    reply = await model.ainvoke(
        [Message(role="system", content=system), Message(role="user", content=user)]
    )
    return reply.content


def _print_role_table(manager: ModelManager) -> None:
    for role in ROLES:
        model = manager.get_model(role)
        desc = model.describe_config()
        print(
            f"  {role:8} -> {desc.adapter:20} "
            f"model_name={desc.model_name!r} provider={desc.provider!r}"
        )


async def tutoring_review(manager: ModelManager, *, cache_role: str = "tutor") -> None:
    """The application. It names roles. It never names a provider, model, or adapter."""
    _print_role_table(manager)

    attempt = await ask(
        manager.get_model("student"),
        "You are a student who makes one mistake. Reply with one sentence only.",
        "Solve 3x + 5 = 20 in one sentence.",
    )
    print(f"  student: {_one_line(attempt)}")

    # default_model is tutor in every settings file; get_model() is the common form.
    tutor = manager.get_model()
    hint = await ask(
        tutor,
        "You are a tutor. Give one hint in one sentence, never the answer.",
        f"Student wrote: {attempt}",
    )
    print(f"  tutor  : {_one_line(hint)}")

    score = await ask(
        manager.get_model("judge"),
        "Grade the hint 1-5 for not revealing the answer. Reply with one sentence: the score and a short reason.",
        hint,
    )
    print(f"  judge  : {_one_line(score)}")

    first = manager.get_model(cache_role)
    again = manager.get_model(cache_role)
    if cache_role == "student":
        note = "(one copy of the weights however many agents ask for it)"
    else:
        note = "(same cached adapter object)"
    print(
        f"  cache  : second get_model({cache_role!r}) is the same object: "
        f"{again is first} {note}"
    )


async def _run_named(
    path: Path,
    *,
    intro: str,
    diff_from: Path | None = None,
    cache_role: str = "tutor",
) -> None:
    print(f"\n=== {path.name} ===")
    print(intro)
    if diff_from is None:
        print(path.read_text(encoding="utf-8").rstrip())
    else:
        _print_diff(diff_from, path)
    manager = ModelManager(load_settings(path))
    print(f"  aliases: {manager.list_models()}")
    await tutoring_review(manager, cache_role=cache_role)


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="fairlib_model_manager_demo_") as tmp:
        directory = Path(tmp)

        laptop = _write_settings(
            directory,
            "laptop.yml",
            _settings_yaml(student_provider="ollama", student_model=OLLAMA_MODEL),
        )
        await _run_named(
            laptop,
            intro=(
                "One application names roles only (tutor, student, judge). "
                "This laptop settings file wires all three to Ollama."
            ),
        )

        lab = _write_settings(
            directory,
            "lab.yml",
            _settings_yaml(student_provider="huggingface", student_model=HF_MODEL),
        )
        await _run_named(
            lab,
            intro=(
                "The deployment moves to the lab GPU box; the application is not "
                "edited, one settings row is:"
            ),
            diff_from=laptop,
            cache_role="student",
        )

        typo = _write_settings(
            directory,
            "typo.yml",
            _settings_yaml(
                student_provider="ollama",
                student_model=OLLAMA_MODEL,
                judge_kwargs='    kwargs:\n      hots: "http://localhost:11434"\n',
            ),
        )
        print("\n=== typo.yml ===")
        print(
            "A typo in one row is refused by role name before any model is contacted:"
        )
        _print_diff(laptop, typo)
        typo_manager = ModelManager(load_settings(typo))
        try:
            typo_manager.get_model("judge")
        except ConfigurationError as exc:
            print(f"  ConfigurationError: {exc}")
        student = typo_manager.get_model("student")
        tutor = typo_manager.get_model()
        print(
            f"  the other roles still resolve: "
            f"student={type(student).__name__}, tutor={type(tutor).__name__} "
            f"(only judge refuses)"
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except ConfigurationError as exc:
        print(f"ConfigurationError: {exc}", file=sys.stderr)
        sys.exit(1)
