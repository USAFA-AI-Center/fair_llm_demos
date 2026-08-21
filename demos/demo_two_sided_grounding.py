"""Two-sided RAG grounding: cite-only injection, then citation verification.

Before generation, retrieved hits are numbered [S1]..[Sn] and the model is
told to cite only those markers. After generation, CitationVerifier checks
each extracted citation and reports a fabrication rate that excludes
could_not_check and source_unreachable outcomes.

This demo drives a real local model via HuggingFaceAdapter. Set
FAIR_LLM_DEMO_MODEL to override the default (dolphin3-qwen25-3b). The first
run may download weights.

Run:
    PYTHONPATH=. python demos/demo_two_sided_grounding.py
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

from fairlib import (
    CitationReport,
    CitationVerifier,
    Document,
    GroundedContextBuilder,
    HuggingFaceAdapter,
    Message,
)

MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "dolphin3-qwen25-3b")


def _print_report(title: str, report: CitationReport) -> None:
    rate: Optional[float] = report.fabrication_rate
    rate_text = "None (nothing checkable)" if rate is None else f"{rate:.2f}"
    print(f"\n{title}")
    print(
        f"  verified={report.verified_count} not_found={report.not_found_count} "
        f"could_not_check={report.could_not_check_count} "
        f"source_unreachable={report.source_unreachable_count} "
        f"fabrication_rate={rate_text}"
    )
    for check in report.checks:
        print(f"  - {check.citation!r}: {check.status.value} ({check.reason})")


async def demo_live_generation() -> None:
    """Ground README-style hits, generate with a local model, then verify."""
    print("=== Two-sided grounding: live generation + citation verification ===")
    print(f"Loading {MODEL_NAME} via HuggingFaceAdapter...")

    context = GroundedContextBuilder().build(
        [
            Document(
                "FAIR-LLM core principles are Flexible, Agnostic, Interoperable, "
                "and Reasoning.",
                {"source": "README.md"},
            ),
            Document(
                "The Model Abstraction Layer lets callers swap LLM providers "
                "without rewriting agent code. See https://example.edu/mal",
                {"topic": "mal"},
            ),
        ]
    )
    print("\nInjected prompt block:\n")
    print(context.prompt_block)

    prompt = (
        context.prompt_block
        + "\n\nQuestion: What are the core principles of FAIR-LLM?\n"
        "Answer with citations using only the [S#] markers above."
    )
    llm = HuggingFaceAdapter(MODEL_NAME, max_new_tokens=256)
    answer = await llm.ainvoke([Message(role="user", content=prompt)])
    text = answer.content if hasattr(answer, "content") else str(answer)
    print("\nModel answer:\n", text)
    report = CitationVerifier().verify(text, context)
    _print_report("Live answer verification", report)


def main() -> None:
    asyncio.run(demo_live_generation())


if __name__ == "__main__":
    main()
