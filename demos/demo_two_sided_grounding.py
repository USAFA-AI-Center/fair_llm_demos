"""Two-sided RAG grounding: cite-only retrieval, then citation verification.

The agent below has one tool, RAGQueryTool, over a two-document knowledge
base. Before generation the tool numbers every retrieved passage [S1]..[Sn]
and tells the model to cite only those markers; after generation
CitationVerifier checks each citation in the answer against the sources
the tool actually retrieved and reports a fabrication rate.

The whole path runs through a SimpleAgent so the retrieval, the citation
instruction, and the answer are the agent's own, not a hand-built prompt.
The agent answers two questions, one run each. The numbering is run-scoped:
the agent starts its tools at the start of every run, so each run's sources
begin at [S1], and each answer is verified against the sources of its own
run, read right after that run.

Requires a local model; defaults to HuggingFaceAdapter("qwen25-7b").
Set FAIR_LLM_DEMO_MODEL to override. The first run may download weights.

Run:
    PYTHONPATH=. python demos/demo_two_sided_grounding.py
"""

from __future__ import annotations

import asyncio
import os
from typing import List, Optional

from fairlib import (
    AbstractRetriever,
    CitationReport,
    CitationVerifier,
    Document,
    HuggingFaceAdapter,
    RAGQueryTool,
    RoleDefinition,
    SimpleAgent,
    SimpleReActPlanner,
    ToolExecutor,
    ToolRegistry,
    WorkingMemory,
)

MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "qwen25-7b")

KNOWLEDGE_BASE = [
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


class InMemoryRetriever(AbstractRetriever):
    """The smallest possible knowledge base: every query returns the same passages.

    A real deployment puts a vector store behind this interface (see
    demo_faiss_rag_from_readme.py); the tool and the verifier do not care.
    """

    def __init__(self, documents: List[Document]) -> None:
        self._documents = documents

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Document]:
        return self._documents[:top_k]

    async def aretrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Document]:
        return self.retrieve(query, top_k)


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


async def main() -> None:
    print("=== Two-sided grounding: agent retrieval + citation verification ===")
    print(f"Loading {MODEL_NAME} via HuggingFaceAdapter...")
    llm = HuggingFaceAdapter(MODEL_NAME, max_new_tokens=256)

    # The agent's anatomy: one RAG tool, an executor, memory, and a planner.
    rag_tool = RAGQueryTool(InMemoryRetriever(KNOWLEDGE_BASE), top_k=2)
    tool_registry = ToolRegistry()
    tool_registry.register_tool(rag_tool)
    planner = SimpleReActPlanner(llm, tool_registry)
    planner.prompt_builder.role_definition = RoleDefinition(
        "You answer questions about the FAIR-LLM framework. You MUST call the "
        "'search_knowledge_base' tool before answering, and your final answer "
        "must cite the passages it returns using only their [S#] markers."
    )
    # The questions are unrelated, so the agent is stateless: each run starts
    # from an empty memory, and no earlier run's [S1] can be cited by mistake.
    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=ToolExecutor(tool_registry),
        memory=WorkingMemory(),
        max_steps=6,
        stateless=True,
    )

    questions = [
        "What are the core principles of FAIR-LLM?",
        "What does the Model Abstraction Layer let callers do?",
    ]
    for question in questions:
        print(f"\nYou: {question}")
        answer = await agent.arun(question)
        print(f"\nAgent: {answer}")

        # The tool keeps the sources it handed the model during this run,
        # readable until the next run starts, so the verifier checks the
        # answer against exactly those.
        context = rag_tool.grounded_context
        if context is None:
            print("\nThe agent answered without calling the tool; nothing to verify.")
            continue
        print("\nSources the tool retrieved in this run:")
        for source in context.sources:
            print(f"  {source.marker} {source.content}")
        _print_report(
            "Citation verification", CitationVerifier().verify(answer, context)
        )


if __name__ == "__main__":
    asyncio.run(main())
