# demo_faiss_rag_from_readme.py
"""
FAISS RAG with cross-encoder re-ranking and the full ReAct loop.

This mirrors demo_rag_from_documents.py, but chunks README.md with
DocumentProcessor, stores and retrieves the chunks with FaissVectorStore,
re-ranks every retrieval with a CrossEncoder, and runs the full ReAct agent
loop over the re-ranked search tool. Each knowledge-base search the agent
makes is printed with its query and the [S#] markers it returned. Before the
store directory is removed, its listing is printed: index.faiss and
mapping.json, the store's only files.
"""

import asyncio
import logging
import re
import shutil
from pathlib import Path

from sentence_transformers import CrossEncoder

from fairlib import (
    HuggingFaceAdapter,
    LongTermMemory,
    RAGQueryTool,
    ReActPlanner,
    RoleDefinition,
    SentenceTransformerEmbedder,
    SimpleAgent,
    SimpleRetriever,
    ToolCallPostEvent,
    ToolExecutor,
    ToolRegistry,
    WorkingMemory,
    settings,
)
from fairlib.modules.memory.retriever_rerank import CrossEncoderRerankingRetriever
from fairlib.modules.memory.vector_faiss import FaissVectorStore
from fairlib.utils.document_processor import DocumentProcessor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("demo_faiss_rag_from_documents")


def on_tool_call(event: ToolCallPostEvent) -> None:
    """Print each knowledge-base search: the query and the markers it returned.

    A retrieved passage starts with its marker alone on a line, so only those
    lines count; the cite-only instruction's example marker does not.
    """
    status = "ok" if event.succeeded else "failed"
    numbers = sorted(
        {int(n) for n in re.findall(r"^\[S(\d+)\]$", event.observation or "", re.M)}
    )
    markers = " ".join(f"[S{n}]" for n in numbers)
    print(f"  [{event.tool_name}] {event.tool_input!r} -> {status}: {markers}")


async def main():
    """Set up and run the FAISS + ReRank RAG agent demonstration."""

    logger.info(
        "Initializing FAISS RAG components with DocumentProcessor + Cross-Encoder re-ranking..."
    )

    rag_cfg = getattr(settings, "rag_system", None)

    # Paths
    index_dir = Path(
        getattr(getattr(rag_cfg, "paths", None), "vector_store_dir", "out/vector_store")
    ).resolve()
    index_dir.mkdir(parents=True, exist_ok=True)

    # Models
    embed_model = getattr(
        getattr(rag_cfg, "embeddings", None),
        "embedding_model",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    cross_model = getattr(
        getattr(rag_cfg, "embeddings", None),
        "cross_encoder_model",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    )

    # Retrieval params
    use_gpu = getattr(getattr(rag_cfg, "vector_store", None), "use_gpu", False)
    batch_size = getattr(getattr(rag_cfg, "embeddings", None), "batch_size", 128)
    pool_multiplier = getattr(getattr(rag_cfg, "retrieval", None), "pool_multiplier", 5)
    max_initial_docs = getattr(
        getattr(rag_cfg, "retrieval", None), "max_initial_retrieval_docs", 50
    )
    top_k = 5
    rerank_k = min(top_k * pool_multiplier, max_initial_docs)

    try:
        llm = HuggingFaceAdapter("qwen25-7b")
        embedder = SentenceTransformerEmbedder(model_name=embed_model)
    except Exception as e:
        logger.critical(f"Failed to initialize LLM or embedder: {e}", exc_info=True)
        return

    vector_store = FaissVectorStore(
        embedder=embedder,
        index_dir=str(index_dir),
        use_gpu=use_gpu,
        normalize=True,
        batch_size=batch_size,
    )
    vector_store.load()
    long_term_memory = LongTermMemory(vector_store)

    base_retriever = SimpleRetriever(vector_store)
    cross_encoder = CrossEncoder(cross_model)
    retriever = CrossEncoderRerankingRetriever(
        base=base_retriever, cross_encoder=cross_encoder, rerank_k=rerank_k
    )

    # Load, chunk, and ingest README.md using DocumentProcessor (semantic split)
    readme_path = Path("README.md")
    if not readme_path.exists():
        logger.error(
            "README.md not found in the current directory. Please add one and re-run this demo."
        )
        return

    doc_proc = DocumentProcessor({"files_directory": str(readme_path.parent)})

    # process_file extracts and chunks the file; each returned Document is one
    # chunk carrying its source label.
    documents = doc_proc.process_file(str(readme_path))
    if not documents:
        logger.error("DocumentProcessor returned no documents from README.md.")
        return

    logger.info(
        f"README.md processed into {len(documents)} Document chunks. Ingesting into FAISS..."
    )
    long_term_memory.vector_store.add_documents(documents)
    logger.info("Document successfully ingested into FAISS-backed Long-Term Memory.")

    # Build the ReACT Agent
    # top_k passages per search: a README heading and its bullet list often
    # land in neighbouring chunks, so one search needs room for both.
    rag_tool = RAGQueryTool(retriever, top_k=top_k)
    tool_registry = ToolRegistry()
    tool_registry.register_tool(rag_tool)

    planner = ReActPlanner(llm, tool_registry)
    executor = ToolExecutor(tool_registry)
    working_memory = WorkingMemory()

    # The role reaches the model through the planner's prompt builder, the
    # seam every planner renders its system prompt from.
    planner.prompt_builder.role_definition = RoleDefinition(
        "You are a helpful AI assistant and an expert on the FAIR-LLM framework. "
        "You MUST use the 'search_knowledge_base' tool to answer questions about "
        "the framework, its principles, or its architecture. Your first "
        "action for every new question is a search_knowledge_base call made "
        "for that question, even when earlier passages look related; never "
        "answer a question before that search. "
        "Answer only from what the returned passages say, never from memory. "
        "A passage can stop partway through: when one announces a list or a "
        "definition that none of the passages contains, search again using "
        "the words it introduces before you answer."
    )
    rag_agent = SimpleAgent(llm, planner, executor, working_memory)
    rag_agent.events.subscribe(ToolCallPostEvent, on_tool_call)
    logger.info("RAG Agent created with re-ranked retriever.")

    questions = [
        "What are the core principles of the FAIR-LLM framework?",
        "What is the Model Abstraction Layer (MAL) and why is it important?",
        "How does the framework handle multi-agent collaboration?",
    ]

    for q in questions:
        print(f"\nYou: {q}")
        try:
            resp = await rag_agent.arun(q)
            print(f"Agent: {resp}")
        except Exception as e:
            logger.error(f"Agent error for question '{q}': {e}", exc_info=True)
            print("Agent: I encountered an error and couldn't process your request.")

    # add_documents persisted the store: the directory holds the FAISS index
    # and its JSON mapping (the texts and metadata), and no pickle.
    print(f"\nPersisted FAISS store {index_dir}:")
    for entry in sorted(index_dir.iterdir()):
        print(f"  {entry.name} ({entry.stat().st_size} bytes)")
    print(f"  mapping.pkl present: {(index_dir / 'mapping.pkl').exists()}")

    # remove created faiss directory
    try:
        if index_dir.exists() and index_dir.is_dir():
            shutil.rmtree(index_dir)
            logger.info(f"Cleaned up FAISS store directory: {index_dir}")
    except Exception as e:
        logger.warning(f"Could not remove FAISS store directory {index_dir}: {e}")


if __name__ == "__main__":
    # Ensure a dummy README.md exists for the demo to run out-of-the-box.
    if not Path("README.md").exists():
        Path("README.md").write_text(
            "# FAIR-LLM Framework\n"
            "FAIR-LLM is a Python framework for building modular agentic applications. "
            "Its fairlib.core.principles are being Flexible, Agnostic, and Interoperable. "
            "A key feature is the Model Abstraction Layer (MAL), which allows switching LLM providers easily. "
            "It also supports multi-agent collaboration through workers-as-tools fan-out."
        )
    asyncio.run(main())
