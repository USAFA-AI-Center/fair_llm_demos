# demo_rag_from_documents.py
"""
Retrieval-Augmented Generation (RAG) demonstration.

This script is a hands-on tutorial for RAG, the pattern of giving an LLM
project-specific knowledge at query time. It uses the core FAIR-LLM components
for ingestion, retrieval, and agent-driven generation.

Imagine building an assistant that knows your README and docs. The base model
does not; RAG loads documents, chunks them, embeds them, stores vectors, and
retrieves relevant passages when the user asks a question.

The workflow covers chunking strategies through DocumentProcessor (sentence,
fixed-size, semantic), metadata filters on SimpleRetriever, incremental
add_documents on vector stores, loading README.md, embedding with
SentenceTransformerEmbedder, storing in ChromaDBVectorStore, retrieving with
KnowledgeBaseQueryTool, and generating answers with a local HuggingFace model.

Run: PYTHONPATH=. python demos/demo_rag_from_documents.py
Requires chromadb, sentence-transformers, and a GPU for the agent section.
Set FAIR_LLM_DEMO_MODEL to override the default model.
"""
import asyncio
import os
import logging
from pathlib import Path

try:
    import chromadb
    CHROMADB_LOADED = True
except ImportError:
    print("chromadb not found. To run this RAG demo, please run 'pip install chromadb'")
    chromadb = None
    CHROMADB_LOADED = False

from fairlib.utils.document_processor import DocumentProcessor

from fairlib import (
    Document,
    HuggingFaceAdapter,
    InMemoryVectorStore,
    KnowledgeBaseQueryTool,
    LongTermMemory,
    ChromaDBVectorStore,
    ReActPlanner,
    SentenceTransformerEmbedder,
    SimpleAgent,
    SimpleRetriever,
    ToolExecutor,
    ToolRegistry,
    WorkingMemory,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("FAIR_LLM_DEMO_MODEL", "qwen25-7b")


def demo_chunking_filters_incremental() -> None:
    """Part 1: chunking strategies, metadata filters, incremental adds."""
    print("\n=== Part 1: Chunking, metadata filters, incremental indexing ===")

    processor = DocumentProcessor(
        {
            "files_directory": ".",
            "supported_extensions": {".txt"},
            "max_chunk_chars": 45,
            "chunking_strategy": "sentence",
        }
    )
    sample = "Algebra studies symbols. Geometry studies shapes. Poetry studies language."
    print("Sentence chunks:", processor.split_text(sample))
    print("Fixed chunks:", processor.split_text(sample, strategy="fixed", max_chars=20))
    print("Semantic chunks:", processor.split_text(sample, strategy="semantic", max_chars=45))

    store = InMemoryVectorStore()
    store.add_documents(
        [
            Document("Algebra context: variables and equations.", {"topic": "math"}),
            Document("Geometry context: shapes and proofs.", {"topic": "math"}),
        ]
    )
    # Incremental update: append without rebuilding the store.
    store.add_documents([Document("Poetry context: meter and imagery.", {"topic": "literature"})])

    retriever = SimpleRetriever(store)
    math_hits = retriever.retrieve("context", top_k=3, metadata_filter={"topic": "math"})
    print("Math-only retrieval:", [doc.page_content for doc in math_hits])
    hybrid = retriever.retrieve_hybrid("context", preload_k=1, explore_k=3)
    print("Hybrid preload:", hybrid.preloaded[0])


async def demo_chroma_rag_agent() -> None:
    """Part 2: full Chroma-backed RAG agent with real HuggingFace inference."""
    print("\n=== Part 2: Chroma-backed RAG agent (real inference) ===")

    if not CHROMADB_LOADED:
        logger.critical("ChromaDB library is required for this demo but is not installed. Exiting.")
        return

    logger.info("Initializing RAG components...")
    try:
        llm = HuggingFaceAdapter(MODEL_NAME, max_new_tokens=512)
        embedder = SentenceTransformerEmbedder()
        vector_store = ChromaDBVectorStore(
            client=chromadb.Client(),
            collection_name="readme_rag",
            embedder=embedder,
        )
        long_term_memory = LongTermMemory(vector_store)
        retriever = SimpleRetriever(vector_store)
    except Exception as exc:
        logger.critical(f"Failed to initialize RAG components: {exc}", exc_info=True)
        return

    readme_path = Path("README.md")
    if not readme_path.exists():
        logger.error("README.md not found in the current directory.")
        return

    doc_proc = DocumentProcessor({"files_directory": str(readme_path.parent)})
    document = doc_proc.process_file(str(readme_path))
    if not document:
        logger.error("DocumentProcessor returned no documents from README.md.")
        return

    # Use DocumentProcessor chunking instead of a local splitter helper.
    chunks = doc_proc.split_text(document[0].page_content, strategy="semantic")
    logger.info("Document split into %d semantic chunks.", len(chunks))
    long_term_memory.add_document(chunks)
    logger.info("Document successfully ingested into long-term memory.")

    knowledge_tool = KnowledgeBaseQueryTool(retriever)
    tool_registry = ToolRegistry()
    tool_registry.register_tool(knowledge_tool)

    planner = ReActPlanner(llm, tool_registry)
    executor = ToolExecutor(tool_registry)
    working_memory = WorkingMemory()

    rag_agent = SimpleAgent(llm, planner, executor, working_memory)
    rag_agent.role_description = (
        "You are a helpful AI assistant and an expert on the FAIR-LLM framework. "
        "You MUST use the 'course_knowledge_query' tool to answer questions about "
        "the framework, its principles, or its architecture."
    )
    logger.info("RAG agent created.")

    questions = [
        "What are the core principles of the FAIR-LLM framework?",
        "What is the Model Abstraction Layer (MAL) and why is it important?",
    ]
    for question in questions:
        print(f"\nYou: {question}")
        try:
            response = await rag_agent.arun(question)
            print(f"Agent: {response}")
        except Exception as exc:
            logger.error("Agent run failed for %r: %s", question, exc, exc_info=True)
            print("Agent: I encountered an error and could not process your request.")


async def main() -> None:
    demo_chunking_filters_incremental()
    await demo_chroma_rag_agent()


if __name__ == "__main__":
    if not Path("README.md").exists():
        Path("README.md").write_text(
            "# FAIR-LLM Framework\n"
            "FAIR-LLM is a Python framework for building modular agentic applications. "
            "Its core principles are being Flexible, Agnostic, and Interoperable. "
            "A key feature is the Model Abstraction Layer (MAL), which allows switching LLM providers easily. "
            "It also supports multi-agent collaboration through a HierarchicalAgentRunner."
        )
    asyncio.run(main())
