# demos/demo_faiss_rag_from_readme.py
"""
================================================================================
        FAISS RAG Demonstration (Async, Always Re-Rank, Full ReACT Loop)
================================================================================

This mirrors demo_rag_from_documents.py, but:
  • Uses DocumentProcessor (split_text_semantic) for chunking
  • Stores/retrieves with FaissVectorStore (persistent)
  • Always re-ranks via CrossEncoder
  • Runs the full ReACT agent loop
"""
import os
import asyncio
import logging
from pathlib import Path
import shutil

from fairlib import (
    settings,
    OpenAIAdapter,
    ToolRegistry,
    ToolExecutor,
    WorkingMemory,
    LongTermMemory,
    ReActPlanner,
    SimpleAgent,
    SentenceTransformerEmbedder,
    SimpleRetriever,
    KnowledgeBaseQueryTool
)

from fairlib.utils.document_processor import DocumentProcessor
from fairlib.modules.memory.vector_faiss import FaissVectorStore
from fairlib.modules.memory.retriever_rerank import CrossEncoderRerankingRetriever
from sentence_transformers import CrossEncoder

from dotenv import load_dotenv
load_dotenv()

settings.api_keys.openai_api_key = os.getenv("OPENAI_API_KEY")
settings.api_keys.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("demo_faiss_rag_from_documents")

async def main():
    """Set up and run the FAISS + ReRank RAG agent demonstration."""

    logger.info("Initializing FAISS RAG components with DocumentProcessor + Cross-Encoder re-ranking...")

    rag_cfg = getattr(settings, "rag_system", None)

    # Paths
    index_dir = Path(getattr(getattr(rag_cfg, "paths", None), "vector_store_dir", "out/vector_store")).resolve()
    index_dir.mkdir(parents=True, exist_ok=True)

    # Models
    embed_model = getattr(getattr(rag_cfg, "embeddings", None), "embedding_model",
                          "sentence-transformers/all-MiniLM-L6-v2")
    cross_model = getattr(getattr(rag_cfg, "embeddings", None), "cross_encoder_model",
                          "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Retrieval params
    use_gpu = getattr(getattr(rag_cfg, "vector_store", None), "use_gpu", False)
    batch_size = getattr(getattr(rag_cfg, "embeddings", None), "batch_size", 128)
    pool_multiplier = getattr(getattr(rag_cfg, "retrieval", None), "pool_multiplier", 5)
    max_initial_docs = getattr(getattr(rag_cfg, "retrieval", None), "max_initial_retrieval_docs", 50)
    top_k = 5
    rerank_k = min(top_k * pool_multiplier, max_initial_docs)

    try:
        llm = OpenAIAdapter(
            api_key=getattr(settings.api_keys, "openai_api_key"),
            model_name=settings.models.get("openai_gpt4", {"model_name": "gpt-4o"}).model_name
        )
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
        base=base_retriever,
        cross_encoder=cross_encoder,
        rerank_k=rerank_k
    )

    # Load, chunk, and ingest README.md using DocumentProcessor (semantic split)
    readme_path = Path("README.md")
    if not readme_path.exists():
        logger.error("README.md not found in the current directory. Please add one and re-run this demo.")
        return

    doc_proc = DocumentProcessor({"files_directory": str(readme_path.parent)})

    # Process a single file -> DP handles extraction + split_text_semantic internally
    # Important note: document processor now returns a Document object, instead of chunks and metadata
    documents = doc_proc.process_file(str(readme_path))
    if not documents:
        logger.error("DocumentProcessor returned no documents from README.md.")
        return
    
    logger.info(f"README.md processed into {len(documents)} Document chunks. Ingesting into FAISS...")
    long_term_memory.vector_store.add_documents(documents)
    logger.info("Document successfully ingested into FAISS-backed Long-Term Memory.")

    # Build the ReACT Agent 
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
    logger.info("RAG Agent created with re-ranked retriever.")

    questions = [
        "What are the core principles of the FAIR-LLM framework?",
        "What is the Model Abstraction Layer (MAL) and why is it important?",
        "How does the framework handle multi-agent collaboration?"
    ]

    for q in questions:
        print(f"\n👤 You: {q}")
        try:
            resp = await rag_agent.arun(q)
            print(f"🤖 Agent: {resp}")
        except Exception as e:
            logger.error(f"Agent error for question '{q}': {e}", exc_info=True)
            print("🤖 Agent: I encountered an error and couldn't process your request.")

    # remove created faiss directory
    try:
        if index_dir.exists() and index_dir.is_dir():
            shutil.rmtree(index_dir)
            logger.info(f"Cleaned up FAISS store directory: {index_dir}")
    except Exception as e:
        logger.warning(f"Could not remove FAISS store directory {index_dir}: {e}")

if __name__ == "__main__":
    asyncio.run(main())