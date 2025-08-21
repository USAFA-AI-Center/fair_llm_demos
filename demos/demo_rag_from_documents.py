# demo_rag_from_documents.py
"""
================================================================================
        Retrieval-Augmented Generation (RAG) Demonstration (Refactored)
================================================================================

**Purpose:**
This script provides a complete, hands-on tutorial for Retrieval-Augmented
Generation (RAG), a fundamental pattern for creating knowledgeable AI agents.
This refactored version demonstrates how to build a RAG pipeline using the
core, reusable components of the FAIR-LLM framework.

**The Use Case:**
Imagine you want to build an AI assistant that is an expert on YOUR project's
documentation. The base language model has no knowledge of your specific code or
README files. RAG is the process of providing the LLM with that knowledge at the
moment it's needed.

**The Workflow:**
1.  **Load:** We will use the framework's `DocumentLoader` to load a document
    (our project's README.md).
2.  **Split:** We'll break the document into smaller, manageable chunks.
3.  **Embed:** We'll use the framework's `SentenceTransformerEmbedder` to convert
    each chunk into a numerical vector.
4.  **Store:** We will store these vectors in a `ChromaDBVectorStore`.
5.  **Retrieve & Augment:** When a user asks a question, the agent will use the
    framework's `KnowledgeBaseQueryTool` to search the vector store for the most
    relevant document chunks and use them as context.
6.  **Generate:** The LLM, now equipped with the relevant context, generates a
    well-informed answer.

This script shows how to assemble these modular components into a powerful,
knowledge-grounded agent.
"""
import asyncio
import logging
from pathlib import Path
import os

try:
    import chromadb
    CHROMADB_LOADED=True
except ImportError:
    print("chromadb not found. To run this RAG demo, please run 'pip install chromadb'")
    chromadb = None
    CHROMADB_LOADED = False


# --- Step 1: Import from the new fairlib.utils.module and the central fairlib API ---
# We now use the DocumentProcessor from our shared utilities.
from fairlib.utils.document_processor import DocumentProcessor

# All other components are imported from the central `fairlib` API, promoting
# consistency and ease of use.
from fairlib import (
    settings,
    OpenAIAdapter,
    ToolRegistry,
    ToolExecutor,
    WorkingMemory,
    LongTermMemory,
    ChromaDBVectorStore,
    ReActPlanner,
    SimpleAgent,
    SentenceTransformerEmbedder,
    SimpleRetriever,
    KnowledgeBaseQueryTool 
)

from dotenv import load_dotenv
load_dotenv()

settings.api_keys.openai_api_key = os.getenv("OPENAI_API_KEY")
settings.api_keys.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Configure logging for the demo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# A simple text splitter for the demo. In a more complex application, this
# could be a more sophisticated utility, perhaps from a library like LangChain.
def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> list[str]:
    """Splits a long text into smaller, overlapping chunks."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


async def main():
    """Main function to set up and run the RAG agent demonstration."""

    # --- Step 2: Initialize Core RAG and Framework Components ---
    logger.info("Initializing RAG components...")

    # Add a check to ensure chromadb was imported successfully before proceeding
    if not CHROMADB_LOADED:
        logger.critical("ChromaDB library is required for this demo but is not installed. Exiting.")
        return

    try:
        llm = OpenAIAdapter(
            api_key=settings.api_keys.openai_api_key,
            model_name=settings.models.get("openai_gpt4", {"model_name": "gpt-4o"}).model_name
        )
        embedder = SentenceTransformerEmbedder()
        
        # Using an in-memory ChromaDB client for this demonstration.
        # For persistence, a server-based client would be used.
        vector_store = ChromaDBVectorStore(
            client=chromadb.Client(),
            collection_name="readme_rag",
            embedder=embedder
        )
        long_term_memory = LongTermMemory(vector_store)
        retriever = SimpleRetriever(vector_store)
        
    except Exception as e:
        logger.critical(f"Failed to initialize fairlib.core.components: {e}", exc_info=True)
        return

    # --- Step 3: Load, Split, and Ingest the Document into LongTermMemory ---
    logger.info("Loading and ingesting document into Long-Term Memory...")
    
    # Use the robust DocumentLoader from our fairlib.utils.module.
    readme_path = Path("README.md")
    if not readme_path.exists():
        logger.error(f"README.md not found in the current directory. Please create one to run this demo.")
        return
        
    doc_proc = DocumentProcessor({"files_directory": str(readme_path.parent)})

    # Process a single file -> DP handles extraction + split_text_semantic internally
    # Important note: document processor now returns a Document object, instead of chunks and metadata
    document = doc_proc.process_file(str(readme_path))
    if not document:
        logger.error("DocumentProcessor returned no documents from README.md.")
        return

    # Split the document into smaller chunks for effective retrieval.
    chunks = split_text(document[0].page_content)
    logger.info(f"Document split into {len(chunks)} chunks.")

    # Add the document chunks to the long-term memory (vector store).
    long_term_memory.vector_store.add_documents(chunks)
    logger.info("✅ Document successfully ingested into Long-Term Memory.")

    # --- Step 4: Create the RAG-Powered Agent ---
    logger.info("\nBuilding the RAG agent...")
    
    # The agent is given the official `KnowledgeBaseQueryTool` to access its new knowledge.
    # This is the same tool used by the `FactChecker` in our autograders.
    knowledge_tool = KnowledgeBaseQueryTool(retriever)
    
    tool_registry = ToolRegistry()
    tool_registry.register_tool(knowledge_tool)
    
    planner = ReActPlanner(llm, tool_registry)
    executor = ToolExecutor(tool_registry)
    working_memory = WorkingMemory()
    
    rag_agent = SimpleAgent(llm, planner, executor, working_memory)
    # This role description is a crucial part of the prompt, guiding the agent
    # to use its tool correctly.
    rag_agent.role_description = (
        "You are a helpful AI assistant and an expert on the FAIR-LLM framework. "
        "You MUST use the 'course_knowledge_query' tool to answer questions about "
        "the framework, its principles, or its architecture."
    )
    logger.info("✅ RAG Agent created.")

    # --- Step 5: Interact with the Agent ---
    logger.info("\n--- Starting Interaction with RAG Agent ---")
    questions = [
        "What are the fairlib.core.principles of the FAIR-LLM framework?",
        "What's the biggest advantage that comes from using this framework for a profession looking to build his/her AI engineering skills?",
        "What is the Model Abstraction Layer (MAL) and why is it important?",
        "How does the framework handle multi-agent collaboration?"
    ]

    for question in questions:
        print(f"\n👤 You: {question}")
        try:
            response = await rag_agent.arun(question)
            print(f"🤖 Agent: {response}")
        except Exception as e:
            logger.error(f"An error occurred during the agent run for question '{question}': {e}", exc_info=True)
            print("🤖 Agent: I encountered an error and couldn't process your request.")

if __name__ == "__main__":
    # Ensure a dummy README.md exists for the demo to run out-of-the-box.
    if not Path("README.md").exists():
        Path("README.md").write_text(
            "# FAIR-LLM Framework\n"
            "FAIR-LLM is a Python framework for building modular agentic applications. "
            "Its fairlib.core.principles are being Flexible, Agnostic, and Interoperable. "
            "A key feature is the Model Abstraction Layer (MAL), which allows switching LLM providers easily. "
            "It also supports multi-agent collaboration through a HierarchicalAgentRunner."
        )
    asyncio.run(main())
