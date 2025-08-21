# demo_committee_of_agents_essay_autograder.py

"""
================================================================================
            Multi-Agent AI Essay Autograder with RAG
================================================================================

**Purpose:**
This script provides a powerful, automated tool for grading student essays,
designed to save educators time and increase grading consistency. It can
process a batch of essays in `.docx`, `.pdf`, or `.txt` format, evaluating
each one against a custom, user-provided rubric.

**Who is this for?**
This tool is for college-level educators, teaching assistants, or any instructor
who needs to grade a large number of subjective papers and wants to leverage
AI to streamline the process while maintaining high standards of fairness and
transparency.

--------------------------------------------------------------------------------

**How It Works: A "Grading Committee" of AI Agents**

This script uses a sophisticated multi-agent system to mimic the collaborative
process of a human grading committee. Instead of one AI trying to do everything,
we have a team of specialists, each with a distinct role:

1.  **GradingManager (The Lead Instructor):**
    -   Oversees the entire process for each essay.
    -   Delegates specific tasks to its team of worker agents.
    -   Synthesizes all feedback into a final, coherent report.

2.  **ContentAnalyst (The Subject Matter Expert):**
    -   Focuses exclusively on the essay's content, analyzing the strength of
        arguments, the quality of evidence, and the depth of analysis.

3.  **FactChecker (The Research Assistant - RAG Powered):**
    -   When provided with course materials (lecture notes, textbooks), this
        agent uses Retrieval-Augmented Generation (RAG) to verify the factual
        accuracy of claims made in the essay against the provided context.
        This ensures the grading is grounded in the course's specific knowledge.

4.  **ClarityAndStyleChecker (The Writing Tutor):**
    -   Evaluates the mechanics of the writing: grammar, spelling, sentence
        structure, clarity, and overall style. It ignores the content's
        accuracy to focus purely on communication quality.

5.  **RubricAligner (The Detail-Oriented TA):**
    -   This is the key to fair and consistent grading. It takes the analyses
        from all other agents and its sole job is to fill out a structured
        JSON form based on the specific criteria in the instructor's rubric.
        This forces the AI to justify every point awarded, ensuring transparency.

--------------------------------------------------------------------------------

**How to Use This Tool: A Step-by-Step Guide**

**Step 1: Prepare Your Folders**
Create the following three folders in the same directory as this script:
  - `essays_to_grade/`: Place all student essays (.docx, .pdf, .txt) here.
  - `course_materials/` (Optional): Place any relevant course materials
    (lecture notes, textbook chapters as .txt, .pdf, etc.) here. This will
    activate the RAG-powered FactChecker agent. If you have no materials, you
    can leave this folder empty or omit the `--materials` argument.
  - `graded_essays/`: This is where the final grade reports will be saved.

**Step 2: Create Your Grading Rubric**
Create a text file (e.g., `grading_rubric.txt`) that contains the rubric for
the assignment. Be as detailed as possible, including criteria and point
values. For example:

    - **Thesis Statement (15 points):** Must be clear, arguable, and located
      in the introduction.
    - **Argument & Evidence (40 points):** Arguments must be well-supported
      with specific, relevant evidence. Claims should be factually accurate.
    - ...and so on.

**Step 3: Run the Script from Your Terminal**
Open your terminal or command prompt, navigate to the directory containing this
script and your folders, and run the script using the following command structure.

**Basic Usage:**
```bash
python demo_essay_autograder.py --essays essays_to_grade/ --rubric grading_rubric.txt --output graded_essays/
```

**Usage with RAG Fact-Checking:**
```bash
python demo_essay_autograder.py --essays essays_to_grade/ --rubric grading_rubric.txt --output graded_essays/ --materials course_materials/
```

The script will then process each essay in the `essays_to_grade` folder and
generate a detailed `.txt` report for each one in the `graded_essays` folder.

================================================================================
"""
import os
import asyncio
import logging
import argparse
from pathlib import Path
import json

# --- Step 1: Import from the new fairlib.utils.module and the central fairlib API ---
from fairlib.utils.autograder_utils import (
    setup_knowledge_base, create_agent, format_report, FinalGrade
)
from fairlib.utils.document_processor import DocumentProcessor
from fairlib import (
    settings, OpenAIAdapter, HierarchicalAgentRunner, ManagerPlanner, SimpleRetriever,
    KnowledgeBaseQueryTool, GradeEssayFromRubricTool, WorkingMemory, SimpleAgent  
)

from dotenv import load_dotenv
load_dotenv()

settings.api_keys.openai_api_key = os.getenv("OPENAI_API_KEY")
settings.api_keys.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Configure logger for this specific module
logger = logging.getLogger(__name__)

# --- Step 2: Main Essay Grading Orchestration ---
async def grade_single_essay(essay_doc, rubric, knowledge_base):
    """
    Orchestrates the entire multi-agent grading process for one essay.
    This function sets up the agent "committee" and the manager prompt.
    """
    essay_text = essay_doc.page_content
    essay_filename = Path(essay_doc.metadata.get("source", "unknown_essay")).name
    logger.info(f"--- Starting essay grading for: {essay_filename} ---")

    rubric_text = "\n".join([doc.page_content for doc in rubric])
    
    llm = OpenAIAdapter(api_key=settings.api_keys.openai_api_key, model_name=settings.models.get("openai_gpt4", {"model_name": "gpt-4o"}).model_name)

    # --- Create the "Grading Committee" using tools from the framework ---
    fact_checker_tools = [KnowledgeBaseQueryTool(SimpleRetriever(knowledge_base.vector_store))] if knowledge_base else []
    
    # Conditionally create the FactChecker only if it has tools (i.e., materials were provided)
    workers = {}
    if fact_checker_tools:
        workers["FactChecker"] = create_agent(llm, "A research assistant. Use the 'course_knowledge_query' tool to verify claims made in a text against the course materials.", fact_checker_tools)

    workers.update({
        "ContentAnalyst": create_agent(llm, "A university professor. Analyze the essay's content for strength of argument, quality of evidence, and depth of analysis."),
        "ClarityAndStyleChecker": create_agent(llm, "A university writing tutor. Analyze the essay's grammar, clarity, and style."),
        "RubricAligner": create_agent(llm, "A teaching assistant. Use the 'grade_essay_from_rubric' tool to generate the final grade.", [GradeEssayFromRubricTool(llm)])
    })
    
    # --- Create the Manager agent directly, not with the worker factory ---
    # The manager's role is to plan and delegate, not execute tools.
    manager_memory = WorkingMemory()
    manager_planner = ManagerPlanner(llm, workers)
    manager_agent = SimpleAgent(
        llm=llm,
        planner=manager_planner,
        tool_executor=None, # A manager does not execute tools directly
        memory=manager_memory
    )
    manager_agent.role_description = "The lead instructor managing the grading committee."

    team_runner = HierarchicalAgentRunner(manager_agent, workers, max_steps=10) # Increased max_steps for more complex workflow
    
    # --- Correct the delegation workflow in the manager's prompt ---
    # The manager must delegate all tasks directly.
    workflow_steps = [
        "Delegate to the `ClarityAndStyleChecker` to get a report on writing quality."
    ]
    # Conditionally add the FactChecker step if it's available.
    if "FactChecker" in workers:
        workflow_steps.insert(0, "Delegate to the `FactChecker` to verify any factual claims in the essay.")
    
    workflow_steps.extend([
        "After gathering initial reports, delegate to the `ContentAnalyst`, providing it with the original essay AND the reports from the other workers for full context.",
        "Synthesize all reports (style, content, and fact-checking).",
        "Delegate to the `RubricAligner` with all synthesized information to get the final structured grade.",
        "Present the structured grade as your final answer."
    ])
    
    manager_prompt = f"""
    Please coordinate your team to grade the following student essay based on the provided rubric.

    Workflow Steps:
    {"".join([f"{i+1}. {step}\n" for i, step in enumerate(workflow_steps)])}
    **Rubric:**
    {rubric_text}

    **Student Essay to be Graded:**
    {essay_text}
    """
    
    try:
        final_evaluation = await team_runner.arun(manager_prompt)
        logger.info(f"Successfully completed agent run for {essay_filename}")
        return final_evaluation
    except Exception as e:
        logger.error(f"The multi-agent run failed for {essay_filename}: {e}", exc_info=True)
        return json.dumps({"error": f"A critical error occurred during the agent execution for this essay. Details: {e}"})


# --- Main execution block ---
async def main(essays_dir, rubric_path, output_dir, materials_dir):
    """Main function to run the batch grading process."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    doc_proc = DocumentProcessor()
    rubric_content = doc_proc.process_file(str(Path(rubric_path)))
    if not rubric_content:
        logger.critical(f"Could not load rubric from '{rubric_path}'. Exiting.")
        return

    knowledge_base = setup_knowledge_base(materials_dir) if materials_dir else None
    student_essays = doc_proc.load_documents_from_folder(essays_dir)

    if not student_essays:
        logger.warning(f"No essays found in '{essays_dir}'. Exiting.")
        return

    # Process each essay, wrapping the main call in error handling
    # This ensures that one failed essay does not stop the entire batch.
    for essay in student_essays:
        try:
            grade_json = await grade_single_essay(essay, rubric_content, knowledge_base)
            original_filename = Path(essay.metadata["source"]).stem
            report_filepath = output_path / f"{original_filename}_grade_report.txt"
            report_content = format_report(grade_json, Path(essay.metadata["source"]).name)
            report_filepath.write_text(report_content, encoding='utf-8')
            logger.info(f"✅ Grade report saved to: {report_filepath}")
        except Exception as e:
            logger.error(f"A critical error occurred while processing {essay.metadata.get('source', 'an essay')}. Skipping. Error: {e}", exc_info=True)
            # Optionally, write an error report for the failed essay
            error_report_path = output_path / f"{Path(essay.metadata.get('source', 'failed_essay')).stem}_error_report.txt"
            error_report_path.write_text(f"Failed to grade this essay due to a critical error:\n{e}")
    
    logger.info("\n--- Essay Grading Batch Complete ---")


if __name__ == "__main__":
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Multi-Agent AI Essay Autograder")
    parser.add_argument("--essays", type=str, required=True, help="Directory with student essays.")
    parser.add_argument("--rubric", type=str, required=True, help="Path to the grading rubric .txt file.")
    parser.add_argument("--output", type=str, required=True, help="Directory to save grade reports.")
    parser.add_argument("--materials", type=str, default=None, help="Optional: Directory with course materials for RAG.")
    args = parser.parse_args()

    # Create dummy directories and files for demonstration if they don't exist
    Path(args.essays).mkdir(exist_ok=True)
    if not list(Path(args.essays).glob('*')):
        (Path(args.essays) / "sample_essay.txt").write_text("This is a sample essay.")

    if not Path(args.rubric).exists():
        Path(args.rubric).write_text("- Thesis (10 pts): Clear and concise.")

    if args.materials:
        Path(args.materials).mkdir(exist_ok=True)

    # Run the main asynchronous function
    asyncio.run(main(args.essays, args.rubric, args.output, args.materials))

