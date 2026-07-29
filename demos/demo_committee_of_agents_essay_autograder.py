# demo_committee_of_agents_essay_autograder.py

"""
Multi-Agent AI Essay Autograder with RAG

Purpose:
This script provides a powerful, automated tool for grading student essays,
designed to save educators time and increase grading consistency. It can
process a batch of essays in .docx, .pdf, or .txt format, evaluating
each one against a custom, user-provided rubric.

Who is this for?
This tool is for college-level educators, teaching assistants, or any instructor
who needs to grade a large number of subjective papers and wants to leverage
AI to streamline the process while maintaining high standards of fairness and
transparency.

How It Works: A Grading Committee of AI Agents

This script uses a multi-agent system to mimic the collaborative process of a
human grading committee. Instead of one AI trying to do everything, we have a
team of specialists, each with a distinct role. The team wiring is the
workers-as-tools pattern: every specialist is an ordinary stateless
SimpleAgent wrapped in a WorkerAgentTool, and the manager is a plain
SimpleAgent built by build_worker_manager whose tools ARE the committee. The
manager model reads each worker's job from its tool description in the
rendered tool catalog and decides the delegation order itself, guided by the
workflow in its prompt. A delegation turn is an ordinary ToolCallBatch, so
the side-effect-aware executor applies: every grader here is declared
READ_ONLY (they analyze and retrieve; nothing mutates state), which means
independent delegations issued in one turn - the fact check and the style
check, for example - can run concurrently instead of strictly one after
another.

The committee:

1. The grading manager (the lead instructor):
   - A plain SimpleAgent over the worker tools; no special orchestrator
     class. It delegates subtasks and synthesizes the final report.

2. content_analyst (the subject matter expert):
   - Focuses exclusively on the essay's content, analyzing the strength of
     arguments, the quality of evidence, and the depth of analysis.

3. fact_checker (the research assistant - RAG powered):
   - When provided with course materials (lecture notes, textbooks), this
     agent uses Retrieval-Augmented Generation (RAG) to verify the factual
     accuracy of claims made in the essay against the provided context.
     This ensures the grading is grounded in the course's specific knowledge.

4. clarity_style_checker (the writing tutor):
   - Evaluates the mechanics of the writing: grammar, spelling, sentence
     structure, clarity, and overall style. It ignores the content's
     accuracy to focus purely on communication quality.

5. rubric_aligner (the detail-oriented TA):
   - This is the key to fair and consistent grading. It takes the analyses
     from all other agents and its sole job is to fill out a structured
     JSON form based on the specific criteria in the instructor's rubric.
     This forces the AI to justify every point awarded, ensuring transparency.

What it shows:
  - WorkerAgentTool adapts each grader into a typed tool; the tool
    description is what the manager model reads to choose a worker.
  - build_worker_manager is pure wiring: the manager is an ordinary
    SimpleAgent with a batch-capable planner over the worker tools.
  - READ_ONLY side-effect declarations let independent analyses
    (fact-checking and style-checking) overlap when the manager delegates
    them together in one turn; a mutating worker would keep the
    conservative EXTERNAL default and act as a sequential barrier.
  - Workers stay ordinary stateless SimpleAgents with their own planners
    and tools - the same agents you would build standalone.

The manager is driven by a real local model, so the delegation order is the
model's own decision each run; the workflow in the prompt guides it, and an
imperfect run still produces a report.

How to Use This Tool: A Step-by-Step Guide

Step 1: Prepare Your Folders
Create the following three folders in the same directory as this script:
  - essays_to_grade/: Place all student essays (.docx, .pdf, .txt) here.
  - course_materials/ (Optional): Place any relevant course materials
    (lecture notes, textbook chapters as .txt, .pdf, etc.) here. This will
    activate the RAG-powered FactChecker agent. If you have no materials, you
    can leave this folder empty or omit the --materials argument.
  - graded_essays/: This is where the final grade reports will be saved.

Step 2: Create Your Grading Rubric
Create a text file (e.g., grading_rubric.txt) that contains the rubric for
the assignment. Be as detailed as possible, including criteria and point
values. For example:

    - Thesis Statement (15 points): Must be clear, arguable, and located
      in the introduction.
    - Argument and Evidence (40 points): Arguments must be well-supported
      with specific, relevant evidence. Claims should be factually accurate.
    - ...and so on.

Step 3: Run the Script from Your Terminal
Open your terminal or command prompt, navigate to the directory containing this
script and your folders, and run the script using the following command structure.

Basic Usage:

    python demo_essay_autograder.py --essays essays_to_grade/ --rubric grading_rubric.txt --output graded_essays/

Usage with RAG Fact-Checking:

    python demo_essay_autograder.py --essays essays_to_grade/ --rubric grading_rubric.txt --output graded_essays/ --materials course_materials/

The script will then process each essay in the essays_to_grade folder and
generate a detailed .txt report for each one in the graded_essays folder.
"""
import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

from fairlib import (
    AgentEventBus,
    GradeEssayFromRubricTool,
    HuggingFaceAdapter,
    KnowledgeBaseQueryTool,
    SimpleAgent,
    SimpleRetriever,
    WorkerAgentTool,
    build_worker_manager,
)
from fairlib.core.events import ToolCallPostEvent
from fairlib.core.interfaces.llm import AbstractChatModel
from fairlib.core.interfaces.tools import AbstractTool, SideEffect

# Step 1: Import from the fairlib.utils module and the central fairlib API
from fairlib.utils.autograder_utils import (
    create_agent,
    format_report,
    setup_knowledge_base,
)
from fairlib.utils.document_processor import DocumentProcessor

# Configure logger for this specific module
logger = logging.getLogger(__name__)

# Step 2: Committee construction. Each grader is an ordinary stateless
# SimpleAgent from the shared autograder factory; nothing about a grader is
# committee-specific until WorkerAgentTool wraps it. Stateless matters: each
# delegation must be planned fresh, not against the history of the previous
# essay's delegations. The manager model learns what each grader is for from
# the WorkerAgentTool description in its rendered tool catalog, so the
# role_description shapes only the grader itself.
def create_grader(
    llm: AbstractChatModel,
    role_description: str,
    tools: Optional[List[AbstractTool]] = None,
    events: Optional[AgentEventBus] = None,
) -> SimpleAgent:
    """Build one stateless committee grader via the shared agent factory."""
    return create_agent(
        llm, role_description, tools, stateless=True, events=events
    )


# Step 3: Main Essay Grading Orchestration
async def grade_single_essay(essay_doc, rubric, knowledge_base):
    """
    Orchestrates the entire multi-agent grading process for one essay.
    This function sets up the agent committee and the manager prompt.
    """
    essay_text = essay_doc.page_content
    essay_filename = Path(essay_doc.metadata.get("source", "unknown_essay")).name
    logger.info(f"--- Starting essay grading for: {essay_filename} ---")

    # The manager inlines the essay and committee reports into delegation
    # payloads under the strict multi-action JSON contract; that needs a
    # capable instruct model and enough tokens that the longest essay does
    # not truncate mid-JSON. Override with FAIR_LLM_DEMO_MODEL to
    # experiment.
    llm = HuggingFaceAdapter(
        os.environ.get("FAIR_LLM_DEMO_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        max_new_tokens=2048,
    )

    # One shared bus, wired before the committee exists: the rubric
    # aligner's own executor publishes the grading tool's typed result, so
    # the demo captures the structured grade at its source instead of
    # trusting the manager model to echo JSON verbatim (consumers subscribe
    # to typed events; they do not parse them back out of model text).
    bus = AgentEventBus()
    structured_grades: List[str] = []

    def _capture_grade(event: ToolCallPostEvent) -> None:
        if event.tool_name == "grade_essay_from_rubric" and event.succeeded:
            structured_grades.append(event.observation)

    bus.subscribe(ToolCallPostEvent, _capture_grade)

    # Create the Grading Committee: each specialist is a stateless
    # SimpleAgent wrapped as a typed tool. Every grader is READ_ONLY - they
    # retrieve, analyze, and compute, but mutate nothing - which is the
    # declaration that lets independent delegations issued in one manager
    # turn run concurrently. A grader that wrote files or updated a
    # gradebook would keep WorkerAgentTool's conservative EXTERNAL default
    # and run as a sequential barrier instead.
    worker_tools = []

    # Conditionally create the fact checker only if materials were provided.
    if knowledge_base:
        fact_checker = create_grader(
            llm,
            "A research assistant. Use the 'course_knowledge_query' tool to "
            "verify claims made in a text against the course materials.",
            [KnowledgeBaseQueryTool(SimpleRetriever(knowledge_base.vector_store))],
        )
        worker_tools.append(
            WorkerAgentTool(
                fact_checker,
                name="fact_checker",
                description=(
                    "Delegate a fact-checking subtask: give this research "
                    "assistant the claims to verify and it checks them "
                    "against the course materials via RAG retrieval."
                ),
                side_effect=SideEffect.READ_ONLY,
            )
        )

    content_analyst = create_grader(
        llm,
        "A university professor. Analyze the essay's content for strength "
        "of argument, quality of evidence, and depth of analysis.",
    )
    clarity_checker = create_grader(
        llm,
        "A university writing tutor. Analyze the essay's grammar, clarity, "
        "and style.",
    )
    rubric_aligner = create_grader(
        llm,
        "A teaching assistant. Use the 'grade_essay_from_rubric' tool to "
        "generate the final grade.",
        [GradeEssayFromRubricTool(llm)],
        events=bus,
    )
    worker_tools.extend([
        WorkerAgentTool(
            content_analyst,
            name="content_analyst",
            description=(
                "Delegate a content-analysis subtask: include the essay text "
                "and any earlier committee reports, and this professor "
                "evaluates argument strength, evidence, and depth."
            ),
            side_effect=SideEffect.READ_ONLY,
        ),
        WorkerAgentTool(
            clarity_checker,
            name="clarity_style_checker",
            description=(
                "Delegate a writing-quality subtask: include the essay text, "
                "and this writing tutor reports on grammar, clarity, and "
                "style only, ignoring content accuracy."
            ),
            side_effect=SideEffect.READ_ONLY,
        ),
        WorkerAgentTool(
            rubric_aligner,
            name="rubric_aligner",
            description=(
                "Delegate the final grading subtask: include the rubric, the "
                "essay, and the synthesized committee reports, and this TA "
                "returns the structured JSON grade."
            ),
            side_effect=SideEffect.READ_ONLY,
        ),
    ])

    # The manager is a plain SimpleAgent over the worker tools: no special
    # orchestrator class, no separate roster. The model sees the committee
    # through the rendered tool catalog and delegates through ordinary
    # typed tool calls.
    manager_agent = build_worker_manager(llm, worker_tools, max_steps=10, events=bus)

    # The delegation workflow in the manager's prompt. Independent analyses
    # may be delegated together in one turn; because the graders are
    # READ_ONLY, such a batch runs concurrently.
    workflow_steps = [
        "Delegate to the `clarity_style_checker` tool to get a report on "
        "writing quality."
    ]
    # Conditionally add the fact-checking step if it's available.
    if knowledge_base:
        workflow_steps.insert(0, "Delegate to the `fact_checker` tool to verify any factual claims in the essay.")
        workflow_steps.insert(
            2,
            "The fact-checking and writing-quality delegations are "
            "independent of each other; you may issue both in the same turn "
            "and they will run concurrently.",
        )

    workflow_steps.extend([
        "After gathering initial reports, delegate to the `content_analyst` tool, providing it with the original essay AND the reports from the other workers for full context.",
        "Synthesize all reports (style, content, and fact-checking).",
        "Delegate to the `rubric_aligner` tool with all synthesized information to get the final structured grade.",
        "Present the structured grade as your final answer."
    ])

    workflow_text = "".join(f"{i+1}. {step}\n" for i, step in enumerate(workflow_steps))
    manager_prompt = f"""
    Please coordinate your team to grade the following student essay based on the provided rubric.

    Workflow Steps:
    {workflow_text}
    **Rubric:**
    {rubric}

    **Student Essay to be Graded:**
    {essay_text}
    """

    try:
        final_evaluation = await manager_agent.arun(manager_prompt)
        logger.info(f"Successfully completed agent run for {essay_filename}")
        # The grading tool's typed GradeResult, captured off the event bus
        # at its source, is the authoritative structured grade; the
        # manager's final text is presentation on top of it. Fall back to
        # the raw text only when the aligner never produced a validated
        # grade.
        if structured_grades:
            return structured_grades[-1]
        return final_evaluation
    except Exception as e:
        logger.error(f"The multi-agent run failed for {essay_filename}: {e}", exc_info=True)
        return json.dumps({"error": f"A critical error occurred during the agent execution for this essay. Details: {e}"})


# Main execution block
async def main(essays_dir, rubric_path, output_dir, materials_dir):
    """Main function to run the batch grading process."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    doc_proc = DocumentProcessor()
    # The prompt needs the rubric as one plain-text block, and grading
    # judges each essay whole with one report per student, so both loads
    # use the processor's whole-file surface rather than its RAG chunking.
    rubric_content = doc_proc.read_file_text(str(Path(rubric_path)))
    if not rubric_content:
        logger.critical(f"Could not load rubric from '{rubric_path}'. Exiting.")
        return

    knowledge_base = setup_knowledge_base(materials_dir) if materials_dir else None
    student_essays = doc_proc.load_whole_documents(essays_dir)

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
            logger.info(f"Grade report saved to: {report_filepath}")
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

