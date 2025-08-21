# demo_committee_of_agents_coding_autograder.py
# demo_programming_autograder.py
"""
================================================================================
        Multi-Agent AI Programming Assignment Autograder
================================================================================

**Purpose:**
This script provides a framework for auto-grading student programming assignments.
It evaluates submissions based on correctness (by running unit tests), code
quality, style, and the logical soundness of the implemented solution.

**Who is this for?**
This tool is designed for computer science instructors and TAs who need to
grade programming assignments and want an AI-assisted workflow to handle the
repetitive aspects of code review, such as running tests and checking for
style, while also getting a high-level analysis of the student's approach.

--------------------------------------------------------------------------------

**CRITICAL SECURITY WARNING: CODE EXECUTION**
This tool includes a `CodeRunner` agent that executes student-submitted code
to run unit tests. Executing untrusted code from any source is **EXTREMELY
DANGEROUS** and poses a significant security risk.

The `sandbox_code_execution` function in this demo script is a **NON-SECURE
PLACEHOLDER**. It uses a simple Python subprocess, which **DOES NOT** provide
adequate isolation. For any real-world application, this function **MUST** be
replaced with a robust, secure sandboxing technology like:
  - **Docker Containers:** Running each submission in an isolated container.
  - **gVisor or Firecracker:** Providing a secure kernel-level sandbox.
  - **A dedicated, secure third-party code execution service.**

**DO NOT RUN THIS SCRIPT IN A PRODUCTION ENVIRONMENT WITHOUT A PROPER SANDBOX.**

--------------------------------------------------------------------------------

**How It Works: A "Code Review Committee" of AI Agents**

1.  **GradingManager (The Senior Developer/Tech Lead):**
    -   Orchestrates the entire code review process for each submission.
    -   Delegates specific analysis tasks to its specialized team members.

2.  **CodeRunner (The QA Engineer) - OPTIONAL:**
    -   If enabled, this agent runs the student's code against unit tests.
    -   This is the most critical agent for correctness, but it can be disabled
        for assignments where only static analysis is needed.

3.  **StaticAnalyzer (The Linter & Style Cop):**
    -   Reviews the code without running it. It checks for style guides
        (e.g., PEP 8), complexity, and comment quality.

4.  **LogicAndEfficiency (The Principal Architect):**
    -   Performs a conceptual review of the student's approach. Is the
        algorithm efficient? Is the logic sound?

5.  **RubricAligner (The TA):**
    -   Synthesizes all reports to fill out a structured JSON grade, ensuring
        every rubric criterion is scored with a clear justification.

--------------------------------------------------------------------------------

**How to Use This Tool**

**Step 1: Prepare Your Folders**
  - `submissions/`: Place all student code files (e.g., `student1.py`) here.
  - `tests/` (Optional): Place the unit test file here if you plan to run tests.
  - `reports/`: This is where the output grade reports will be saved.

**Step 2: Write Unit Tests**
Create a Python file with `pytest`-compatible unit tests. This file will be
used to evaluate all submissions for the assignment.

**Step 3: Run from the Terminal**

**To run WITH test execution:**
```bash
python demo_programming_autograder.py --submissions submissions/ --tests tests/test_assignment1.py --rubric rubric.txt --output reports/
```

**To run WITHOUT test execution (static analysis only):**
```bash
python demo_programming_autograder.py --submissions submissions/ --rubric rubric.txt --output reports/ --no-run
```
Note: The `--tests` argument is not needed when using `--no-run`.
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
    create_agent, format_report, FinalGrade
)
from fairlib.utils.document_processor import DocumentProcessor
from fairlib import (
    settings, OpenAIAdapter, HierarchicalAgentRunner, ManagerPlanner,
    CodeExecutionTool, GradeCodeFromRubricTool, WorkingMemory, SimpleAgent
)

from dotenv import load_dotenv
load_dotenv()

settings.api_keys.openai_api_key = os.getenv("OPENAI_API_KEY")
settings.api_keys.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

logger = logging.getLogger(__name__)

# --- Step 2: Main Code Grading Orchestration ---
async def grade_single_submission(submission_doc, test_code, rubric, run_tests: bool):
    """
    Orchestrates the multi-agent grading process for a single code submission.
    """
    submission_text = submission_doc.page_content
    submission_filename = Path(submission_doc.metadata.get("source", "unknown_submission")).name
    logger.info(f"--- Starting code grading for: {submission_filename} (Run tests: {run_tests}) ---")

    llm = OpenAIAdapter(
        api_key=settings.api_keys.openai_api_key,
        model_name=settings.models.get("openai_gpt4", {"model_name": "gpt-4o"}).model_name
    )

    # --- Define the "Code Review Committee" ---
    # Agents are created dynamically based on whether execution is needed.
    static_analyzer = create_agent(llm, "A senior developer. Analyze the code for style, clarity, comments, and complexity. Do not run it.")
    logic_reviewer = create_agent(llm, "A principal software architect. Review the code for its algorithmic approach, logic, and efficiency.")
    rubric_aligner = create_agent(llm, "A teaching assistant. Use the 'grade_code_from_rubric' tool to generate the final grade.", [GradeCodeFromRubricTool(llm)])
    
    workers = {
        "StaticAnalyzer": static_analyzer,
        "LogicAndEfficiency": logic_reviewer,
        "RubricAligner": rubric_aligner
    }
    
    # Conditionally add the CodeRunner agent to the team
    if run_tests:
        workers["CodeRunner"] = create_agent(llm, "A QA Engineer. Use the 'run_code_with_tests' tool.", [CodeExecutionTool()])
    
    # --- Create the Manager agent directly, not with the worker factory ---
    # The manager's role is to plan and delegate, not execute tools, so its
    # tool_executor should be None.
    manager_memory = WorkingMemory()
    manager_planner = ManagerPlanner(llm, workers)
    manager_agent = SimpleAgent(
        llm=llm,
        planner=manager_planner,
        tool_executor=None, # A manager does not execute tools directly
        memory=manager_memory
    )
    # This role description is used by the ManagerPlanner's prompt builder.
    manager_agent.role_description = "The lead developer managing the code review."

    team_runner = HierarchicalAgentRunner(manager_agent, workers, max_steps=8)
    
    # --- Dynamically construct the manager's prompt ---
    # The workflow instructions change based on whether the CodeRunner is active.
    workflow_steps = ["Delegate to `StaticAnalyzer` and `LogicAndEfficiency` for their reviews."]
    if run_tests:
        workflow_steps.insert(0, "Delegate to the `CodeRunner` to execute the code against the tests.")
    workflow_steps.append("Synthesize all results.")
    workflow_steps.append("Delegate to the `RubricAligner` with all information to get the final structured grade.")
    workflow_steps.append("Present the structured grade as your final answer.")
    
    manager_prompt = f"""
Please coordinate your team to grade the following programming assignment.
Workflow: {" ".join([f"{i+1}. {step}" for i, step in enumerate(workflow_steps)])}

**Rubric:** {rubric}
**Unit Tests (for context, not execution unless CodeRunner is used):** ```python\n{test_code if run_tests else "N/A - Execution is disabled."}\n```
**Student Code:** ```python\n{submission_text}\n```
"""

    try:
        final_evaluation = await team_runner.arun(manager_prompt)
        logger.info(f"Successfully completed agent run for {submission_filename}. Raw output:\n{final_evaluation}")
        # Return the raw string output. format_report will handle parsing and validation.
        return final_evaluation
    except Exception as e:
        logger.error(f"The multi-agent run failed for {submission_filename}: {e}", exc_info=True)
        # Return a structured error message that format_report can handle
        return json.dumps({"error": f"A critical error occurred during the agent execution for this submission ({type(e).__name__}). Details: {e}"})

async def main(submissions_dir, rubric_path, output_dir, tests_path=None, run_tests=True):
    """Main function to run the batch grading process for code."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    doc_proc = DocumentProcessor()
    rubric_content = doc_proc.process_file(str(Path(rubric_path)))
    if not rubric_content:
        logger.critical(f"Could not load rubric from '{rubric_path}'. Exiting.")
        return

    test_code_content = None
    if run_tests:
        if not tests_path:
            logger.critical("--tests argument is required when running with execution. Exiting.")
            return
        test_code_content = doc_proc.process_file(str(Path(tests_path)))
        if not test_code_content:
            logger.critical(f"Could not load unit tests from '{tests_path}'. Exiting.")
            return

    student_submissions = doc_proc.load_documents_from_folder(submissions_dir)
    if not student_submissions:
        logger.warning(f"No submissions found in '{submissions_dir}'. Exiting.")
        return

    for submission in student_submissions:
        try:
            grade_json = await grade_single_submission(submission, test_code_content, rubric_content, run_tests)
            original_filename = Path(submission.metadata["source"]).stem
            report_filepath = output_path / f"{original_filename}_grade_report.txt"
            report_content = format_report(grade_json, Path(submission.metadata["source"]).name)
            report_filepath.write_text(report_content, encoding='utf-8')
            logger.info(f"✅ Grade report saved to: {report_filepath}")
        except Exception as e:
            logger.error(f"A critical error occurred while processing {submission.metadata.get('source', 'a submission')}. Skipping. Error: {e}", exc_info=True)
    
    logger.info("\n--- Programming Grading Batch Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent AI Programming Autograder")
    parser.add_argument("--submissions", type=str, required=True, help="Directory with student code submissions.")
    parser.add_argument("--rubric", type=str, required=True, help="Path to the grading rubric .txt file.")
    parser.add_argument("--output", type=str, required=True, help="Directory to save grade reports.")
    parser.add_argument("--tests", type=str, help="Path to the pytest unit tests file. Required unless --no-run is specified.")
    parser.add_argument("--no-run", action="store_true", help="Disable code execution. The grader will only perform static analysis.")
    args = parser.parse_args()
    
    run_tests_flag = not args.no_run

    # Create dummy files and folders for demonstration if they don't exist
    Path(args.submissions).mkdir(exist_ok=True)
    Path(args.output).mkdir(exist_ok=True)

    asyncio.run(main(args.submissions, args.rubric, args.output, args.tests, run_tests_flag))
