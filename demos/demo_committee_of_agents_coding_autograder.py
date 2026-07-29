# demo_committee_of_agents_coding_autograder.py
"""
Multi-Agent AI Programming Assignment Autograder

Purpose:
This script provides a framework for auto-grading student programming
assignments. It evaluates submissions based on correctness (by running unit
tests), code quality, style, and the logical soundness of the implemented
solution. A real local model (loaded through the HuggingFaceAdapter) plays
every seat on the committee.

Who is this for?
This tool is designed for computer science instructors and TAs who need to
grade programming assignments and want an AI-assisted workflow to handle the
repetitive aspects of code review, such as running tests and checking for
style, while also getting a high-level analysis of the student's approach.

CRITICAL SECURITY WARNING: CODE EXECUTION
This tool includes a code_runner agent that executes student-submitted code
to run unit tests. Executing untrusted code from any source is EXTREMELY
DANGEROUS and poses a significant security risk.

The code execution path in this demo is a NON-SECURE PLACEHOLDER. It uses a
simple Python subprocess, which DOES NOT provide adequate isolation. For any
real-world application, it MUST be replaced with a robust, secure sandboxing
technology like:
  - Docker Containers: Running each submission in an isolated container.
  - gVisor or Firecracker: Providing a secure kernel-level sandbox.
  - A dedicated, secure third-party code execution service.

DO NOT RUN THIS SCRIPT IN A PRODUCTION ENVIRONMENT WITHOUT A PROPER SANDBOX.

How It Works: A Code Review Committee of AI Agents

The committee is wired on the workers-as-tools pattern: each grader is an
ordinary stateless SimpleAgent wrapped in a WorkerAgentTool, and the manager
is a plain SimpleAgent built by build_worker_manager whose tools ARE the
graders. The manager model learns who is on the committee from each worker
tool's description in its rendered tool catalog, and a turn that delegates
to several graders at once is an ordinary ToolCallBatch: the side-effect-
aware executor runs READ_ONLY delegations concurrently and treats the
default EXTERNAL classification as a sequential barrier.

The seats:

1. The manager (The Senior Developer/Tech Lead):
   - A plain SimpleAgent over the worker tools. It orchestrates the review
     by delegating subtasks; each delegation must be self-contained, so the
     manager includes the student code in the subtask text.

2. code_runner (The QA Engineer) - OPTIONAL:
   - If enabled, this agent runs the student's code against unit tests.
   - It executes untrusted code (subprocess, temp files), so its wrapper
     keeps the conservative EXTERNAL default: it runs alone, as a
     sequential barrier, never overlapped with other delegations.

3. static_analyzer (The Linter and Style Cop):
   - Reviews the code without running it: style (e.g. PEP 8), complexity,
     and comment quality. Declared READ_ONLY.

4. logic_reviewer (The Principal Architect):
   - A conceptual review of the student's approach: algorithm, logic,
     efficiency. Declared READ_ONLY.

5. rubric_aligner (The TA):
   - Synthesizes all reports into a structured JSON grade via the
     grade_code_from_rubric tool. Declared READ_ONLY (it evaluates; it
     mutates nothing).

What it shows:
  - WorkerAgentTool adapts each grader into a typed tool; the tool
    description is what the manager model reads in its rendered tool
    catalog.
  - build_worker_manager is pure wiring: the manager is a plain SimpleAgent
    with a batch-capable planner over the worker-tool registry.
  - The read-only reviewers (static_analyzer, logic_reviewer) can be
    delegated together in one turn and genuinely run concurrently. The
    ToolBatchScheduledEvent printout makes the scheduler's grouping
    decision visible.
  - The code_runner keeps the conservative EXTERNAL default because it
    mutates state, so the scheduler serializes it.

A real model drives every decision, so the delegation flow is best-effort:
the model may delegate one review per turn instead of batching, and grading
output can vary between runs. The run still completes either way.

How to Use This Tool

Step 1: Prepare Your Folders
  - submissions/: Place all student code files (e.g., student1.py) here.
  - tests/ (Optional): Place the unit test file here if you plan to run tests.
  - reports/: This is where the output grade reports will be saved.

Step 2: Write Unit Tests
Create a Python file with pytest-compatible unit tests. This file will be
used to evaluate all submissions for the assignment.

Step 3: Run from the Terminal

To run WITH test execution:

    python demos/demo_committee_of_agents_coding_autograder.py --submissions submissions/ --tests tests/test_assignment1.py --rubric rubric.txt --output reports/

To run WITHOUT test execution (static analysis only):

    python demos/demo_committee_of_agents_coding_autograder.py --submissions submissions/ --rubric rubric.txt --output reports/ --no-run

Note: The --tests argument is not needed when using --no-run.
"""
import argparse
import asyncio
import json
import os
import logging
from pathlib import Path
from typing import List, Optional

from fairlib import (
    AgentEventBus,
    CodeExecutionTool,
    Example,
    GradeCodeFromRubricTool,
    HuggingFaceAdapter,
    PromptBuilder,
    RoleDefinition,
    SimpleAgent,
    WorkerAgentTool,
    build_worker_manager,
)
from fairlib.core.events import ToolBatchScheduledEvent, ToolCallPostEvent
from fairlib.core.interfaces.llm import AbstractChatModel
from fairlib.core.interfaces.tools import AbstractTool, SideEffect
from fairlib.utils.autograder_utils import create_agent, format_report
from fairlib.utils.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


# Each committee member is an ordinary stateless SimpleAgent from the shared
# autograder factory; nothing about a grader is manager-specific until
# WorkerAgentTool wraps it. The graders are stateless so every delegation is
# planned fresh, instead of against the accumulated history of the previous
# submissions' reviews.
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


def _on_schedule(event: ToolBatchScheduledEvent) -> None:
    """Show how the executor grouped the turn's delegations."""
    print(f"[scheduler] {event.batch_size} delegation(s) in this turn:")
    for i, group in enumerate(event.groups, start=1):
        how = "PARALLEL" if group.parallel else "sequential"
        print(f"  group {i}: {how:11} [{group.side_effect.value}] {', '.join(group.tool_names)}")


async def grade_single_submission(submission_doc, test_code, rubric, run_tests: bool):
    """
    Orchestrates the multi-agent grading process for a single code submission.
    """
    submission_text = submission_doc.page_content
    submission_filename = Path(submission_doc.metadata.get("source", "unknown_submission")).name
    logger.info(f"--- Starting code grading for: {submission_filename} (Run tests: {run_tests}) ---")

    # The multi-action JSON contract with code-heavy delegation payloads
    # needs a capable instruct model; override with FAIR_LLM_DEMO_MODEL
    # to experiment.
    llm = HuggingFaceAdapter(
        os.environ.get("FAIR_LLM_DEMO_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        # The manager inlines the full submission into each delegation
        # payload; 1024 tokens truncates mid-JSON on the longest file.
        max_new_tokens=2048,
    )

    # One shared bus, wired before the committee exists: the manager's
    # executor prints the scheduler's grouping decisions, and the rubric
    # aligner's own executor publishes the grading tool's typed result so
    # the demo can capture the structured grade at its source instead of
    # trusting an LLM to echo JSON verbatim (consumers subscribe to typed
    # events; they do not parse it back out of model text).
    bus = AgentEventBus()
    bus.subscribe(ToolBatchScheduledEvent, _on_schedule)

    structured_grades: List[str] = []

    def _capture_grade(event: ToolCallPostEvent) -> None:
        if event.tool_name == "grade_code_from_rubric" and event.succeeded:
            structured_grades.append(event.observation)

    bus.subscribe(ToolCallPostEvent, _capture_grade)

    # Define the Code Review Committee. Each grader is an ordinary stateless
    # SimpleAgent; wrapping it in WorkerAgentTool is what puts it on the
    # manager's roster. The manager model reads these descriptions from its
    # rendered tool catalog, so they carry each seat's role and the
    # instruction that a delegation must be self-contained.
    static_analyzer = create_grader(
        llm, "A senior developer. Analyze the code for style, clarity, comments, and complexity. Do not run it."
    )
    logic_reviewer = create_grader(
        llm, "A principal software architect. Review the code for its algorithmic approach, logic, and efficiency."
    )
    rubric_aligner = create_grader(
        llm,
        "A teaching assistant. Use the 'grade_code_from_rubric' tool to generate the final grade.",
        [GradeCodeFromRubricTool(llm)],
        events=bus,
    )

    # The graders are read-only evaluators: READ_ONLY is the author's
    # assertion that they mutate nothing, which is what lets the executor
    # run several review delegations concurrently in one turn.
    worker_tools = [
        WorkerAgentTool(
            static_analyzer,
            name="static_analyzer",
            description=(
                "Delegate a static code review to a senior developer who checks "
                "style, clarity, comments, and complexity without running the "
                "code. Include the full student code in the subtask."
            ),
            side_effect=SideEffect.READ_ONLY,
        ),
        WorkerAgentTool(
            logic_reviewer,
            name="logic_reviewer",
            description=(
                "Delegate a conceptual review to a principal architect who "
                "assesses the algorithmic approach, logic, and efficiency. "
                "Include the full student code in the subtask."
            ),
            side_effect=SideEffect.READ_ONLY,
        ),
        WorkerAgentTool(
            rubric_aligner,
            name="rubric_aligner",
            description=(
                "Delegate the final grading synthesis to a teaching assistant "
                "who fills out the rubric as structured JSON. Include the "
                "rubric, the student code, and all committee findings in the "
                "subtask."
            ),
            side_effect=SideEffect.READ_ONLY,
        ),
    ]

    # Conditionally add the code_runner seat. It executes untrusted student
    # code (subprocess, temp files on disk), so it keeps WorkerAgentTool's
    # conservative EXTERNAL default: the scheduler runs it alone as a
    # sequential barrier instead of overlapping it with other delegations.
    if run_tests:
        code_runner = create_grader(
            llm, "A QA Engineer. Use the 'run_code_with_tests' tool.", [CodeExecutionTool()]
        )
        worker_tools.append(
            WorkerAgentTool(
                code_runner,
                name="code_runner",
                description=(
                    "Delegate test execution to a QA engineer who runs the "
                    "student code against the unit tests. Include the full "
                    "student code and the full test code in the subtask."
                ),
            )
        )

    # Application prompt content for the manager: fairlib ships only the
    # mandatory JSON format rules, and a small local model needs a role and
    # a worked example of the multi-action turn shape to hit the contract
    # reliably. Two example turns teach delegation and the final answer.
    manager_builder = PromptBuilder()
    manager_builder.role_definition = RoleDefinition(
        "You are the lead developer coordinating a code review committee. "
        "You grade a submission by delegating complete, self-contained "
        "subtasks to your committee tools and synthesizing their reports."
    )
    manager_builder.examples.append(
        Example(
            'User: Review this submission.\n'
            'Assistant: {"thought": "The two reviews are independent, so I '
            'delegate both in one turn.", "actions": ['
            '{"tool_name": "static_analyzer", "tool_input": "Review the '
            'style of this code: <full code here>"}, '
            '{"tool_name": "logic_reviewer", "tool_input": "Review the '
            'logic of this code: <full code here>"}]}'
        )
    )
    manager_builder.examples.append(
        Example(
            'Observation: [rubric_aligner] {"graded_criteria": [{"criterion": '
            '"...", "score": 2, "max_score": 2, "justification": "..."}], '
            '"overall_feedback": "...", "final_score": 7}\n'
            'Assistant: {"thought": "The rubric aligner returned the '
            'structured grade; I present it as my final answer.", '
            '"actions": [{"tool_name": "final_answer", "tool_input": '
            '"<that structured grade JSON, copied verbatim>"}]}'
        )
    )

    # The manager is a plain SimpleAgent over the worker tools; delegation
    # is ordinary typed tool calling, so the batch loop does all the work.
    manager = build_worker_manager(
        llm,
        worker_tools,
        prompt_builder=manager_builder,
        events=bus,
        max_steps=8,
    )

    # Dynamically construct the manager's prompt.
    # The workflow instructions change based on whether the code_runner is active.
    workflow_steps = [
        "Delegate to `static_analyzer` and `logic_reviewer` for their reviews. "
        "Both are read-only, so you may delegate to both in the same turn and "
        "they will run concurrently."
    ]
    if run_tests:
        workflow_steps.insert(0, "Delegate to `code_runner` to execute the code against the tests.")
    workflow_steps.append("Synthesize all results.")
    workflow_steps.append("Delegate to `rubric_aligner` with all information to get the final structured grade.")
    workflow_steps.append(
        "STOP CONDITION: once an Observation from rubric_aligner appears in "
        "the history, do NOT delegate again. Your next turn must be the "
        "single final_answer action whose tool_input is that structured "
        "grade JSON, verbatim."
    )

    manager_prompt = f"""
You are the lead developer managing this code review. Coordinate your team to
grade the following programming assignment. Every delegation must be a
complete, self-contained task: the worker sees only the subtask text you
send, so include the code (and anything else the worker needs) each time.
Workflow: {" ".join([f"{i+1}. {step}" for i, step in enumerate(workflow_steps)])}

**Rubric:** {rubric}
**Unit Tests (for context, not execution unless code_runner is used):** ```python\n{test_code if run_tests else "N/A - Execution is disabled."}\n```
**Student Code:** ```python\n{submission_text}\n```
"""

    try:
        final_evaluation = await manager.arun(manager_prompt)
        logger.info(f"Successfully completed agent run for {submission_filename}. Raw output:\n{final_evaluation}")
        # The grading tool's typed GradeResult, captured off the event bus at
        # its source, is the authoritative structured grade; the manager's
        # final text is presentation on top of it. Fall back to the raw text
        # only when the aligner never produced a validated grade.
        if structured_grades:
            return structured_grades[-1]
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
    # Prompts need the rubric and tests as plain-text blocks, and grading
    # judges each submission whole with one report per student, so these
    # loads use the processor's whole-file surface rather than its RAG
    # chunking.
    rubric_content = doc_proc.read_file_text(str(Path(rubric_path)))
    if not rubric_content:
        logger.critical(f"Could not load rubric from '{rubric_path}'. Exiting.")
        return

    test_code_content = None
    if run_tests:
        if not tests_path:
            logger.critical("--tests argument is required when running with execution. Exiting.")
            return
        test_code_content = doc_proc.read_file_text(str(Path(tests_path)))
        if not test_code_content:
            logger.critical(f"Could not load unit tests from '{tests_path}'. Exiting.")
            return

    student_submissions = doc_proc.load_whole_documents(submissions_dir)
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
            logger.info(f"Grade report saved to: {report_filepath}")
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

    if not list(Path(args.submissions).glob('*')):
        (Path(args.submissions) / "student1_assignment.py").write_text("def add(a, b):\n    return a + b\n")

    if run_tests_flag and args.tests and not Path(args.tests).exists():
        (Path(args.tests)).write_text("from temp_student_code import add\n\ndef test_add():\n    assert add(2, 3) == 5\n\ndef test_add_negative():\n    assert add(-1, -1) == -2\n")

    if not Path(args.rubric).exists():
        (Path(args.rubric)).write_text("- Correctness (10 pts): Passes all unit tests.\n- Style (5 pts): Follows PEP 8.")

    asyncio.run(main(args.submissions, args.rubric, args.output, args.tests, run_tests_flag))
