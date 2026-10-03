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

Running student code:
The code_runner seat runs each submission against the unit tests with a
CodeExecutionTool bound to the submission, which hands the work to the
security manager's
asandbox_code_execution: pytest runs in a child confined at the isolated
level (Landlock and seccomp), which reads only the interpreter, its
libraries and fairlib's code, writes only its own scratch directory, and
opens no network socket. That level runs on Linux x86_64 with Landlock ABI 4
or later; elsewhere the code_runner's calls are refused, typed, and the run
continues without test results. Each grade is bounded by a
GRADING_WALL_SECONDS wall clock, so a submission that loops forever is
reported as timed out. The grade is what the confined run reports, not a
tamper-proof one: the submission runs in the same interpreter as pytest.

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
     by delegating subtasks. It never copies the student code: the demo
     keeps the submission, the tests and the rubric in a case file, each
     seat's WorkerAgentTool attaches the code to the subtask (the
     _compose_task extension point), and the committee's reports are filed
     into the case file off the event bus as each delegation completes.
     Code, quotes and newlines copied through a JSON tool_input are what a
     model cannot escape reliably, so none of the case data travels that
     way.

2. code_runner (The QA Engineer) - OPTIONAL:
   - If enabled, this agent runs the student's code against unit tests in
     an isolated child, through a CodeExecutionTool bound to the case file
     so its call needs no input. Its wrapper keeps the conservative EXTERNAL
     default: it runs alone, as a sequential barrier, never overlapped
     with other delegations. The BoundedRunEvent printout shows the level
     each grading run reached.

3. static_analyzer (The Linter and Style Cop):
   - Reviews the code without running it: style (e.g. PEP 8), complexity,
     and comment quality. Declared READ_ONLY.

4. logic_reviewer (The Principal Architect):
   - A conceptual review of the student's approach: algorithm, logic,
     efficiency. Declared READ_ONLY.

5. rubric_aligner (The TA):
   - The grade_code_from_rubric tool itself, bound to the case file: it
     fills the rubric from the rubric, the code, the pytest report and the
     reviews on file, and refuses, typed, while a review is missing. Every
     field of its form is case data, so no agent sits between the manager
     and the tool to copy it.

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
  - The code_runner's executor holds a BasicSecurityManager built from the
    settings, with a capability bag granting the code_execution tool and the
    model, so the student code runs only in an isolated child.
  - Each seat has its own event bus, and the demo prints every tool call,
    every planner parse error (with the raw model output) and every loop
    guard per seat, so a run shows who did what.

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
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ValidationError

from fairlib import (
    AbstractSecurityManager,
    AgentEventBus,
    BasicSecurityManager,
    BoundedRunEvent,
    CapabilityBag,
    CodeExecutionTool,
    Example,
    FinalGrade,
    GradeCodeFromRubricTool,
    HuggingFaceAdapter,
    PromptBuilder,
    RoleDefinition,
    SimpleAgent,
    WorkerAgentTool,
    build_worker_manager,
    settings,
)
from fairlib.core.errors import ToolInvocationError
from fairlib.core.events import (
    LoopGuardTrippedEvent,
    PlannerParseErrorEvent,
    ToolBatchScheduledEvent,
    ToolCallPostEvent,
    ToolCallPreEvent,
)
from fairlib.core.interfaces.llm import AbstractChatModel
from fairlib.core.interfaces.tools import AbstractTool, SideEffect, ToolOutput
from fairlib.modules.action.tools.code_execution_tool import CodeExecutionInput
from fairlib.modules.action.tools.grading_tool import GradeCodeInput
from fairlib.utils.autograder_utils import create_agent, format_report
from fairlib.utils.document_processor import DocumentProcessor


def _delegation_text(tool_input: object) -> str:
    """A delegation's input as the model wrote it: the subtask alone, or JSON."""
    if isinstance(tool_input, dict) and set(tool_input) == {"subtask"}:
        return str(tool_input["subtask"])
    if isinstance(tool_input, (dict, list)):
        return json.dumps(tool_input)
    return str(tool_input)


logger = logging.getLogger(__name__)

# The wall clock each grading run may take, the child's start-up included.
# A submission that loops forever is reported as timed out at this bound.
GRADING_WALL_SECONDS = 60

# The committee seats whose reports the manager files into the case file.
COMMITTEE_SEATS = frozenset({"code_runner", "static_analyzer", "logic_reviewer"})


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
    security_manager: Optional[AbstractSecurityManager] = None,
) -> SimpleAgent:
    """Build one stateless committee grader via the shared agent factory."""
    return create_agent(
        llm,
        role_description,
        tools,
        stateless=True,
        events=events,
        security_manager=security_manager,
    )


def _on_bounded_run(event: BoundedRunEvent) -> None:
    """Show one grading run: the level its child reached and how it ended."""
    achieved = event.achieved.value if event.achieved is not None else "none"
    print(
        f"[BoundedRunEvent] {event.entry_point} required={event.required.value} "
        f"achieved={achieved} outcome={event.outcome.value} "
        f"duration={event.duration_seconds:.2f}s"
    )


def _watch_seat(bus: AgentEventBus, seat: str) -> AgentEventBus:
    """Print one seat's tool calls, parse errors and loop guards, labelled."""

    def _pre(event: ToolCallPreEvent) -> None:
        print(f"[{seat}] call {event.tool_name}:\n{_delegation_text(event.tool_input)}")

    def _post(event: ToolCallPostEvent) -> None:
        status = "ok" if event.succeeded else "FAILED"
        print(f"[{seat}] {event.tool_name} {status}:\n{event.observation}")

    def _parse_error(event: PlannerParseErrorEvent) -> None:
        raw = event.raw_output or ""
        print(
            f"[{seat}] planner parse error (attempt {event.attempt}, "
            f"retry={event.will_retry}, truncated={event.truncated}, "
            f"{len(raw)} chars):\n{raw}"
        )

    def _guard(event: LoopGuardTrippedEvent) -> None:
        print(
            f"[{seat}] loop guard {event.guard_type.value} tripped at step {event.step}"
        )

    bus.subscribe(ToolCallPreEvent, _pre)
    bus.subscribe(ToolCallPostEvent, _post)
    bus.subscribe(PlannerParseErrorEvent, _parse_error)
    bus.subscribe(LoopGuardTrippedEvent, _guard)
    return bus


def _on_schedule(event: ToolBatchScheduledEvent) -> None:
    """Show how the executor grouped the turn's delegations."""
    print(f"[scheduler] {event.batch_size} delegation(s) in this turn:")
    for i, group in enumerate(event.groups, start=1):
        how = "PARALLEL" if group.parallel else "sequential"
        print(
            f"  group {i}: {how:11} [{group.side_effect.value}] {', '.join(group.tool_names)}"
        )


@dataclass
class CaseFile:
    """Everything the demo knows about one submission, held outside any model.

    The submission, the rubric and the tests are data the demo already has,
    so no model ever copies them into a JSON tool_input: the committee seats
    receive them from here with each delegation, and the case-bound tools
    read them from here. The committee's reports land here off the event
    bus as each delegation completes.
    """

    name: str
    code: str
    rubric: str
    tests: Optional[str]
    reports: Dict[str, str] = field(default_factory=dict)
    test_report: Optional[str] = None

    def brief(self, subtask: str, *, with_code: bool = True) -> str:
        """The manager's subtask with the submission attached."""
        if not with_code:
            return subtask
        return (
            f"{subtask}\n\nSubmission under review ({self.name}):\n"
            f"```python\n{self.code}\n```"
        )


class CaseSeat(WorkerAgentTool):
    """A committee seat whose every delegation carries the submission.

    _compose_task is WorkerAgentTool's extension point for the task text:
    the manager writes only what to do, and the case file supplies the code.
    """

    def __init__(
        self,
        worker: SimpleAgent,
        case: CaseFile,
        *,
        with_code: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(worker, **kwargs)
        self._case = case
        self._with_code = with_code

    def _compose_task(self, tool_input: BaseModel) -> str:
        return self._case.brief(
            super()._compose_task(tool_input), with_code=self._with_code
        )


class CaseToolInput(BaseModel):
    """A case-bound tool takes no input: the case file holds all it reads."""


class RunCaseTestsTool(CodeExecutionTool):
    """CodeExecutionTool bound to the submission under review and the assignment's tests."""

    input_schema = CaseToolInput
    description = (
        CodeExecutionTool.description
        + " It runs the submission under review against the assignment's "
        "tests, both taken from the case file, so it needs no input."
    )

    def __init__(self, case: CaseFile, wall_seconds: float) -> None:
        super().__init__(wall_seconds=wall_seconds)
        self._case = case

    async def acall(self, tool_input: CaseToolInput) -> ToolOutput:
        return await super().acall(
            CodeExecutionInput(
                student_code=self._case.code, test_code=self._case.tests or ""
            )
        )


class GradeCaseTool(GradeCodeFromRubricTool):
    """GradeCodeFromRubricTool bound to the case file: the rubric aligner's seat.

    Every input of the grading form is case data (the rubric, the code, the
    test results and the reviews), so the tool reads them itself. It refuses,
    typed, while a review is missing, so the manager learns what to delegate.
    """

    input_schema = CaseToolInput
    description = (
        "Fills the rubric for the submission under review as a structured "
        "JSON grade, from the rubric, the code, the test results and every "
        "committee report on file. Call it once every review is in; it needs "
        "no input."
    )

    def __init__(
        self, llm: AbstractChatModel, case: CaseFile, required: List[str]
    ) -> None:
        super().__init__(llm)
        self._case = case
        self._required = required

    async def acall(self, tool_input: CaseToolInput) -> ToolOutput:
        missing = [seat for seat in self._required if seat not in self._case.reports]
        if missing:
            raise ToolInvocationError(
                f"The case file has no report yet from {', '.join(missing)}; "
                "delegate to them before grading.",
                tool_name=self.name,
            )
        if self._case.tests is None:
            test_results = "Test execution was disabled for this run."
        else:
            test_results = self._case.test_report or self._case.reports.get(
                "code_runner", "No test report."
            )
        return await super().acall(
            GradeCodeInput(
                rubric=self._case.rubric,
                test_results=test_results,
                static_analysis=self._case.reports["static_analyzer"],
                logic_review=self._case.reports["logic_reviewer"],
                code=self._case.code,
            )
        )


async def grade_single_submission(
    llm: AbstractChatModel,
    submission_doc,
    test_code,
    rubric,
    run_tests: bool,
    label: str,
):
    """
    Orchestrates the multi-agent grading process for a single code submission.

    The committee sees the submission only under its anonymous label (for
    example "Submission 1"), never its file name, so a name such as
    student1_excellent cannot reach a reviewer and bias the grade.
    """
    submission_filename = Path(
        submission_doc.metadata.get("source", "unknown_submission")
    ).name
    print(f"\n=== Grading {label} ({submission_filename}, run tests: {run_tests}) ===")
    case = CaseFile(
        name=label,
        code=submission_doc.page_content,
        rubric=rubric,
        tests=test_code if run_tests else None,
    )

    # One bus per seat, each printing its seat's tool calls and parse errors.
    # The manager's bus also files every completed review into the case
    # file and captures the grading tool's typed result, so the demo takes
    # the structured grade at its source instead of trusting a model to
    # echo JSON verbatim (consumers subscribe to typed events; they do not
    # parse it back out of model text).
    bus = _watch_seat(AgentEventBus(), "manager")
    bus.subscribe(ToolBatchScheduledEvent, _on_schedule)

    structured_grades: List[str] = []

    def _file_report(event: ToolCallPostEvent) -> None:
        if not event.succeeded:
            return
        if event.tool_name == GradeCaseTool.name:
            structured_grades.append(event.observation)
        elif event.tool_name in COMMITTEE_SEATS:
            case.reports[event.tool_name] = event.observation

    bus.subscribe(ToolCallPostEvent, _file_report)

    # Define the Code Review Committee. Each grader is an ordinary stateless
    # SimpleAgent; wrapping it in a CaseSeat is what puts it on the
    # manager's roster. The manager model reads these descriptions from its
    # rendered tool catalog.
    static_analyzer = create_grader(
        llm,
        "A senior developer. Analyze the submitted code for style, clarity, "
        "comments, and complexity without running it, and report your "
        "findings as your final answer.",
        events=_watch_seat(AgentEventBus(), "static_analyzer"),
    )
    logic_reviewer = create_grader(
        llm,
        "A principal software architect. Review the submitted code for its "
        "algorithmic approach, logic, edge cases, and efficiency, and report "
        "your findings as your final answer.",
        events=_watch_seat(AgentEventBus(), "logic_reviewer"),
    )

    # The reviewers are read-only evaluators: READ_ONLY is the author's
    # assertion that they mutate nothing, which is what lets the executor
    # run several review delegations concurrently in one turn.
    worker_tools: List[AbstractTool] = [
        CaseSeat(
            static_analyzer,
            case,
            name="static_analyzer",
            description=(
                "Delegate a static code review to a senior developer who checks "
                "style, clarity, comments, and complexity without running the "
                "code. The submission is attached to the subtask for you."
            ),
            side_effect=SideEffect.READ_ONLY,
        ),
        CaseSeat(
            logic_reviewer,
            case,
            name="logic_reviewer",
            description=(
                "Delegate a conceptual review to a principal architect who "
                "assesses the algorithmic approach, logic, and efficiency. "
                "The submission is attached to the subtask for you."
            ),
            side_effect=SideEffect.READ_ONLY,
        ),
    ]
    required_reports = ["static_analyzer", "logic_reviewer"]

    # Conditionally add the code_runner seat. It runs untrusted student code,
    # so its executor holds a security manager that runs the tests in an
    # isolated child, with a capability bag granting the code execution tool
    # and the model, and every grade is bounded by GRADING_WALL_SECONDS. The
    # seat keeps WorkerAgentTool's conservative EXTERNAL default: the
    # scheduler runs it alone as a sequential barrier.
    if run_tests:
        bag = CapabilityBag.from_grants(
            tools=["code_execution"],
            models=[llm.describe_config().model_name],
        )
        security = BasicSecurityManager.from_settings(settings)
        runner_bus = _watch_seat(AgentEventBus(), "code_runner")
        runner_bus.subscribe(BoundedRunEvent, _on_bounded_run)

        def _file_test_report(event: ToolCallPostEvent) -> None:
            if event.succeeded and event.tool_name == RunCaseTestsTool.name:
                case.test_report = event.observation

        runner_bus.subscribe(ToolCallPostEvent, _file_test_report)
        code_runner = create_grader(
            llm,
            "A QA Engineer. Call the 'run_code_with_tests' tool once. Then "
            "report how many tests passed and failed, and which, as your "
            "final answer.",
            [RunCaseTestsTool(case, GRADING_WALL_SECONDS)],
            events=runner_bus,
            security_manager=security.with_capability_bag(bag),
        )
        worker_tools.insert(
            0,
            CaseSeat(
                code_runner,
                case,
                with_code=False,
                name="code_runner",
                description=(
                    "Delegate test execution to a QA engineer who runs the "
                    "submission against the assignment's unit tests in an "
                    "isolated sandbox and reports the results."
                ),
            ),
        )
        required_reports.insert(0, "code_runner")

    # The rubric aligner's seat is the grading tool itself, bound to the case
    # file: every field of its form is case data, so no model has to copy
    # the rubric, the code or the reviews into its input.
    worker_tools.append(GradeCaseTool(llm, case, required_reports))

    # Application prompt content for the manager: fairlib ships only the
    # mandatory JSON format rules, and a small local model needs a role and
    # a worked example of the multi-action turn shape to hit the contract
    # reliably. Three example turns teach delegation, grading and the final
    # answer.
    manager_builder = PromptBuilder()
    manager_builder.role_definition = RoleDefinition(
        "You are the lead developer coordinating a code review committee. "
        "You grade a submission by delegating subtasks to your committee "
        "tools and then calling the grading tool. The submission is on file: "
        "every tool receives it automatically, so a subtask says only what "
        "to do and never contains code."
    )
    manager_builder.examples.append(
        Example(
            "User: Grade the submission on file.\n"
            'Assistant: {"thought": "The two reviews are independent, so I '
            'delegate both in one turn.", "actions": ['
            '{"tool_name": "static_analyzer", "tool_input": {"subtask": '
            '"Review the style, comments and complexity of the '
            'submission."}}, '
            '{"tool_name": "logic_reviewer", "tool_input": {"subtask": '
            '"Review the logic, edge cases and efficiency of the '
            'submission."}}]}'
        )
    )
    manager_builder.examples.append(
        Example(
            "Observation: [logic_reviewer] <the last report>\n"
            'Assistant: {"thought": "Every report is in, so I grade.", '
            '"actions": [{"tool_name": "grade_code_from_rubric", "tool_input": {}}]}'
        )
    )
    manager_builder.examples.append(
        Example(
            'Observation: [grade_code_from_rubric] {"graded_criteria": [...], '
            '"overall_feedback": "...", "final_score": 72}\n'
            'Assistant: {"thought": "The grade is on file, so I finish.", '
            '"actions": [{"tool_name": "final_answer", "tool_input": '
            '{"text": "Graded: final score 72."}}]}'
        )
    )
    manager_builder.examples.append(
        Example(
            "Observation: [grade_code_from_rubric] Error in tool 'grade_code_from_rubric': No grade "
            "was produced ...\n"
            'Assistant: {"thought": "Grading failed once, so I call it once '
            'more.", "actions": [{"tool_name": "grade_code_from_rubric", '
            '"tool_input": {}}]}\n'
            "Observation: [grade_code_from_rubric] Error in tool 'grade_code_from_rubric': No grade "
            "was produced ...\n"
            'Assistant: {"thought": "Grading failed twice, so there is no '
            'grade to report.", "actions": [{"tool_name": "final_answer", '
            '"tool_input": {"text": "No grade was produced: the grading '
            'tool failed twice."}}]}'
        )
    )

    # The manager is a plain SimpleAgent over the worker tools and the
    # grading tool; delegation is ordinary typed tool calling, so the batch
    # loop does all the work.
    manager = build_worker_manager(
        llm,
        worker_tools,
        prompt_builder=manager_builder,
        events=bus,
        max_steps=8,
    )

    # The manager's prompt carries the workflow, never the code: the code
    # reaches each seat from the case file.
    workflow_steps = [
        "Delegate to `static_analyzer` and `logic_reviewer` for their reviews. "
        "Both are read-only, so you may delegate to both in the same turn and "
        "they will run concurrently."
    ]
    if run_tests:
        workflow_steps.insert(
            0, "Delegate to `code_runner` to run the submission against the tests."
        )
    workflow_steps.append("Once every review is in, call `grade_code_from_rubric`.")
    workflow_steps.append(
        "STOP CONDITION: once grade_code_from_rubric returns a grade, do NOT "
        "delegate again; your next turn is the single final_answer action "
        "stating that grade's final_score. If grade_code_from_rubric reports "
        "an error, call it once more; if it fails "
        "again, your final_answer says that no grade was produced and states "
        "no score."
    )

    manager_prompt = (
        f"Coordinate your committee to grade {label} "
        "for the calculator programming assignment. Workflow: "
        + " ".join(f"{i + 1}. {step}" for i, step in enumerate(workflow_steps))
    )

    try:
        final_evaluation = await manager.arun(manager_prompt)
        logger.info(
            f"Successfully completed agent run for {submission_filename}. Raw output:\n{final_evaluation}"
        )
        print(f"[manager] final answer:\n{final_evaluation}")
        # The grading tool's typed GradeResult, captured off the event bus at
        # its source, is the authoritative structured grade; the manager's
        # final text is presentation on top of it. Fall back to the raw text
        # only when the grading tool never produced a validated grade.
        if structured_grades:
            return structured_grades[-1]
        return final_evaluation
    except Exception as e:
        logger.error(
            f"The multi-agent run failed for {submission_filename}: {e}", exc_info=True
        )
        # Return a structured error message that format_report can handle
        return json.dumps(
            {
                "error": f"A critical error occurred during the agent execution for this submission ({type(e).__name__}). Details: {e}"
            }
        )


def _score_line(grade_json: str) -> str:
    """The typed grade's final score, or an error line when there is no grade."""
    try:
        grade = FinalGrade.model_validate_json(grade_json)
    except ValidationError:
        return "[ERROR] no grade, see the report"
    out_of = sum(item.max_score for item in grade.graded_criteria)
    return f"FINAL SCORE: {grade.final_score} / {out_of}"


async def main(
    submissions_dir, rubric_path, output_dir, tests_path=None, run_tests=True
):
    """Main function to run the batch grading process for code."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    doc_proc = DocumentProcessor()
    # Prompts need the rubric and tests as plain-text blocks, and grading
    # judges each submission whole with one report per student, so these
    # loads use the processor's whole-file surface rather than its RAG
    # chunking.
    try:
        rubric_content = doc_proc.read_file_text(str(Path(rubric_path)))
    except ImportError as e:
        logger.critical(f"Could not load rubric from '{rubric_path}': {e}. Exiting.")
        return
    if not rubric_content:
        logger.critical(f"Could not load rubric from '{rubric_path}'. Exiting.")
        return

    test_code_content = None
    if run_tests:
        if not tests_path:
            logger.critical(
                "--tests argument is required when running with execution. Exiting."
            )
            return
        try:
            test_code_content = doc_proc.read_file_text(str(Path(tests_path)))
        except ImportError as e:
            logger.critical(
                f"Could not load unit tests from '{tests_path}': {e}. Exiting."
            )
            return
        if not test_code_content:
            logger.critical(f"Could not load unit tests from '{tests_path}'. Exiting.")
            return

    student_submissions = doc_proc.load_whole_documents(submissions_dir)
    if not student_submissions:
        logger.warning(f"No submissions found in '{submissions_dir}'. Exiting.")
        return

    # One model plays every seat for the whole batch. The grade must tell an
    # excellent submission from an average one, and a 7B grader scores them
    # alike, so the default is a 14B instruct model; override with
    # FAIR_LLM_DEMO_MODEL to experiment.
    llm = HuggingFaceAdapter(
        os.environ.get("FAIR_LLM_DEMO_MODEL", "Qwen/Qwen2.5-14B-Instruct"),
        max_new_tokens=2048,
    )

    # The committee grades anonymous labels; the mapping back to file names
    # stays with the demo and is printed here and in the summary.
    # Each submission is labelled by its position in the batch, so two
    # files that share a name get their own labels.
    labels = [f"Submission {i}" for i in range(1, len(student_submissions) + 1)]
    print("\nThe committee sees anonymous labels only:")
    for label, submission in zip(labels, student_submissions):
        print(f"  {label} = {submission.metadata['source']}")

    summary: List[str] = []
    for label, submission in zip(labels, student_submissions):
        filename = Path(submission.metadata["source"]).name
        try:
            grade_json = await grade_single_submission(
                llm,
                submission,
                test_code_content,
                rubric_content,
                run_tests,
                label,
            )
            original_filename = Path(submission.metadata["source"]).stem
            report_filepath = output_path / f"{original_filename}_grade_report.txt"
            report_content = format_report(
                grade_json, Path(submission.metadata["source"]).name
            )
            report_filepath.write_text(report_content, encoding="utf-8")
            logger.info(f"Grade report saved to: {report_filepath}")
            summary.append(f"{label} ({filename}): {_score_line(grade_json)}")
        except Exception as e:
            logger.error(
                f"A critical error occurred while processing {submission.metadata.get('source', 'a submission')}. Skipping. Error: {e}",
                exc_info=True,
            )

    print("\n=== Grades ===")
    for line in summary:
        print(f"  {line}")
    logger.info("\n--- Programming Grading Batch Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-Agent AI Programming Autograder"
    )
    parser.add_argument(
        "--submissions",
        type=str,
        required=True,
        help="Directory with student code submissions.",
    )
    parser.add_argument(
        "--rubric",
        type=str,
        required=True,
        help="Path to the grading rubric .txt file.",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Directory to save grade reports."
    )
    parser.add_argument(
        "--tests",
        type=str,
        help="Path to the pytest unit tests file. Required unless --no-run is specified.",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Disable code execution. The grader will only perform static analysis.",
    )
    args = parser.parse_args()

    run_tests_flag = not args.no_run

    # Create dummy files and folders for demonstration if they don't exist
    Path(args.submissions).mkdir(exist_ok=True)
    Path(args.output).mkdir(exist_ok=True)

    if not list(Path(args.submissions).glob("*")):
        (Path(args.submissions) / "student1_assignment.py").write_text(
            "def add(a, b):\n    return a + b\n"
        )

    if run_tests_flag and args.tests and not Path(args.tests).exists():
        (Path(args.tests)).write_text(
            "from student_code import add\n\ndef test_add():\n    assert add(2, 3) == 5\n\ndef test_add_negative():\n    assert add(-1, -1) == -2\n"
        )

    if not Path(args.rubric).exists():
        (Path(args.rubric)).write_text(
            "- Correctness (10 pts): Passes all unit tests.\n- Style (5 pts): Follows PEP 8."
        )

    asyncio.run(
        main(args.submissions, args.rubric, args.output, args.tests, run_tests_flag)
    )
