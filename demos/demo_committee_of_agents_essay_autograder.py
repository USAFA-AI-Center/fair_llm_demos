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
     class. It delegates subtasks and calls the grading tool. It never
     copies the essay: the demo keeps the essay and the rubric in a case
     file, each seat's WorkerAgentTool attaches the essay to the subtask
     (the _compose_task extension point), and the committee's reports are
     filed into the case file off the event bus as each delegation
     completes. A long essay copied through a JSON tool_input is what a
     model cannot reproduce or escape reliably, so none of the case data
     travels that way.

2. content_analyst (the subject matter expert):
   - Focuses exclusively on the essay's content, analyzing the strength of
     arguments, the quality of evidence, and the depth of analysis. Its
     subtask also carries the committee reports already on file.

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
   - This is the key to fair and consistent grading: the
     grade_essay_from_rubric tool itself, bound to the case file. It fills
     out a structured JSON form from the instructor's rubric, the essay and
     every committee report on file, which forces the AI to justify every
     point awarded, and it refuses, typed, while a report is missing. Every
     field of its form is case data, so no agent sits between the manager
     and the tool to copy it.

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
  - Each seat has its own event bus, and the demo prints every tool call,
    every planner parse error (with the raw model output) and every loop
    guard per seat, so a run shows who did what.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ValidationError

from fairlib import (
    AgentEventBus,
    Example,
    FinalGrade,
    GradeEssayFromRubricTool,
    HuggingFaceAdapter,
    PromptBuilder,
    RAGQueryTool,
    RoleDefinition,
    SimpleAgent,
    SimpleRetriever,
    WorkerAgentTool,
    build_worker_manager,
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
from fairlib.modules.action.tools.grading_tool import GradeEssayInput
from fairlib.utils.autograder_utils import (
    create_agent,
    format_report,
    setup_knowledge_base,
)
from fairlib.utils.document_processor import DocumentProcessor


def _delegation_text(tool_input: object) -> str:
    """A delegation's input as the model wrote it: the subtask alone, or JSON."""
    if isinstance(tool_input, dict) and set(tool_input) == {"subtask"}:
        return str(tool_input["subtask"])
    if isinstance(tool_input, (dict, list)):
        return json.dumps(tool_input)
    return str(tool_input)


# Configure logger for this specific module
logger = logging.getLogger(__name__)

# The committee seats whose reports the manager files into the case file.
COMMITTEE_SEATS = frozenset(
    {"fact_checker", "clarity_style_checker", "content_analyst"}
)


# Step 2: Committee construction. Each grader is an ordinary stateless
# SimpleAgent from the shared autograder factory; nothing about a grader is
# committee-specific until WorkerAgentTool wraps it. Stateless matters: each
# delegation must be planned fresh, not against the history of the previous
# essay's delegations. The manager model learns what each grader is for from
# the WorkerAgentTool description in its rendered tool catalog, so the
# role_description is the grader's own role definition, the role its model is
# prompted with.
def create_grader(
    llm: AbstractChatModel,
    role_description: str,
    tools: Optional[List[AbstractTool]] = None,
    events: Optional[AgentEventBus] = None,
) -> SimpleAgent:
    """Build one stateless committee grader via the shared agent factory."""
    return create_agent(llm, role_description, tools, stateless=True, events=events)


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


# Step 3: The case file and the case-bound committee pieces
@dataclass
class CaseFile:
    """Everything the demo knows about one essay, held outside any model.

    The essay and the rubric are data the demo already has, so no model
    ever copies them into a JSON tool_input: the committee seats receive the
    essay from here with each delegation, and the grading tool reads the
    rubric, the essay and the reports from here. The committee's reports
    land here off the event bus as each delegation completes.
    """

    name: str
    essay: str
    rubric: str
    reports: Dict[str, str] = field(default_factory=dict)

    def brief(
        self, subtask: str, *, with_rubric: bool = False, with_reports: bool = False
    ) -> str:
        """The manager's subtask with the essay (and the rubric or reports) attached."""
        parts = [subtask, f"Essay under review ({self.name}):\n{self.essay}"]
        if with_rubric:
            parts.append(f"Grading rubric:\n{self.rubric}")
        if with_reports:
            filed = "\n\n".join(
                f"Report from {seat}:\n{text}" for seat, text in self.reports.items()
            )
            parts.append(
                "Committee reports on file so far:\n" + (filed or "(none yet)")
            )
        return "\n\n".join(parts)


class CaseSeat(WorkerAgentTool):
    """A committee seat whose every delegation carries the essay.

    _compose_task is WorkerAgentTool's extension point for the task text:
    the manager writes only what to do, and the case file supplies the essay
    (and, for a seat that judges against the rubric or builds on the others,
    the rubric and their reports).
    """

    def __init__(
        self,
        worker: SimpleAgent,
        case: CaseFile,
        *,
        with_rubric: bool = False,
        with_reports: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(worker, **kwargs)
        self._case = case
        self._with_rubric = with_rubric
        self._with_reports = with_reports

    def _compose_task(self, tool_input: BaseModel) -> str:
        return self._case.brief(
            super()._compose_task(tool_input),
            with_rubric=self._with_rubric,
            with_reports=self._with_reports,
        )


class CaseToolInput(BaseModel):
    """A case-bound tool takes no input: the case file holds all it reads."""


class GradeCaseTool(GradeEssayFromRubricTool):
    """GradeEssayFromRubricTool bound to the case file: the rubric aligner's seat.

    Every input of the grading form is case data (the rubric, the essay and
    the committee reports), so the tool reads them itself. It refuses,
    typed, while a report is missing, so the manager learns what to
    delegate.
    """

    input_schema = CaseToolInput
    description = (
        "Fills the rubric for the essay under review as a structured JSON "
        "grade, from the rubric, the essay and every committee report on "
        "file. Call it once every report is in; it needs no input."
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
        reports = self._case.reports
        return await super().acall(
            GradeEssayInput(
                rubric=self._case.rubric,
                content_feedback=reports["content_analyst"],
                style_feedback=reports["clarity_style_checker"],
                fact_check_results=reports.get(
                    "fact_checker", "No course materials were provided."
                ),
                essay=self._case.essay,
            )
        )


# Step 4: Main Essay Grading Orchestration
async def grade_single_essay(
    llm: AbstractChatModel, essay_doc, rubric, knowledge_base, label: str
):
    """
    Orchestrates the entire multi-agent grading process for one essay.
    This function sets up the agent committee and the manager prompt.

    The committee sees the essay only under its anonymous label (for
    example "Essay 1"), never its file name, so a name such as
    student1_excellent cannot reach a reviewer and bias the grade.
    """
    essay_filename = Path(essay_doc.metadata.get("source", "unknown_essay")).name
    print(f"\n=== Grading {label} ({essay_filename}) ===")
    case = CaseFile(name=label, essay=essay_doc.page_content, rubric=rubric)

    # One bus per seat, each printing its seat's tool calls and parse
    # errors. The manager's bus also files every completed report into the
    # case file and captures the grading tool's typed result, so the demo
    # takes the structured grade at its source instead of trusting the
    # manager model to echo JSON verbatim (consumers subscribe to typed
    # events; they do not parse them back out of model text).
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

    # Create the Grading Committee: each specialist is a stateless
    # SimpleAgent wrapped as a typed tool. Every grader is READ_ONLY - they
    # retrieve, analyze, and compute, but mutate nothing - which is the
    # declaration that lets independent delegations issued in one manager
    # turn run concurrently. A grader that wrote files or updated a
    # gradebook would keep WorkerAgentTool's conservative EXTERNAL default
    # and run as a sequential barrier instead.
    worker_tools: List[AbstractTool] = []
    required_reports = ["clarity_style_checker", "content_analyst"]

    # Conditionally create the fact checker only if materials were provided.
    if knowledge_base:
        fact_checker = create_grader(
            llm,
            "A research assistant. Pick the essay's three to five central "
            "factual claims and verify each with one 'search_knowledge_base' "
            "query against the course materials. Then give, as your final "
            "answer, a verdict per claim: supported, contradicted, or not "
            "covered by the materials.",
            [RAGQueryTool(SimpleRetriever(knowledge_base.vector_store))],
            events=_watch_seat(AgentEventBus(), "fact_checker"),
        )
        worker_tools.append(
            CaseSeat(
                fact_checker,
                case,
                name="fact_checker",
                description=(
                    "Delegate fact-checking to a research assistant who "
                    "verifies the essay's claims against the course "
                    "materials via RAG retrieval. The essay is attached to "
                    "the subtask for you."
                ),
                side_effect=SideEffect.READ_ONLY,
            )
        )
        required_reports.insert(0, "fact_checker")

    content_analyst = create_grader(
        llm,
        "A university professor. Judge the essay's content critically "
        "against the rubric you are given: the strength of the argument, the "
        "quality and specificity of the evidence, the use of course concepts "
        "and IPCC findings, the depth across environmental, social and "
        "economic dimensions and equity, and the solutions it discusses. "
        "Take any committee reports you are given into account. Name concrete "
        "strengths and concrete weaknesses, and say how well the essay meets "
        "each content criterion, as your final answer.",
        events=_watch_seat(AgentEventBus(), "content_analyst"),
    )
    clarity_checker = create_grader(
        llm,
        "A university writing tutor. Judge the essay's grammar, spelling, "
        "sentence structure, paragraphing, academic tone and citations "
        "critically, quoting concrete errors, and say how well it meets the "
        "rubric's writing and citation criteria, as your final answer.",
        events=_watch_seat(AgentEventBus(), "clarity_style_checker"),
    )
    worker_tools.extend(
        [
            CaseSeat(
                content_analyst,
                case,
                with_rubric=True,
                with_reports=True,
                name="content_analyst",
                description=(
                    "Delegate content analysis to a professor who evaluates "
                    "argument strength, evidence, and depth. The essay, the "
                    "rubric and the committee reports on file are attached "
                    "to the subtask for you."
                ),
                side_effect=SideEffect.READ_ONLY,
            ),
            CaseSeat(
                clarity_checker,
                case,
                with_rubric=True,
                name="clarity_style_checker",
                description=(
                    "Delegate a writing-quality review to a writing tutor who "
                    "reports on grammar, clarity, and style only, ignoring "
                    "content accuracy. The essay and the rubric are attached "
                    "to the subtask for you."
                ),
                side_effect=SideEffect.READ_ONLY,
            ),
            # The rubric aligner's seat is the grading tool itself, bound to
            # the case file: every field of its form is case data, so no
            # model has to copy the rubric, the essay or the reports.
            GradeCaseTool(llm, case, required_reports),
        ]
    )

    # Application prompt content for the manager: fairlib ships only the
    # mandatory JSON format rules, and a small local model needs a role and
    # a worked example of the multi-action turn shape to hit the contract
    # reliably. Three example turns teach delegation, grading and the final
    # answer.
    manager_builder = PromptBuilder()
    manager_builder.role_definition = RoleDefinition(
        "You are the lead instructor coordinating an essay grading committee. "
        "You grade an essay by delegating subtasks to your committee tools "
        "and then calling the grading tool. The essay is on file: every tool "
        "receives it automatically, so a subtask says only what to do and "
        "never contains essay text."
    )
    # The first example delegates the checks that need no other report, and
    # names only seats that are registered: the fact checker exists only
    # when course materials were given.
    first_subtasks = {
        "fact_checker": "Verify the central factual claims of the essay.",
        "clarity_style_checker": "Review the grammar, clarity and style of the essay.",
    }
    first_calls = [
        {"tool_name": tool.name, "tool_input": {"subtask": first_subtasks[tool.name]}}
        for tool in worker_tools
        if tool.name in first_subtasks
    ]
    first_thought = (
        "The two checks are independent, so I delegate both in one turn."
        if len(first_calls) > 1
        else "The writing review needs no other report, so I delegate it first."
    )
    manager_builder.examples.append(
        Example(
            "User: Grade the essay on file.\n"
            "Assistant: "
            + json.dumps({"thought": first_thought, "actions": first_calls})
        )
    )
    manager_builder.examples.append(
        Example(
            "Observation: [content_analyst] <the last report>\n"
            'Assistant: {"thought": "Every report is in, so I grade.", '
            '"actions": [{"tool_name": "grade_essay_from_rubric", "tool_input": {}}]}'
        )
    )
    manager_builder.examples.append(
        Example(
            'Observation: [grade_essay_from_rubric] {"graded_criteria": [...], '
            '"overall_feedback": "...", "final_score": 72}\n'
            'Assistant: {"thought": "The grade is on file, so I finish.", '
            '"actions": [{"tool_name": "final_answer", "tool_input": '
            '{"text": "Graded: final score 72."}}]}'
        )
    )
    manager_builder.examples.append(
        Example(
            "Observation: [grade_essay_from_rubric] Error in tool 'grade_essay_from_rubric': No grade "
            "was produced ...\n"
            'Assistant: {"thought": "Grading failed once, so I call it once '
            'more.", "actions": [{"tool_name": "grade_essay_from_rubric", '
            '"tool_input": {}}]}\n'
            "Observation: [grade_essay_from_rubric] Error in tool 'grade_essay_from_rubric': No grade "
            "was produced ...\n"
            'Assistant: {"thought": "Grading failed twice, so there is no '
            'grade to report.", "actions": [{"tool_name": "final_answer", '
            '"tool_input": {"text": "No grade was produced: the grading '
            'tool failed twice."}}]}'
        )
    )

    # The manager is a plain SimpleAgent over the worker tools and the
    # grading tool: no special orchestrator class, no separate roster.
    manager_agent = build_worker_manager(
        llm, worker_tools, prompt_builder=manager_builder, max_steps=10, events=bus
    )

    # The delegation workflow in the manager's prompt. Independent analyses
    # may be delegated together in one turn; because the graders are
    # READ_ONLY, such a batch runs concurrently.
    workflow_steps = [
        "Delegate to the `clarity_style_checker` tool to get a report on "
        "writing quality."
    ]
    # Conditionally add the fact-checking step if it's available.
    if knowledge_base:
        workflow_steps.insert(
            0,
            "Delegate to the `fact_checker` tool to verify the essay's factual claims.",
        )
        workflow_steps.insert(
            2,
            "The fact-checking and writing-quality delegations are "
            "independent of each other; issue both in the same turn and "
            "they will run concurrently.",
        )

    workflow_steps.extend(
        [
            "After those reports are in, delegate to the `content_analyst` "
            "tool; it receives the essay and the reports on file.",
            "Once every report is in, call `grade_essay_from_rubric`.",
            "STOP CONDITION: once grade_essay_from_rubric returns a grade, "
            "do NOT delegate again; your next turn is the single final_answer "
            "action stating that grade's final_score. If "
            "grade_essay_from_rubric reports an error, call it once more; if "
            "it fails again, your final_answer says "
            "that no grade was produced and states no score.",
        ]
    )

    workflow_text = "".join(
        f"{i + 1}. {step}\n" for i, step in enumerate(workflow_steps)
    )
    # The manager's prompt carries the workflow, never the essay: the essay
    # reaches each seat from the case file.
    manager_prompt = (
        f"Coordinate your committee to grade the student essay {label} "
        "on climate change and food security.\n\n"
        f"Workflow Steps:\n{workflow_text}"
    )

    try:
        final_evaluation = await manager_agent.arun(manager_prompt)
        logger.info(f"Successfully completed agent run for {essay_filename}")
        print(f"[manager] final answer:\n{final_evaluation}")
        # The grading tool's typed GradeResult, captured off the event bus
        # at its source, is the authoritative structured grade; the
        # manager's final text is presentation on top of it. Fall back to
        # the raw text only when the grading tool never produced a
        # validated grade.
        if structured_grades:
            return structured_grades[-1]
        return final_evaluation
    except Exception as e:
        logger.error(
            f"The multi-agent run failed for {essay_filename}: {e}", exc_info=True
        )
        return json.dumps(
            {
                "error": f"A critical error occurred during the agent execution for this essay. Details: {e}"
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


# Main execution block
async def main(essays_dir, rubric_path, output_dir, materials_dir):
    """Main function to run the batch grading process."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    doc_proc = DocumentProcessor()
    # The prompt needs the rubric as one plain-text block, and grading
    # judges each essay whole with one report per student, so both loads
    # use the processor's whole-file surface rather than its RAG chunking.
    try:
        rubric_content = doc_proc.read_file_text(str(Path(rubric_path)))
    except ImportError as e:
        logger.critical(f"Could not load rubric from '{rubric_path}': {e}. Exiting.")
        return
    if not rubric_content:
        logger.critical(f"Could not load rubric from '{rubric_path}'. Exiting.")
        return

    knowledge_base = setup_knowledge_base(materials_dir) if materials_dir else None
    student_essays = doc_proc.load_whole_documents(essays_dir)

    if not student_essays:
        logger.warning(f"No essays found in '{essays_dir}'. Exiting.")
        return

    # One model plays every seat for the whole batch. The grade must tell an
    # excellent submission from an average one, and a 7B grader scores them
    # alike, so the default is a 14B instruct model; override with
    # FAIR_LLM_DEMO_MODEL to experiment.
    llm = HuggingFaceAdapter(
        os.environ.get("FAIR_LLM_DEMO_MODEL", "Qwen/Qwen2.5-14B-Instruct"),
        max_new_tokens=2048,
    )

    # Process each essay, wrapping the main call in error handling
    # This ensures that one failed essay does not stop the entire batch.
    # The committee grades anonymous labels; the mapping back to file names
    # stays with the demo and is printed here and in the summary.
    # Each essay is labelled by its position in the batch, so two files that
    # share a name get their own labels.
    labels = [f"Essay {i}" for i in range(1, len(student_essays) + 1)]
    print("\nThe committee sees anonymous labels only:")
    for label, essay in zip(labels, student_essays):
        print(f"  {label} = {essay.metadata.get('source', 'unknown')}")

    summary: List[str] = []
    for i, (label, essay) in enumerate(zip(labels, student_essays)):
        name = Path(essay.metadata.get("source", f"essay {i}")).name
        try:
            grade_json = await grade_single_essay(
                llm, essay, rubric_content, knowledge_base, label
            )
            original_filename = Path(essay.metadata["source"]).stem
            report_filepath = output_path / f"{original_filename}_grade_report.txt"
            report_content = format_report(grade_json, name)
            report_filepath.write_text(report_content, encoding="utf-8")
            logger.info(f"Grade report saved to: {report_filepath}")
            summary.append(f"{label} ({name}): {_score_line(grade_json)}")
        except Exception as e:
            logger.error(
                f"A critical error occurred while processing {name}. Skipping. Error: {e}",
                exc_info=True,
            )
            # Optionally, write an error report for the failed essay
            error_report_path = (
                output_path
                / f"{Path(essay.metadata.get('source', 'failed_essay')).stem}_error_report.txt"
            )
            error_report_path.write_text(
                f"Failed to grade this essay due to a critical error:\n{e}"
            )
            summary.append(f"{label} ({name}): [ERROR] {e}")

    print("\n=== Grades ===")
    for line in summary:
        print(f"  {line}")
    logger.info("\n--- Essay Grading Batch Complete ---")


if __name__ == "__main__":
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Multi-Agent AI Essay Autograder")
    parser.add_argument(
        "--essays", type=str, required=True, help="Directory with student essays."
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
        "--materials",
        type=str,
        default=None,
        help="Optional: Directory with course materials for RAG.",
    )
    args = parser.parse_args()

    # Create dummy directories and files for demonstration if they don't exist
    Path(args.essays).mkdir(exist_ok=True)
    if not list(Path(args.essays).glob("*")):
        (Path(args.essays) / "sample_essay.txt").write_text("This is a sample essay.")

    if not Path(args.rubric).exists():
        Path(args.rubric).write_text("- Thesis (10 pts): Clear and concise.")

    if args.materials:
        Path(args.materials).mkdir(exist_ok=True)

    # Run the main asynchronous function
    asyncio.run(main(args.essays, args.rubric, args.output, args.materials))
