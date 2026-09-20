"""
Golden-string tests for the per-agent assignment templates.

These pin the exact templates (and the final prompt built as
TEMPLATE + sample upstream text) against the strings that existed
in commit f21d60b ("Added security state"), the last commit before
the "Modified security state" refactor (33e9169) that introduced the
module-level *_ASSIGNMENT_TEMPLATE constants and, for the researcher,
silently dropped the trailing "OUTLINE TO RESEARCH:\n" label.

If one of these tests fails, the live prompt sent to an agent no
longer matches the pre-refactor prompt and must be investigated
before merging (regression class: P6).
"""

from __future__ import annotations

from environment.mas_environment import (
    ANALYST_ASSIGNMENT_TEMPLATE,
    COORDINATOR_ASSIGNMENT_TEMPLATE,
    EXECUTOR_ASSIGNMENT_TEMPLATE,
    OUTLINE_ASSIGNMENT_TEMPLATE,
    RESEARCHER_ASSIGNMENT_TEMPLATE,
)

# Golden templates, copied verbatim from `git show f21d60b:environment/mas_environment.py`.

GOLDEN_OUTLINE_TEMPLATE = (
    "Role: Outline (step 1 of 3). Create the proposal "
    "outline for the original topic in the assignment "
    "below: one overall topic plus 4-8 researchable "
    "sub-topics. Do NOT gather evidence and do NOT write "
    "the proposal. Output goes to the Researcher.\n\n"
)

GOLDEN_RESEARCHER_TEMPLATE = (
    "Role: Researcher (step 2 of 3). Gather evidence "
    "relevant to the proposal topic in the outline below. "
    "For EVERY sub-topic, report the important key points "
    "supported by the collected sources. Do not skip a "
    "sub-topic. Do NOT write the proposal. Output goes to "
    "the Executor.\n\n"
    "OUTLINE TO RESEARCH:\n"
)

GOLDEN_ANALYST_TEMPLATE = (
    "Perform the primary analysis using the "
    "available findings. Develop the central "
    "interpretation and conclusions.\n\n"
)

GOLDEN_EXECUTOR_TEMPLATE = (
    "Role: Executor (step 3 of 3). Write the final "
    "proposal using the outline (step 1) and the "
    "Researcher's key points (step 2). Do NOT gather new "
    "evidence and do NOT re-design the outline. The "
    "proposal must answer the ORIGINAL USER PROMPT. Output "
    "goes to the Coordinator for a final check.\n\n"
)

# The pre-refactor coordinator role text is not a module-level
# constant that is concatenated with upstream text anywhere; the
# coordinator's planning prompt is `self.coordinator.create_plan(task)`
# both before and after the refactor. COORDINATOR_ASSIGNMENT_TEMPLATE
# is only used to register the trusted assignment for semantic
# comparison, so there is no pre-refactor golden string to diff it
# against; we only pin its current value so accidental edits are caught.
GOLDEN_COORDINATOR_TEMPLATE = (
    "Role: Coordinator. Plan the three bounded pipeline "
    "roles (Outline, Researcher, Executor) for the original "
    "goal, then verify the final proposal against it.\n\n"
)

SAMPLE_UPSTREAM_TEXT = "Sample upstream content produced by the previous agent."


def test_outline_template_matches_pre_refactor():
    assert OUTLINE_ASSIGNMENT_TEMPLATE == GOLDEN_OUTLINE_TEMPLATE


def test_researcher_template_matches_pre_refactor():
    # This is the P6 regression: the trailing label was dropped by
    # the refactor and must be restored.
    assert RESEARCHER_ASSIGNMENT_TEMPLATE == GOLDEN_RESEARCHER_TEMPLATE


def test_analyst_template_matches_pre_refactor():
    assert ANALYST_ASSIGNMENT_TEMPLATE == GOLDEN_ANALYST_TEMPLATE


def test_executor_template_matches_pre_refactor():
    assert EXECUTOR_ASSIGNMENT_TEMPLATE == GOLDEN_EXECUTOR_TEMPLATE


def test_coordinator_template_current_value_pinned():
    assert COORDINATOR_ASSIGNMENT_TEMPLATE == GOLDEN_COORDINATOR_TEMPLATE


def test_final_prompts_with_sample_upstream_text():
    """
    Golden final prompts: TEMPLATE + sample upstream text, as built
    by the corresponding *_node methods in MASEnvironment.
    """

    assert (
        OUTLINE_ASSIGNMENT_TEMPLATE + SAMPLE_UPSTREAM_TEXT
        == GOLDEN_OUTLINE_TEMPLATE + SAMPLE_UPSTREAM_TEXT
    )
    assert (
        RESEARCHER_ASSIGNMENT_TEMPLATE + SAMPLE_UPSTREAM_TEXT
        == GOLDEN_RESEARCHER_TEMPLATE + SAMPLE_UPSTREAM_TEXT
    )
    assert (
        ANALYST_ASSIGNMENT_TEMPLATE + SAMPLE_UPSTREAM_TEXT
        == GOLDEN_ANALYST_TEMPLATE + SAMPLE_UPSTREAM_TEXT
    )
    assert (
        EXECUTOR_ASSIGNMENT_TEMPLATE + SAMPLE_UPSTREAM_TEXT
        == GOLDEN_EXECUTOR_TEMPLATE + SAMPLE_UPSTREAM_TEXT
    )


def test_registered_assignment_excludes_upstream_part():
    """
    Every *_ASSIGNMENT_TEMPLATE is the value registered as the
    trusted assignment (see `_register_assignment` call sites in
    MASEnvironment); none of them contain upstream/tool-derived
    content such as the sample text above.
    """

    for template in (
        OUTLINE_ASSIGNMENT_TEMPLATE,
        RESEARCHER_ASSIGNMENT_TEMPLATE,
        ANALYST_ASSIGNMENT_TEMPLATE,
        EXECUTOR_ASSIGNMENT_TEMPLATE,
        COORDINATOR_ASSIGNMENT_TEMPLATE,
    ):
        assert SAMPLE_UPSTREAM_TEXT not in template
