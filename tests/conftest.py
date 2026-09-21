"""
Shared pytest fixtures for the adaptive-mas-defense test suite.

The fixtures here let a test run a REAL episode through
``MASEnvironment.execute_task`` with no live LLM and no network:

  * ``stub_llm``     - patches ``get_llm`` in every agent module with a
                       deterministic stub whose responses are valid for
                       the coordinator -> outline -> researcher ->
                       executor pipeline.
  * ``stub_tools``   - swaps every real tool (internet_search, report
                       writer, calendar, email, ...) for an in-memory
                       stub so nothing touches the network, writes a
                       .docx, or mutates configs/mock_calendar.json.
  * ``stub_assessor``- a deterministic, download-free semantic
                       assessor.

These are used by tests/test_no_attack_vocabulary_leak.py (P7) and are
intended to be reused by any future end-to-end test that needs a
whole episode without the Ollama backend.
"""

from __future__ import annotations

import numpy as np
import pytest


# =================================================================
# Deterministic LLM stub
# =================================================================

# Legitimate agent prose for an AI-cybersecurity topic. It deliberately
# contains the words "attack" and "malicious": a correct P7 lint must
# NOT fail on agent-authored text that merely discusses attacks.
RESEARCHER_TEXT = (
    "Key points on AI in cyber security: machine-learning detectors "
    "flag malicious traffic, and an attacker who poisons training data "
    "can degrade them. Defences include robust feature engineering and "
    "continuous monitoring of suspicious behaviour."
)

OUTLINE_TOPIC = "AI in cyber security"

COORDINATOR_PLAN_JSON = (
    '{"outline": {"objective": "Outline the proposal on AI in cyber '
    'security", "tasks": ["list sub-topics"], "required_output": '
    '"topic and sub-topics"}, "research": {"objective": "Research AI '
    'in cyber security", "tasks": ["gather evidence"], '
    '"required_output": "key points"}, "execution": {"objective": '
    '"Write the proposal on AI in cyber security", "tasks": ["compile '
    'report"], "required_output": "final proposal"}}'
)

OUTLINE_JSON = (
    '{"topic": "AI in cyber security", "sub_topics": ['
    '{"title": "Malicious traffic detection", '
    '"focus": "how ML flags attacks", '
    '"guiding_questions": ["Which signals?"]}]}'
)

EXECUTOR_TEXT = (
    "Proposal: AI strengthens cyber security through malicious-traffic "
    "detection, while defenders must guard against attacks on the "
    "models themselves."
)

FINAL_TEXT = (
    "Final proposal on AI in cyber security covering detection of "
    "malicious activity and defences against attacks on models."
)


class _StubResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class StubLLM:
    """
    Deterministic stand-in for ``ChatOllama``.

    Chooses a canned response based on a marker in the prompt so the
    same object works for every agent. Records every prompt it saw.
    """

    def __init__(
        self,
        researcher_text: str = RESEARCHER_TEXT,
        encoder=None,
    ) -> None:
        self.prompts: list[str] = []
        self.researcher_text = researcher_text
        self.encoder = encoder

    def invoke(self, prompt: str):
        self.prompts.append(str(prompt))
        text = str(prompt)

        # Coordinator planning: asks for the three-stage JSON plan.
        if "Required JSON structure" in text or "Coordinator plan" in text:
            return _StubResponse(COORDINATOR_PLAN_JSON)

        # Coordinator final aggregation.
        if "performing the final check" in text or "Final answer:" in text:
            return _StubResponse(FINAL_TEXT)

        # Researcher tool decision: return "no tool" so no tool is
        # called (keeps the episode deterministic and offline).
        if '"need_tool"' in text or "need_tool" in text:
            return _StubResponse(
                '{"need_tool": false, "tool_name": null, "arguments": {}}'
            )

        # Outline agent.
        if "sub_topics" in text or "proposal outline" in text:
            return _StubResponse(OUTLINE_JSON)

        # Executor / everything else -> agent prose about cyber security.
        if "Executor" in text or "final proposal" in text:
            return _StubResponse(EXECUTOR_TEXT)

        return _StubResponse(self.researcher_text)


class StubSearchTool:
    """In-memory internet/academic search tool (no network)."""

    def __init__(self, sample_results=None) -> None:
        self.sample_results = sample_results or [
            {
                "id": "https://example.org/doc1",
                "title": "AI for Cyber Security",
                "url": "https://example.org/doc1",
                "source_url": "https://example.org/doc1",
                "score": 0.9,
                "snippet": "Machine learning improves detection of malicious traffic.",
                "content": (
                    "Survey of AI in cyber security: detectors learn to "
                    "spot malicious traffic, while attackers may try to "
                    "poison the training data."
                ),
                "content_status": "collected",
            }
        ]
        self.call_count = 0

    def search(self, query: str, max_results: int = 5):
        self.call_count += 1
        return [dict(item) for item in self.sample_results[:max_results]]


class StubReportWriter:
    """In-memory stand-in for the docx report writer."""

    def write_report(self, title=None, content=None, filename=None):
        return {
            "filename": filename or "stub_report.docx",
            "path": None,
            "status": "success",
        }


class StubCalendar:
    def list_events(self):
        return []

    def create_event(self, **kwargs):
        return {"status": "created", **kwargs}

    def get_event(self, event_id):
        return None

    def delete_event(self, event_id):
        return {"status": "deleted"}

    def reset(self):
        return None


class StubEmail:
    def send_email(self, to=None, subject=None, body=None, sender=None):
        return {"status": "sent", "to": to}

    def list_messages(self):
        return []

    def get_message(self, message_id):
        return None


# =================================================================
# Deterministic semantic assessor (no model download)
# =================================================================

class StubEncoder:
    """Bag-of-keywords encoder: deterministic and download-free."""

    KEYWORDS = (
        "security",
        "prompt",
        "injection",
        "ignore",
        "instructions",
        "risk",
        "malicious",
        "data",
    )

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=False):
        rows = []
        for text in texts:
            lowered = str(text).lower()
            rows.append([float(lowered.count(k)) for k in self.KEYWORDS])
        array = np.asarray(rows, dtype=np.float32)
        if array.size and not array.any(axis=1).all():
            array[~array.any(axis=1), 0] = 1e-6
        return array


# =================================================================
# Fixtures
# =================================================================

@pytest.fixture
def stub_llm(monkeypatch):
    """
    Patch ``get_llm`` in every agent module so MASEnvironment builds
    its agents with a deterministic offline LLM.
    """
    import agents.coordinator as coordinator_module
    import agents.outline as outline_module
    import agents.researcher as researcher_module
    import agents.analyst as analyst_module
    import agents.executor as executor_module

    stub = StubLLM()
    for module in (
        coordinator_module,
        outline_module,
        researcher_module,
        analyst_module,
        executor_module,
    ):
        monkeypatch.setattr(module, "get_llm", lambda: stub)
    return stub


@pytest.fixture
def stub_tools():
    """
    Return a callable that swaps every tool on an environment's
    ToolManager for an in-memory stub (no network, no docx, no config
    mutation).
    """
    def _install(env):
        tools = env.tool_manager.tools
        search = StubSearchTool()
        for name in ("internet_search", "academic_search"):
            if name in tools:
                tools[name] = search
        if "report_writer" in tools:
            tools["report_writer"] = StubReportWriter()
        if "mock_calendar" in tools:
            tools["mock_calendar"] = StubCalendar()
        if "mock_calender" in tools:
            tools["mock_calender"] = StubCalendar()
        if "mock_email" in tools:
            tools["mock_email"] = StubEmail()
        return env

    return _install


@pytest.fixture
def stub_assessor():
    """A deterministic, download-free SemanticAssessor."""
    from security.semantic_assessor import SemanticAssessor

    return SemanticAssessor(model=StubEncoder())
