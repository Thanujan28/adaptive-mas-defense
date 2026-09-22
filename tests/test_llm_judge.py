"""
Tests for the Tier-3 LLM judge (Task K).

No Ollama backend: an injected fake LLM (recording client) exercises the
prompt/parse/cache path, and ``stub=True`` exercises the deterministic
heuristic. No network, no model download, no real cache files committed.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from security.llm_judge import (
    JUDGE_PROMPT,
    LLMJudge,
    JudgeVerdict,
)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    """Records prompts and returns a fixed two-line verdict."""

    def __init__(self, content: str = "CONTRADICTS: YES\nREASONING: reverses the claim") -> None:
        self.content = content
        self.prompts: list[str] = []
        self.calls = 0

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        self.calls += 1
        return _FakeResponse(self.content)


class StubModeTests(unittest.TestCase):

    def test_stub_flags_explicit_negation(self):
        judge = LLMJudge(stub=True)
        verdict = judge.judge(
            chunk_text="AI offers no benefit in cybersecurity.",
            evidence_text="AI detects threats in real time.",
            task="Assess AI in cyber security.",
        )
        self.assertIsInstance(verdict, JudgeVerdict)
        self.assertTrue(verdict.contradicts_evidence)
        self.assertTrue(verdict.stubbed)

    def test_stub_does_not_flag_neutral(self):
        judge = LLMJudge(stub=True)
        verdict = judge.judge(
            chunk_text="Penquins live in the Antarctic.",
            evidence_text="AI detects threats in real time.",
            task="Assess AI in cyber security.",
        )
        self.assertFalse(verdict.contradicts_evidence)


class PromptAndParseTests(unittest.TestCase):

    def test_narrow_structured_prompt_is_used(self):
        fake = _FakeLLM()
        judge = LLMJudge(llm=fake, cache_enabled=False)
        judge.judge(
            chunk_text="CHUNK-BODY",
            evidence_text="EVIDENCE-BODY",
            task="TASK-BODY",
        )
        prompt = fake.prompts[0]
        # The prompt is the narrow structured template, filled in.
        self.assertIn("CHUNK-BODY", prompt)
        self.assertIn("EVIDENCE-BODY", prompt)
        self.assertIn("TASK-BODY", prompt)
        self.assertIn("CONTRADICTS: YES or NO", prompt)
        self.assertIn("not merely unsupported", prompt.lower())
        self.assertEqual(prompt, JUDGE_PROMPT.format(
            evidence="EVIDENCE-BODY",
            chunk="CHUNK-BODY",
            task="TASK-BODY",
        ))

    def test_parses_yes_and_reasoning(self):
        judge = LLMJudge(
            llm=_FakeLLM("CONTRADICTS: YES\nREASONING: it reverses the claim"),
            cache_enabled=False,
        )
        verdict = judge.judge("c", "e", "t")
        self.assertTrue(verdict.contradicts_evidence)
        self.assertEqual(verdict.reasoning, "it reverses the claim")

    def test_parses_no(self):
        judge = LLMJudge(
            llm=_FakeLLM("CONTRADICTS: NO\nREASONING: only unsupported"),
            cache_enabled=False,
        )
        verdict = judge.judge("c", "e", "t")
        self.assertFalse(verdict.contradicts_evidence)
        self.assertEqual(verdict.reasoning, "only unsupported")

    def test_malformed_response_is_conservative_no(self):
        judge = LLMJudge(
            llm=_FakeLLM("I think maybe? unclear."), cache_enabled=False
        )
        verdict = judge.judge("c", "e", "t")
        self.assertFalse(verdict.contradicts_evidence)
        self.assertEqual(verdict.reasoning, "")


class CacheTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self._tmp.name) / "judge_cache"

    def tearDown(self):
        self._tmp.cleanup()

    def test_second_identical_call_is_served_from_cache(self):
        fake = _FakeLLM("CONTRADICTS: NO\nREASONING: r")
        judge = LLMJudge(llm=fake, cache_dir=self.cache_dir)
        first = judge.judge("chunk", "evidence", "task")
        second = judge.judge("chunk", "evidence", "task")

        self.assertEqual(fake.calls, 1)         # only one real call
        self.assertFalse(first.cached)
        self.assertTrue(second.cached)
        self.assertEqual(
            first.contradicts_evidence, second.contradicts_evidence
        )

    def test_different_inputs_hit_the_model_again(self):
        fake = _FakeLLM("CONTRADICTS: YES\nREASONING: r")
        judge = LLMJudge(llm=fake, cache_dir=self.cache_dir)
        judge.judge("chunkA", "evidence", "task")
        judge.judge("chunkB", "evidence", "task")
        self.assertEqual(fake.calls, 2)

    def test_cache_files_are_json_verdicts_only(self):
        fake = _FakeLLM("CONTRADICTS: YES\nREASONING: short")
        judge = LLMJudge(llm=fake, cache_dir=self.cache_dir)
        judge.judge("chunk", "evidence", "task")

        files = list(self.cache_dir.glob("*.json"))
        self.assertEqual(len(files), 1)
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        self.assertEqual(
            set(payload.keys()),
            {"contradicts_evidence", "reasoning"},
        )

    def test_stub_mode_writes_no_cache(self):
        judge = LLMJudge(stub=True, cache_dir=self.cache_dir)
        judge.judge("chunk", "evidence", "task")
        self.assertFalse(self.cache_dir.exists())


class TemperatureTests(unittest.TestCase):

    def test_judge_client_forces_temperature_zero(self):
        """
        The judge must build its own temperature-0 client rather than
        reusing the agents' shared one.
        """

        judge = LLMJudge(cache_enabled=False)
        client = judge._build_judge_llm()
        # ChatOllama stores temperature on the instance.
        temperature = getattr(client, "temperature", None)
        if temperature is None:
            temperature = getattr(
                getattr(client, "kwargs", {}), "get", lambda *_: None
            )("temperature")
        self.assertEqual(float(temperature), 0.0)


if __name__ == "__main__":
    unittest.main()
