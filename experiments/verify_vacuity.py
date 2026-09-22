"""
experiments/verify_vacuity.py

Task G.2: a reusable NON-VACUITY check for the negative controls in
``tests/test_no_attack_vocabulary_leak.py``.

The negative controls there assert that a deliberately-leaked field makes
the lint raise ``AssertionError``. A control can pass vacuously: if the
guard it is meant to catch is itself broken (e.g. disabled or never
reached), the control could still "pass" for the wrong reason, or a future
refactor could silently neuter it. This script proves each control has
teeth by TEMPORARILY disabling exactly the guard the control exercises,
confirming the control then FAILS, then restoring the file byte-for-byte.

It never leaves the test file modified: the original contents are held in
memory and rewritten in a ``finally`` block, and the final contents are
compared against the original.

Usage:
    python -m experiments.verify_vacuity

Exit code 0 iff every control is non-vacuous and the file is restored.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TEST_PATH = Path("tests/test_no_attack_vocabulary_leak.py")

# (name, test id, guard anchor text, disabled replacement)
#
# Each guard anchor must appear EXACTLY ONCE in the test file; the script
# refuses to run otherwise so a refactor cannot silently mis-target it.
CASES: tuple[tuple[str, str, str, str], ...] = (
    (
        "metadata-vocabulary",
        "test_negative_control_attack_vocabulary_in_metadata_fails",
        (
            '        metadata = event.get("metadata", {})\n'
            "        hits = _vocabulary_hits(metadata)\n"
            '        assert not hits, (event_type, "metadata", event, hits)\n'
        ),
        (
            '        metadata = event.get("metadata", {})\n'
            "        hits = _vocabulary_hits(metadata)\n"
            "        pass  # GUARD DISABLED FOR VACUITY CHECK\n"
        ),
    ),
    (
        "environment-content",
        "test_negative_control_attack_vocabulary_in_environment_content_fails",
        (
            "        if event_type in ENVIRONMENT_AUTHORED_EVENT_TYPES:\n"
            '            hits = _vocabulary_hits(event.get("content"))\n'
            '            assert not hits, (event_type, "content", event, hits)\n'
        ),
        (
            "        if event_type in ENVIRONMENT_AUTHORED_EVENT_TYPES:\n"
            '            hits = _vocabulary_hits(event.get("content"))\n'
            "            pass  # GUARD DISABLED FOR VACUITY CHECK\n"
        ),
    ),
    (
        # The control injects a forbidden key into an ARTIFACT's
        # metadata, so it is the ARTIFACT loop's guard that must be
        # disabled for this control to have a chance to fail.
        "forbidden-key",
        "test_negative_control_forbidden_key_in_metadata_fails",
        "        assert not found, (artifact, found)\n",
        "        pass  # GUARD DISABLED FOR VACUITY CHECK\n",
    ),
)


def _run_test(test_id: str) -> bool:
    """Return True if the given test passes."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            f"{TEST_PATH}::{test_id}",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def main() -> int:
    if not TEST_PATH.exists():
        print(f"ERROR: {TEST_PATH} not found; run from the repo root.")
        return 2

    original = TEST_PATH.read_text(encoding="utf-8")
    all_good = True

    try:
        print("=== baseline (guards intact): every control must PASS ===")
        for name, test_id, _anchor, _replacement in CASES:
            passed = _run_test(test_id)
            all_good &= passed
            print(f"  {name:22s} passes={passed}")

        print()
        print("=== guard disabled: every control must now FAIL (non-vacuous) ===")
        for name, test_id, anchor, replacement in CASES:
            count = original.count(anchor)
            if count != 1:
                print(
                    f"  {name:22s} SKIPPED: guard anchor found "
                    f"{count} times (expected exactly 1)"
                )
                all_good = False
                continue

            TEST_PATH.write_text(
                original.replace(anchor, replacement), encoding="utf-8"
            )
            try:
                passed = _run_test(test_id)
            finally:
                TEST_PATH.write_text(original, encoding="utf-8")

            non_vacuous = not passed
            all_good &= non_vacuous
            verdict = (
                "has teeth (fails as expected)"
                if non_vacuous
                else "VACUOUS - control still passes with guard disabled"
            )
            print(f"  {name:22s} passes={passed}  -> {verdict}")
    finally:
        TEST_PATH.write_text(original, encoding="utf-8")

    restored = TEST_PATH.read_text(encoding="utf-8") == original
    print()
    print(f"test file restored byte-for-byte: {restored}")
    all_good &= restored

    print()
    print("RESULT:", "PASS (all controls non-vacuous)" if all_good else "FAIL")
    return 0 if all_good else 1


if __name__ == "__main__":
    raise SystemExit(main())
