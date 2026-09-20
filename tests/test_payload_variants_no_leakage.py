"""
Guards against leakage between the held-out payload variants (Task
2c) and the detector's own patterns: these variants must never be
used to tune the detector, and — checked here — must not trivially
match a GENERIC pattern verbatim. If they did, TPR-on-variants would
be an artifact of the variant wording rather than a genuine test of
detector generalisation.
"""

from __future__ import annotations

from evaluation.payload_variants import PAYLOAD_VARIANTS
from security.content_detector import scan_text


def test_variants_do_not_verbatim_match_generic_patterns():
    for variant in PAYLOAD_VARIANTS:
        evidence = scan_text(variant.text, use_template_signatures=False)
        assert evidence.total == 0, (
            f"variant {variant.variant_id!r} matched a GENERIC pattern "
            f"verbatim: {evidence.matched_examples}"
        )


def test_at_least_twelve_variants_defined():
    assert len(PAYLOAD_VARIANTS) >= 12


def test_variants_do_not_contain_legacy_indicator_phrases():
    from attacks.prompt_infection import check_infection_indicators

    for variant in PAYLOAD_VARIANTS:
        assert check_infection_indicators(variant.text) is False, (
            variant.variant_id
        )
