"""
Central location for model-name constants used across the security
stack and the experiment scripts.

This exists so the semantic encoder name is defined ONCE. Before this,
the literal ``"sentence-transformers/all-MiniLM-L6-v2"`` was copy-pasted
into five files (security/semantic_assessor.py and four experiment
scripts), which is exactly the kind of duplication that silently drifts:
a model swap in one script would not be reflected in another, and the
paper's "same encoder everywhere" claim would stop being true.

Import from here instead of repeating the literal.
"""

from __future__ import annotations

# The semantic (Tier-1) sentence encoder. This is the single source of
# truth; ``security/semantic_assessor.py`` uses it as its default and the
# experiment scripts import it as their ``MODEL_NAME``.
SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"