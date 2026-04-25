"""Classify a distilled intent into a fixed taxonomy for metadata filtering.

Use the resulting label as `metadata={"action_type": label}` on `client.trace()`
and as `metadata_filter={"action_type": label}` on `query_memories` to scope
retrieval to the same kind of task.

REPLACE the taxonomy below with one that fits your domain. The current set
is a customer-service example; coding agents might use
[refactor, bugfix, feature, doc, test], research agents might use
[lookup, synthesize, compare, summarize], etc.
"""
from __future__ import annotations

import litellm

# Edit this taxonomy for your domain.
_VALID_LABELS: tuple[str, ...] = ("cancel", "book", "modify", "refund", "inquire", "other")

_SYSTEM_PROMPT = (
    "You classify a one-sentence user intent into a fixed taxonomy. "
    "Output exactly one of these labels, lowercase, nothing else:\n"
    + "\n".join(f"- {label}" for label in _VALID_LABELS)
    + "\nIf none clearly fit, output 'other'."
)


def classify_action_type(intent: str, *, model: str = "claude-haiku-4-5") -> str:
    """Return one label from `_VALID_LABELS`. Returns 'other' on any failure."""
    if not intent or not intent.strip():
        return "other"
    try:
        resp = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": intent.strip()},
            ],
            temperature=0.0,
        )
        text = (resp.choices[0].message.content or "").strip().lower()
        text = text.strip("\"'` .,:;")
        if text in _VALID_LABELS:
            return text
    except Exception:  # noqa: BLE001
        pass
    return "other"
