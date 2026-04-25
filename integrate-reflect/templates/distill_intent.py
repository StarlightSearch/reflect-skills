"""Distill a multi-turn conversation into a one-sentence intent.

Used twice in conversational agents — early and late:

    early_intent = distill_intent(messages_so_far)        # mid-conversation
    # ... agent retrieves memories with early_intent as the query ...
    # ... conversation continues, intent may drift ...
    late_intent  = distill_intent(full_trajectory)        # at conversation end
    # ... memory written with late_intent as the task field ...

The split matters because conversational intent often drifts: a customer
who opens with "modify my reservation" may end the call with a cancellation.
Retrieval should use the early intent (when the agent decides what to do);
the memory written should reflect the late intent (the resolved goal).

When using the Reflect SDK context manager, pass early_intent to
``client.trace(task=early_intent)`` and pass late_intent to
``ctx.set_output(task=late_intent)`` — same single context manager,
both intents wired correctly. Single-task agents skip the late intent
entirely (early == late).

Falls back to the first user message on LLM failure so the agent loop is
never blocked by distillation. Use ``is_useful_intent()`` to gate memory
writes when the distiller couldn't produce a real one-sentence summary.
"""
from __future__ import annotations

from typing import Any

import litellm  # or use openai/anthropic directly — litellm just normalizes

# REPLACE this prompt with one that fits your domain. The current prompt is
# domain-agnostic and will give middling results on specialized agents.
_SYSTEM_PROMPT = (
    "You summarize a conversation into one precise sentence describing what "
    "the user actually wants. Focus on the actionable intent — what they want "
    "done, to what object, under what constraint. Omit identifiers, "
    "pleasantries, and meta-commentary. Output the sentence only, nothing else."
)


def _format_transcript(messages: list[Any]) -> str:
    """Render messages as `role: content` lines, skipping system/tool turns."""
    lines: list[str] = []
    for m in messages:
        # Support both dict-shaped and object-shaped messages.
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        if role in {None, "system", "tool"} or not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _first_user_message(messages: list[Any]) -> str:
    for m in messages:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        if role == "user" and content:
            return str(content)
    return ""


_SENTINEL_INTENTS = {
    "",
    "no conversation provided.",
    "no conversation provided",
    "no actionable intent detected.",
    "no actionable intent detected",
    "no actionable intent.",
    "no actionable intent",
}


def is_useful_intent(text: str | None) -> bool:
    """Return True when ``text`` is a real distilled intent worth writing.

    Use this as a gate before calling ``ctx.set_output(task=text)`` /
    ``client.create_trace(task=text)`` — writing a memory whose task is
    "no actionable intent" pollutes the bank with low-signal entries that
    later get retrieved by similar-category queries.
    """
    if not text:
        return False
    return text.strip().lower() not in _SENTINEL_INTENTS


def distill_intent(messages: list[Any], *, model: str = "claude-haiku-4-5") -> str:
    """Return a one-sentence summary of the user's intent.

    On any LLM failure (exception or empty response), falls back to the first
    user message. Returns the empty string only when neither is available —
    in that case skip retrieval rather than searching with no signal.
    """
    transcript = _format_transcript(messages)
    if not transcript:
        return ""
    try:
        resp = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"<transcript>\n{transcript}\n</transcript>"},
            ],
            temperature=0.0,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text:
            return text
    except Exception:  # noqa: BLE001 — distillation must never block the loop
        pass
    return _first_user_message(messages)
