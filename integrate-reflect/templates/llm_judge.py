"""LLM-as-judge: auto-grade an agent run as pass/fail so q-values can update.

Returns None on any failure (API error, parse error, invalid result) —
fall back to result=None on the trace rather than poisoning q-values with a
fabricated label.

KNOWN LIMITATION: The system prompt below is domain-agnostic and will
mis-grade 10-30% of runs on real domains. Before shipping to production:
  1. Replace _SYSTEM_PROMPT with a rubric that knows your domain.
  2. Where possible, prefer programmatic checks (run tests, run a linter,
     check a tool result) over an LLM judge — they are cheaper and more
     accurate.
  3. Pass `alpha=0.1` on `set_output` for auto-grades so each noisy verdict
     contributes less to q-value updates than a human review would.
  4. Spot-check ~20 graded runs against a human before letting the judge
     drive learning at scale.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import litellm

logger = logging.getLogger(__name__)

# REPLACE THIS PROMPT for your domain. Add a domain rubric, examples, and
# anti-patterns. The default below is a generic best-effort starting point.
_SYSTEM_PROMPT = """You are an impartial grader for an agent's run.

You will be given a TASK and the agent's FINAL_RESPONSE. Decide whether the
response is a correct, useful, complete answer to the task.

Rubric:
- "pass": directly addresses the task, is technically correct, usable as-is,
  reasonably complete. Minor stylistic issues are fine.
- "fail": wrong, off-topic, refuses without good reason, buggy/non-runnable,
  or clearly incomplete.

Respond ONLY with a JSON object on one line, no prose, no code fences:
{"result": "pass" | "fail", "reason": "<one short sentence>"}"""


@dataclass
class JudgeVerdict:
    result: str       # "pass" or "fail"
    reason: str


def judge_run(
    *,
    task: str,
    final_response: str,
    trajectory: Any | None = None,    # optional; passed if your domain needs full context
    model: str = "claude-haiku-4-5",
) -> JudgeVerdict | None:
    """Return a JudgeVerdict, or None if the judge cannot reliably grade."""
    user_msg = f"TASK:\n{task}\n\nFINAL_RESPONSE:\n{final_response}\n\nGrade this run."
    if trajectory:
        user_msg = f"TRAJECTORY:\n{trajectory}\n\n" + user_msg

    try:
        resp = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw = resp.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("judge call failed: %s", exc)
        return None

    try:
        data: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("judge returned non-JSON: %r", raw[:200])
        return None

    result = str(data.get("result", "")).strip().lower()
    if result not in ("pass", "fail"):
        logger.warning("judge returned invalid result: %r", data)
        return None

    reason = str(data.get("reason", "")).strip() or "(no reason given)"
    return JudgeVerdict(result=result, reason=reason)
