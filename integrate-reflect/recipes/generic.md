# Recipe: Generic (no specific framework)

Fallback for anything not covered by another recipe. Wrap whatever your
"agent run" function is in `client.trace(...)`.

## When to use

- Custom agent loops written from scratch
- Direct `openai`/`anthropic` SDK calls without an agent framework
- LiteLLM-based loops
- Any other Python agent

## Starter snippet (fresh projects)

Use when the user has no agent yet and picked "Generic / not sure". Lowest-commitment scaffold — direct OpenAI SDK call. Copy into `agent.py`, run once, *then* layer Reflect on top.

```python
"""Minimal direct-SDK starter."""
from __future__ import annotations
import os

from openai import OpenAI

oai = OpenAI()


def solve(task: str) -> str:
    resp = oai.chat.completions.create(
        model="gpt-5.4",
        messages=[{"role": "user", "content": task}],
    )
    return resp.choices[0].message.content or ""


if __name__ == "__main__":
    print(solve(os.environ.get("TASK", "What is the capital of France?")))
```

`pyproject.toml`:

```toml
[project]
dependencies = ["openai>=1.50.0", "reflect-sdk>=0.5.0"]
```

Swap `openai` for `anthropic` (with a `client.messages.create` call) or `litellm.completion` if the user prefers — same shape.

## Single-task pattern

```python
from openai import OpenAI
from reflect_sdk import ReflectClient
from llm_judge import judge_run

reflect = ReflectClient(base_url="...", api_key="...", project_id="my-agent")
oai = OpenAI()


def solve(task: str) -> str:
    with reflect.trace(
        task,
        limit=5, lambda_=0.1, similarity_threshold=0.2,
    ) as ctx:
        resp = oai.chat.completions.create(
            model="gpt-5.4",
            messages=[{"role": "user", "content": ctx.augmented_task}],
        )
        answer = resp.choices[0].message.content or ""

        verdict = judge_run(task=task, final_response=answer)

        ctx.set_output(
            trajectory=f"User: {task}\nAssistant: {answer}",   # summary string is cheaper
            final_response=answer,
            result=verdict.result if verdict else None,
            feedback_text=f"[auto-judge] {verdict.reason}" if verdict else None,
            alpha=0.1 if verdict else None,
        )
    return answer
```

## Conversational pattern

```python
from distill_intent import distill_intent

class Bot:
    def __init__(self) -> None:
        self.history: list[dict] = []

    def turn(self, user_msg: str) -> str:
        self.history.append({"role": "user", "content": user_msg})
        intent = distill_intent(self.history) or user_msg

        with reflect.trace(
            intent,
            limit=5, lambda_=0.1, similarity_threshold=0.2,
        ) as ctx:
            resp = oai.chat.completions.create(
                model="gpt-5.4",
                messages=[{"role": "system", "content": ctx.augmented_task}] + self.history,
            )
            answer = resp.choices[0].message.content or ""
            self.history.append({"role": "assistant", "content": answer})

            ctx.set_output(
                trajectory=f"User wanted: {intent}. Assistant said: {answer[:500]}",
                final_response=answer,
                result=None,    # judge at conversation end, not per-turn
            )
        return answer
```

## Notes

- Trajectory as **summary string** is the right default for most agents
  (~10× cheaper to embed and reflect on than a full message list).
- Switch to a full `list[dict]` trajectory only when reflections need
  per-tool-call detail.
- `client.trace()` (sync) and `client.trace_async()` (async) have identical
  signatures — pick to match your loop.
