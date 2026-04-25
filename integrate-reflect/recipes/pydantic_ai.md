# Recipe: Pydantic AI

Pydantic AI's `Agent.run(...)` and `Agent.run_sync(...)` are the natural
trace boundary. Pass `ctx.augmented_task` as the input.

## Detect

`pyproject.toml` has `pydantic-ai`. Imports include
`from pydantic_ai import Agent`.

## Starter snippet (fresh projects)

Use when the user has no agent yet. Copy into `agent.py`, run it once, *then* layer Reflect on top.

```python
"""Minimal Pydantic AI starter."""
from __future__ import annotations
import asyncio, os

from pydantic_ai import Agent

agent = Agent(
    "anthropic:claude-sonnet-4-6",
    system_prompt="You are a helpful assistant. Answer concisely.",
)


async def main(prompt: str) -> str:
    result = await agent.run(prompt)
    return str(result.output)


if __name__ == "__main__":
    print(asyncio.run(main(os.environ.get("PROMPT", "What is the capital of France?"))))
```

`pyproject.toml`:

```toml
[project]
dependencies = ["pydantic-ai>=0.0.13", "reflect-sdk>=0.5.0"]
```

## Single-task

```python
from pydantic_ai import Agent
from reflect_sdk import ReflectClient

reflect = ReflectClient(base_url="...", api_key="...", project_id="my-agent")
agent = Agent("anthropic:claude-sonnet-4-6", system_prompt="...")


async def solve(task: str) -> str:
    async with reflect.trace_async(
        task,
        limit=5, lambda_=0.1, similarity_threshold=0.2,
    ) as ctx:
        result = await agent.run(ctx.augmented_task)

        # --- Review wiring ---
        from llm_judge import judge_run
        verdict = judge_run(task=task, final_response=str(result.output))

        # Pydantic AI exposes structured messages via result.all_messages()
        trajectory = [
            {"role": m.kind, "content": str(m)} for m in result.all_messages()
        ]
        ctx.set_output(
            trajectory=trajectory,
            final_response=str(result.output),
            result=verdict.result if verdict else None,
            feedback_text=f"[auto-judge] {verdict.reason}" if verdict else None,
            alpha=0.1 if verdict else None,
        )
    return str(result.output)
```

## Conversational

Pydantic AI supports message history via `message_history=` kwarg. Distill
on each turn before retrieval:

```python
from distill_intent import distill_intent

class Conv:
    def __init__(self) -> None:
        self.history: list = []

    async def turn(self, user_msg: str) -> str:
        # Build a transcript for distillation
        transcript = [{"role": "user" if i % 2 == 0 else "assistant",
                       "content": str(m)} for i, m in enumerate(self.history)]
        transcript.append({"role": "user", "content": user_msg})
        intent = distill_intent(transcript) or user_msg

        async with reflect.trace_async(
            intent, limit=5, lambda_=0.1, similarity_threshold=0.2,
        ) as ctx:
            result = await agent.run(
                ctx.augmented_task + "\n\nLatest user message: " + user_msg,
                message_history=self.history,
            )
            self.history = result.all_messages()
            answer = str(result.output)

            ctx.set_output(
                trajectory=f"User: {user_msg}\nAssistant: {answer}",
                final_response=answer,
                result=None,    # judge at conversation end
            )
        return answer
```

## Notes

- Pydantic AI's `result.output` is structured (not a string) when the agent
  has a `result_type` — `str(result.output)` is fine for tracing but consider
  storing the structured form in `metadata` for richer reflections.
- `result.all_messages()` returns Pydantic model instances — convert to dicts
  before passing to `set_output(trajectory=...)`.
