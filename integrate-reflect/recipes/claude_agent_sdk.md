# Recipe: Claude Agent SDK (Anthropic)

Wraps the `claude_agent_sdk.query(...)` or `Agent` invocation in
`client.trace(...)`. The SDK is sync-friendly; both `trace()` and
`trace_async()` work.

## Detect

`pyproject.toml` has `claude-agent-sdk` or `anthropic[agent]`. Imports include
`from claude_agent_sdk import ...`.

## Starter snippet (fresh projects)

Use when the user has no agent yet. Copy into `agent.py`, run it once, *then* layer Reflect on top.

```python
"""Minimal Claude Agent SDK starter."""
from __future__ import annotations
import os

from claude_agent_sdk import query, ClaudeAgentOptions


def main(prompt: str) -> str:
    options = ClaudeAgentOptions(
        system_prompt="You are a helpful assistant. Answer concisely.",
        max_turns=5,
    )
    parts: list[str] = []
    for msg in query(prompt=prompt, options=options):
        parts.append(str(msg.content))
    return parts[-1] if parts else ""


if __name__ == "__main__":
    print(main(os.environ.get("PROMPT", "What is the capital of France?")))
```

`pyproject.toml`:

```toml
[project]
dependencies = ["claude-agent-sdk>=0.1.0", "reflect-sdk>=0.5.0"]
```

## Single-task

```python
from claude_agent_sdk import query, ClaudeAgentOptions
from reflect_sdk import ReflectClient

reflect = ReflectClient(base_url="...", api_key="...", project_id="my-agent")

def solve(task: str) -> str:
    with reflect.trace(
        task,
        limit=5, lambda_=0.1, similarity_threshold=0.2,
    ) as ctx:
        # Inject retrieved memories into the system prompt or as the first user message.
        options = ClaudeAgentOptions(
            system_prompt=ctx.augmented_task,    # contains task + memory blocks
            max_turns=10,
        )
        messages: list[dict] = []
        for msg in query(prompt=task, options=options):
            messages.append({"role": msg.role, "content": str(msg.content)})

        final = messages[-1]["content"] if messages else ""

        # --- Review wiring ---
        from llm_judge import judge_run
        verdict = judge_run(task=task, final_response=final, model="claude-haiku-4-5")

        ctx.set_output(
            trajectory=messages,
            final_response=final,
            result=verdict.result if verdict else None,
            feedback_text=f"[auto-judge] {verdict.reason}" if verdict else None,
            alpha=0.1 if verdict else None,
        )
    return final
```

## Conversational

Distill the conversation to retrieve, write memory at conversation end:

```python
from distill_intent import distill_intent

class Conversation:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    def turn(self, user_msg: str) -> str:
        self.messages.append({"role": "user", "content": user_msg})
        intent = distill_intent(self.messages) or user_msg

        with reflect.trace(intent, limit=5, lambda_=0.1, similarity_threshold=0.2) as ctx:
            options = ClaudeAgentOptions(system_prompt=ctx.augmented_task, max_turns=5)
            for msg in query(prompt=user_msg, options=options):
                self.messages.append({"role": msg.role, "content": str(msg.content)})
            answer = self.messages[-1]["content"]

            # Light per-turn record. Judge end-to-end at conversation close.
            ctx.set_output(
                trajectory=f"User: {user_msg}\nAssistant: {answer}",
                final_response=answer,
                result=None,
            )
        return answer
```

## Notes

- `ctx.augmented_task` is plain text — passes cleanly into `system_prompt`.
- The Claude SDK's streaming `query()` returns one message at a time; collect
  them into a list before passing to `set_output`.
- For tool-calling agents, include tool calls in the trajectory so the
  judge and reflection have the full picture.
