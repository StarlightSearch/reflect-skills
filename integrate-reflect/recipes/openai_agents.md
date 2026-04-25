# Recipe: OpenAI Agents SDK

Wraps `Runner.run(agent, input=...)` in `client.trace_async(...)`. Works for
both single-task and conversational agents.

## Detect

`pyproject.toml` has `openai-agents`. Imports include `from agents import Agent, Runner, ...`.

## Starter snippet (fresh projects)

Use when the user has no agent yet. Copy this into `agent.py`, run it once to confirm output, *then* layer the Reflect integration below on top.

```python
"""Minimal OpenAI Agents SDK starter."""
from __future__ import annotations
import asyncio, os

from agents import Agent, Runner, WebSearchTool


async def main(prompt: str) -> str:
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant. Answer concisely.",
        tools=[WebSearchTool()],
        model="gpt-5.4",
    )
    result = await Runner.run(agent, input=prompt)
    return result.final_output


if __name__ == "__main__":
    print(asyncio.run(main(os.environ.get("PROMPT", "What is the capital of France?"))))
```

`pyproject.toml`:

```toml
[project]
dependencies = ["openai-agents>=0.4.0", "reflect-sdk>=0.5.0"]
```

## Single-task (one prompt → one answer)

```python
from agents import Agent, Runner, WebSearchTool
from reflect_sdk import ReflectClient
from reflect_sdk.converters import from_openai_agents

reflect = ReflectClient(base_url="...", api_key="...", project_id="research-agent")

async def research(topic: str) -> str:
    task = f"Research: {topic}"
    async with reflect.trace_async(
        task,
        # Cold-start: see SKILL.md step 5. Switch to lambda_=0.5, threshold=0.5
        # once you have ~100 reviewed memories.
        limit=5, lambda_=0.1, similarity_threshold=0.2,
        metadata={"topic": topic},
    ) as ctx:
        agent = Agent(name="Researcher", instructions="...", tools=[WebSearchTool()], model="gpt-5.4")
        result = await Runner.run(agent, input=ctx.augmented_task)

        # --- Review wiring (LLM-as-judge) ---
        from llm_judge import judge_run
        verdict = judge_run(task=task, final_response=result.final_output)

        ctx.set_output(
            trajectory=from_openai_agents(result),  # converter handles message shape
            final_response=result.final_output,
            result=verdict.result if verdict else None,
            feedback_text=f"[auto-judge] {verdict.reason}" if verdict else None,
            alpha=0.1 if verdict else None,
            metadata={"topic": topic},
        )
    return result.final_output
```

## Conversational (multi-turn)

Use `Runner` in a loop. Distill intent before retrieving:

```python
from distill_intent import distill_intent

async def chat() -> None:
    history: list[dict] = []
    agent = Agent(name="Support", instructions="...", model="gpt-5.4-mini")

    async def turn(user_msg: str) -> str:
        history.append({"role": "user", "content": user_msg})

        # Distill the conversation so far into a search-friendly intent.
        intent = distill_intent(history)
        if not intent:
            # First turn might be empty; fall back to the raw message.
            intent = user_msg

        async with reflect.trace_async(
            task=intent,                         # NOT the raw history — the distilled intent
            limit=5, lambda_=0.1, similarity_threshold=0.2,
        ) as ctx:
            result = await Runner.run(agent, input=ctx.augmented_task + "\n\nUser: " + user_msg)
            answer = result.final_output
            history.append({"role": "assistant", "content": answer})

            # For conversational: write a memory only at conversation end (see end_chat),
            # OR per-turn if each turn is independently graded. Default: per-turn with an
            # explicit summary trajectory.
            ctx.set_output(
                trajectory=f"User asked: {user_msg}\nAssistant: {answer}",
                final_response=answer,
                result=None,  # judge at end of conversation, not per-turn (cheaper)
            )
        return answer
    # ...
```

## Notes

- `from_openai_agents(result)` is the SDK's adapter for the Agents SDK
  `RunResult` — preserves tool calls and message structure.
- For long-running conversational agents, prefer to call the judge **at
  conversation end** (when the user signals satisfaction/exit), not per-turn.
- `Runner.run` is async. Use `client.trace_async`, not `client.trace`.
