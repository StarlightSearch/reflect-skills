---
name: integrate-reflect
description: Use when the user wants to add Reflect long-term memory to an existing Python agent, give an agent the ability to learn from past runs, wire client.trace into an agent loop, integrate the reflect-sdk, set up LLM-as-judge for an existing Reflect integration, or debug empty ctx.memories / stagnant q-values. Skip for general docs lookup (point at https://docs.starlight-search.com/llms-full.txt), MCP-only setup with no SDK code, or non-Python projects.
---

# Integrate Reflect

Walk a coding agent through adding Reflect long-term memory to a Python agent project end-to-end: SDK install, loop placement, parameter tuning, review wiring (LLM-as-judge or programmatic), and a smoke test that proves the loop closes.

**Canonical SDK reference (fetch first, do not skip):**
`https://docs.starlight-search.com/llms-full.txt` (text/plain — fetch with `curl` or `urllib`, NOT a JS-rendered HTML fetcher; Mintlify renders HTML on the JS-rendered URLs).

## When to use

Trigger on phrases like:
- "add Reflect to my agent"
- "give my agent memory"
- "make my agent learn from past runs"
- "wire up `client.trace`" / "integrate reflect-sdk"
- "set up an LLM judge for Reflect"
- "ctx.memories is empty" / "q-values aren't moving"

**Do NOT use for:** answering "what is Reflect" (docs site), setting up only the MCP server with no SDK code (different skill), or non-Python projects (Python SDK only as of v0.5).

## The Iron Law

**An integration is not done until the smoke test in step 6 passes.** Wiring code without verifying that a memory is created and then retrieved on a subsequent run is unproven and will silently fail in production. No exceptions.

### Offline / sandbox mode (no network or creds available)

When you're operating in an environment without network access or live API credentials (CI sandbox, fresh container, no `REFLECT_API_KEY` set):

- **Step 0 fetch:** skip the `curl` and `inspect.signature` calls. Use the canonical signatures embedded in this skill's "Quick reference" section as your source of truth — they are version-pinned and accurate.
- **Step 6 smoke test:** copy `templates/smoke_test.py` into the user's project anyway, but do NOT execute it. **Flag prominently** in your hand-off to the user that the integration is **unverified** and the smoke test is the first thing they must run with live env vars before relying on the integration. Do NOT mark the work as "done" — phrase it as "ready for user verification."

This is the correct behavior, not a discipline violation. Pretending to verify (e.g., by running the test against a mock) is worse — it gives false confidence.

## Step 0 — Fetch canonical reference & verify SDK shape

Before generating any integration code:

```bash
# Pull the full docs as plain text (~50KB):
curl -sS https://docs.starlight-search.com/llms-full.txt -o /tmp/reflect-docs.txt

# Confirm the SDK is installed in the user's env:
uv add reflect-sdk          # or pip install reflect-sdk

# Verify call signatures — the SDK evolves; do not trust your memory of it:
python -c "
import inspect
from reflect_sdk import ReflectClient
for name in ['__init__', 'trace', 'trace_async', 'create_trace', 'query_memories', 'review_trace']:
    fn = getattr(ReflectClient, name, None)
    if fn: print(f'{name}{inspect.signature(fn)}')
"
```

If `inspect.signature` reveals a kwarg you weren't planning to pass (e.g. `metadata`, `reference_context`, `alpha`), reconsider — those are usually load-bearing.

## Step 1 — Determine agent shape

**Ask the user (or detect from code):**

> "Is your agent **single-task** (each invocation runs one task to completion: research, code-fix, RAG bot, eval) or **conversational** (multi-turn dialog where intent emerges over several turns: customer support, agentic chat, copilot)?"

Single-task → straight `client.trace(task=<prompt>)`.
Conversational → install `templates/distill_intent.py` and call it before retrieval. **Raw conversation history is a bad search query.** A 10-turn refund call retrieves nothing useful when fed verbatim — distill it to one sentence first.

If conversational, also ask:

> "When in the conversation should retrieval fire?"
> - After the **first user message** (cheapest, fine for most support bots)
> - After **N turns** (default 3) — for chats where intent crystallizes slowly
> - On a **tool-use signal** — for agents that retrieve only when about to act
> - **Manually** — user code calls a `recall()` helper

> "Does the user's intent **drift** as the conversation progresses?" (e.g., a
> "modify" request that ends up being processed as a "cancel" once policy
> rules are applied; a "lookup" that becomes a "refund" once the customer
> sees the answer)
>
> - **Stable intent** (intent crystallizes by message 1–2): pass one
>   distilled intent to both `client.trace(task=...)` and submit. Single
>   `distill_intent` call, simpler.
> - **Drifting intent** (intent often changes mid-conversation): use the
>   **late-intent pattern** — distill *twice*. The early intent (from
>   partial transcript) drives retrieval; the late intent (from the full
>   transcript at conversation end) becomes the `task` on the written
>   memory. Wire it via `ctx.set_output(task=late_intent, ...)` — the SDK
>   uses `task=late_intent` on the write while retrieval already happened
>   with `task=early_intent`. Tag both `action_type` labels on the memory's
>   metadata so it's findable from either query side.

The simpler shape works for most chat bots; the drifting-intent pattern is
canonical for customer-service / policy-bound conversational agents (tau2,
support, e-commerce returns). When in doubt, ship the simpler shape and
upgrade later if your retrieval recall on past conversations seems poor.

## Step 2 — Ask the integration questions

Run through these with the user. Some are conditional. **Document the answers in the integration code as comments** so future readers know the choices.

1. **Framework** — detect from `pyproject.toml`. Confirm. Recipes available: `openai_agents`, `claude_agent_sdk`, `langgraph`, `pydantic_ai`, `generic`.
2. **(if conversational)** Retrieval trigger timing.
3. **Review source** — pick exactly one default + optional fallbacks:
   - **LLM-as-judge** (recommended default when no programmatic check exists) — install `templates/llm_judge.py`. Cheap fast model auto-grades each run inline.
   - **Programmatic** — test pass/fail, lint clean, tool error absent, user thumbs-up. Best signal when available.
   - **Human** — dashboard review only. Submit traces with `result=None` and let humans review later. *Last resort* — see "Why deferred-only is the failure mode" below.
   - **Mix** — programmatic when available, judge as fallback.
4. **Pass/fail definition** — *user-supplied*: "What does 'pass' mean for your agent?" Capture the answer verbatim as a docstring on the `result=` call site.
5. **Reference context** — "Does your agent operate against a fixed policy, schema, or ruleset that should always be in context?" If yes, set `reference_context=` on every `trace()`.
6. **Project granularity** — one project for the whole agent, per-user (multi-tenant), or per-environment (dev/staging/prod)? Affects `project_id`.
7. **(optional)** Action-type metadata? Install `templates/classify_action_type.py` if yes — enables filtered retrieval.
8. **Trajectory shape** — summary string (cheaper, recommended) or full message list?
9. **(if LLM-judge)** Judge model — default `claude-haiku-4-5` (cheap, fast, JSON-mode); override if domain needs more horsepower.

## Step 3 — Apply the framework recipe

Read `recipes/<framework>.md` for the exact loop placement. Each recipe shows:
- Where the `ReflectClient` is constructed
- Where `client.trace(...)` wraps the agent call
- Where distillation runs (conversational only)
- Where the judge runs (if LLM-judge picked)
- Where `set_output(...)` is called and what kwargs go on it

**Available recipes** (in `recipes/`):
- `openai_agents.md` — OpenAI Agents SDK, single-task or conversational
- `claude_agent_sdk.md` — Anthropic Claude Agent SDK
- `langgraph.md` — LangGraph (multi-turn graph node placement)
- `pydantic_ai.md` — Pydantic AI agents
- `generic.md` — bare-loop fallback for everything else

## Step 4 — Wire reviews

**Why deferred-only is the failure mode:** The default `result=None` is "I'll review later via the dashboard." Three out of four real-world integrations never get reviewed → q-values stuck at 0.5 → similarity-only ranking → users see no learning happening → users churn thinking Reflect is broken.

**Pick exactly one based on Q3:**

### LLM-as-judge (recommended default)

Copy `templates/llm_judge.py` into the user's project. Call it inline before `set_output`:

```python
from llm_judge import judge_run

verdict = judge_run(task=task, final_response=answer, model="claude-haiku-4-5")
ctx.set_output(
    trajectory=trajectory,
    final_response=answer,
    result=verdict.result if verdict else None,
    feedback_text=f"[auto-judge] {verdict.reason}" if verdict else None,
    alpha=0.1 if verdict else None,  # auto-judge is noisy; lower step size
)
```

**Mandatory caveat to surface to the user:** A generic judge will mis-grade ~10–30% of runs on real domains. Tell them explicitly: *"The judge prompt in `templates/llm_judge.py` is a starting point — replace it with one that knows your domain (rubric, examples of pass/fail) before you ship to production. The `alpha=0.1` override caps how much each noisy auto-grade can move a q-value."*

### Programmatic

```python
ctx.set_output(
    trajectory=trajectory,
    final_response=answer,
    result="pass" if test_outcome.passed else "fail",
    feedback_text=test_outcome.error_msg if not test_outcome.passed else None,
)
```

### Human (only when explicitly chosen)

```python
ctx.set_output(trajectory=trajectory, final_response=answer, result=None)
# User reviews in dashboard at https://reflect.starlight-search.com
```

## Step 5 — Tune parameters

**Defaults are calibrated for a corpus with ≥100 reviewed memories.** Cold-start integrations need different values or `ctx.memories` will be empty for the first weeks.

| Param | Cold-start (≤100 memories, no reviews yet) | Mature (≥100 memories, regular reviews) |
|---|---|---|
| `lambda_` | **0.1** | 0.5 |
| `similarity_threshold` | **0.2** | 0.5 |
| `limit` | 5 | 5 |

Why: with all `q_value=0.5` (no reviews yet), the q-term contributes a constant 0.25 with `lambda_=0.5`. Similarity is halved. Most blended scores land below the 0.5 threshold and retrieval returns empty. Lowering `lambda_` to 0.1 lets similarity dominate; lowering threshold to 0.2 keeps the long tail visible.

**Once you have ~100 memories with reviews, raise both back toward the mature defaults.** State this transition explicitly in a code comment so the user knows when to revisit.

| Param | Default | Override when |
|---|---|---|
| `Q_LEARNING_ALPHA` (env var) | `0.3` (per MemRL) | Lower to 0.1 if reviews are noisy/sparse; raise to 0.5 if reviews are very high-signal |
| Per-review `alpha` override | unset | `0.1` for LLM-judge calls (noisy); `0.5+` for high-confidence human reviews |

## Step 6 — Smoke test (NOT optional)

Copy `templates/smoke_test.py` into the user's project (or a `tests/` dir). Run it.

**The test does exactly two things:**

1. Submit a canned trace with `result="pass"`. Wait for ingest (poll `ingest_status` up to 30s).
2. Submit a *similar* canned task and assert `len(ctx.memories) >= 1`.

If step 2 fails, the integration is broken — surface the cause (auth / project scoping / threshold / API down) and STOP. Do not declare the integration done.

**Common rationalization to refuse:** *"User didn't ask for tests, I'll skip it."* The smoke test is the only thing that proves the integration works end-to-end. Skipping it = shipping unverified code = the user finds breakage in week 2 and blames Reflect.

## Quick reference — canonical SDK signatures

Compensates for the docs site being a JS SPA when accessed from agent contexts. Verify with `inspect.signature` (Step 0); these are accurate as of `reflect-sdk==0.5.x`:

```python
ReflectClient(
    base_url: str,
    api_key: str,
    project_id: str,
    timeout: float = 30.0,
)

# Context-manager API (recommended):
with client.trace(
    task: str,
    *,
    limit: int = 5,
    lambda_: float = 0.5,
    similarity_threshold: float = 0.5,
    reference_context: str | None = None,
    metadata: dict[str, Any] | None = None,
) as ctx:
    # ctx.augmented_task: str          (task + retrieved memory blocks)
    # ctx.memories: list[MemoryResponse]
    # ctx.retrieved_memory_ids: list[str]
    ctx.set_output(
        trajectory: list[dict] | str,         # message list OR summary string
        final_response: str,
        result: Literal["pass","fail"] | None = None,
        feedback_text: str | None = None,
        alpha: float | None = None,           # per-review Q-learning step override
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
        task: str | None = None,              # late-intent override (drifting-intent pattern)
    )

# Async variant:
async with client.trace_async(...) as ctx: ...

# Direct query (no trace recording):
memories: list[MemoryResponse] = client.query_memories(
    task: str,
    limit: int = 5,
    lambda_: float = 0.5,
    similarity_threshold: float = 0.5,
    metadata_filter: dict | None = None,
)
```

## Common mistakes & rationalizations

| Excuse | Reality |
|---|---|
| "Deferred review is fine — user will review later." | Three out of four integrations never get reviewed. Default to LLM-judge or programmatic. Deferred is opt-in only. |
| "I'll skip the smoke test, user just wants wiring." | Then the wiring is unverified. The smoke test is the integration proof. Mandatory. |
| "Intent distillation is overkill for a simple chat bot." | Multi-turn conversation as a search query returns garbage. One LLM call to distill is the difference between useful retrieval and empty memories. |
| "Defaults look fine — `lambda_=0.5`, `threshold=0.5`." | Those are *mature-corpus* defaults. Cold-start needs `lambda_=0.1`, `threshold=0.2` or retrieval returns empty for weeks. |
| "I don't know the kwarg name, I'll guess." | Run `inspect.signature` first. Guessing leads to silent kwarg-drop or `TypeError` at runtime. |
| "Generic judge prompt is good enough to start." | It will mis-grade 10–30% of runs. Wire it AND tell the user explicitly to replace the rubric with a domain-specific one before shipping. |
| "I'll fetch reflect.starlight-search.com for docs." | That's the JS-rendered console. Use `https://docs.starlight-search.com/llms-full.txt` (plain text). |
| "User has no human reviewer, just leave `result=None`." | Then q-values never update and the learning loop is dormant. Wire LLM-judge with `alpha=0.1` and tell the user. |
| "I'll skip `retrieved_memory_ids` on `set_output`." | Don't. Without them, no Q-learning credit assignment happens — retrieved memories never get rewarded for contributing to a successful run. (`client.trace()` passes them automatically — verify it actually does in the recipe.) |
| "Trajectory should always be the full message list." | Summary string is ~10× cheaper to embed and reflect on. Default to summary unless the user is doing fine-grained analysis. |
| "Distill once, use the same intent for retrieval and write." | Fine if intent is stable. If conversations drift (modify→cancel, lookup→refund), use the late-intent pattern: early intent for retrieval (`client.trace(task=early)`), late intent for the written memory (`ctx.set_output(task=late)`). Tag both `action_type` labels in metadata. |
| "Late-distilled intent looks empty or like 'no actionable intent'." | Don't write the memory. Use `is_useful_intent()` from `templates/distill_intent.py` to gate the write — empty intents pollute the bank with low-signal memories that get retrieved later by similar-category queries. |

## Red flags — STOP and revisit

- You wrote integration code without running `inspect.signature` first
- You're about to ship without running `templates/smoke_test.py`
- You picked `result=None` as the default review path without explicit user opt-in
- You're integrating a conversational agent without intent distillation
- You used `lambda_=0.5` and `similarity_threshold=0.5` for a fresh-from-zero project
- You haven't told the user the judge prompt needs domain customization

If any of these are true: stop, fix, re-run smoke test.

## Files in this skill

- `templates/distill_intent.py` — one-sentence intent distiller for conversational agents
- `templates/classify_action_type.py` — fixed-taxonomy classifier for metadata filtering
- `templates/llm_judge.py` — pass/fail auto-grader, JSON-mode, alpha=0.1 override
- `templates/smoke_test.py` — two-call create+retrieve verification
- `recipes/openai_agents.md` — OpenAI Agents SDK loop placement
- `recipes/claude_agent_sdk.md` — Claude Agent SDK loop placement
- `recipes/langgraph.md` — LangGraph node placement (incl. distillation node)
- `recipes/pydantic_ai.md` — Pydantic AI integration
- `recipes/generic.md` — bare loop fallback
- `doc_links.md` — symbolic name → URL map
