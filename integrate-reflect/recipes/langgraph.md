# Recipe: LangGraph

LangGraph agents are almost always conversational, and conversational intent
often **drifts** as the dialog progresses (a "modify" request that ends as a
"cancel"). The canonical pattern uses **two distillations**:

- **Early intent** — distilled mid-conversation, used as the *retrieval query*.
- **Late intent** — distilled at conversation end, used as the *task* on the
  memory written.

Both go into a single `client.trace()` context manager: pass `task=early_intent`
at construction, then `ctx.set_output(task=late_intent)` to override at submit.

## Detect

`pyproject.toml` has `langgraph` and/or `langchain-*`. Imports include
`from langgraph.graph import StateGraph`.

## Node placement

```
classify (existing) → retrieve (NEW) → tools (existing) → respond (existing) → record (NEW) → END
```

## Code

```python
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict

from reflect_sdk import ReflectClient
from distill_intent import distill_intent, is_useful_intent
from classify_action_type import classify_action_type     # optional
from llm_judge import judge_run                            # optional but recommended

reflect = ReflectClient(base_url="...", api_key="...", project_id="support-bot")


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: str | None              # cheap classifier label
    early_intent: str | None        # one-sentence distilled at retrieval time
    early_action_type: str | None
    retrieved_memory_ids: list[str]
    # The TraceContext lives across nodes within one app.invoke call;
    # stash it in state so `record` can call set_output on the same one.
    _trace_ctx: object | None


def retrieve(state: State) -> State:
    """Distill conversation → query Reflect → open trace ctx → inject memories."""
    convo = [{"role": _role(m), "content": m.content} for m in state["messages"]]
    early_intent = distill_intent(convo) or (state["messages"][-1].content if state["messages"] else "")
    early_action_type = classify_action_type(early_intent) if early_intent else "other"

    # Open the trace context manually (no `with`) so it can span nodes.
    # Cold-start params; switch to 0.5/0.5 once you have ~100 reviewed memories.
    ctx = reflect.trace(
        task=early_intent,
        limit=5, lambda_=0.1, similarity_threshold=0.2,
        metadata={"action_types": [early_action_type]} if early_action_type != "other" else None,
    ).__enter__()

    msgs: list[BaseMessage] = []
    if ctx.memories:
        bullets = "\n".join(f"- {m.reflection}" for m in ctx.memories)
        if len(bullets) > 2000:
            bullets = bullets[:2000] + "\n…(truncated)"
        msgs.append(AIMessage(content=(
            f"<past_resolutions>\n{bullets}\n</past_resolutions>\n"
            "Use these prior resolutions when applicable; do not invent details."
        )))

    return {
        "messages": msgs,
        "early_intent": early_intent,
        "early_action_type": early_action_type,
        "retrieved_memory_ids": [m.id for m in ctx.memories],
        "_trace_ctx": ctx,
    }


def record(state: State) -> State:
    """At conversation end: distill late intent + judge + close the trace."""
    ctx = state.get("_trace_ctx")
    if ctx is None:
        return {}

    convo = [{"role": _role(m), "content": m.content} for m in state["messages"]]
    late_intent = distill_intent(convo)
    late_action_type = classify_action_type(late_intent) if late_intent else "other"

    bot_reply = state["messages"][-1].content if state["messages"] else ""
    trajectory = (
        f"User wanted: {state.get('early_intent', '')}. "
        f"Resolved as: {late_intent}. "
        f"Bot replied: {bot_reply[:500]}"
    )

    # Gate the write: skip when distillation produced no useful intent.
    if not is_useful_intent(late_intent):
        # Still close the context manually; just don't submit.
        ctx.__exit__(None, None, None)  # type: ignore[attr-defined]
        return {}

    # Tag BOTH action types — retrieval filters by early-type only, but
    # storing both makes a memory findable from either query side when
    # intent drifts (e.g. opener "modify" → resolution "cancel").
    early_at = state.get("early_action_type")
    action_types = sorted({t for t in (early_at, late_action_type) if t and t != "other"}) or ["other"]

    verdict = judge_run(task=late_intent, final_response=bot_reply)

    # Late-intent override: task on the WRITTEN memory becomes late_intent,
    # but retrieval already used early_intent (passed to client.trace above).
    ctx.set_output(  # type: ignore[attr-defined]
        task=late_intent,                         # ← key line: late intent wins on write
        trajectory=trajectory,
        final_response=bot_reply,
        result=verdict.result if verdict else None,
        feedback_text=f"[auto-judge] {verdict.reason}" if verdict else None,
        alpha=0.1 if verdict else None,
        metadata={"action_types": action_types},
    )
    ctx.__exit__(None, None, None)  # type: ignore[attr-defined]
    return {}


def _role(m: BaseMessage) -> str:
    return {"human": "user", "ai": "assistant", "system": "system"}.get(m.type, m.type)


def build_graph():
    graph = StateGraph(State)
    graph.add_node("classify", classify_intent)   # existing
    graph.add_node("retrieve", retrieve)
    graph.add_node("tools", use_tools)            # existing
    graph.add_node("respond", respond)            # existing
    graph.add_node("record", record)
    graph.set_entry_point("classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "tools")
    graph.add_edge("tools", "respond")
    graph.add_edge("respond", "record")
    graph.add_edge("record", END)
    return graph.compile()
```

## Why open the trace context manually

The standard `with reflect.trace(...) as ctx:` pattern works when retrieval
and recording happen in one Python block. In LangGraph, retrieval (`retrieve`
node) and recording (`record` node) are separate function calls — the `with`
block can't span them. Opening with `__enter__()` and closing with
`__exit__(None, None, None)` keeps the simple single-context-manager model
while letting state flow through the graph.

If your bot has multi-turn state (one `app.invoke` per turn), move the trace
boundary out of the graph: open in your turn handler before invoking the
graph, store on `state["_trace_ctx"]`, close after the graph finishes the
session-end turn. Same idea, different scope.

## Simpler shape (when intent doesn't drift)

If you've measured your conversations and intent doesn't drift (e.g., a FAQ
bot where the first user message *is* the goal), skip the late-distill
phase: pass the same early intent to both `client.trace(task=...)` and
`ctx.set_output(...)` (omit `task=` on `set_output`).

## Notes

- **Bye/thanks early-exit must route through the graph**, not `break` out of
  the loop — otherwise the `record` node never runs and the conversation's
  closing memory is never written.
- For multi-turn agents that span multiple `app.invoke` calls, write **one
  memory at session end**, not per-turn. Per-turn writes flood the bank with
  near-duplicate intents that drown out signal at retrieval.
- Action-type metadata uses `action_types` (plural) as the key, matching
  tau2-bench's pattern. Filter on `metadata_filter={"action_types": "refund"}`
  — the server matches when "refund" appears anywhere in the list.
