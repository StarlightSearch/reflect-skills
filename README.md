# Reflect Skills

Skills for adopting [Reflect](https://docs.starlight-search.com) — long-term
memory for AI agents that gets smarter over time.

## Available skills

### `integrate-reflect`

Walks a coding agent through adding Reflect long-term memory to a Python
agent project end-to-end: SDK install, framework-specific loop placement,
parameter tuning, LLM-as-judge wiring, and a mandatory smoke test that
proves the loop closes.

Triggers on phrases like *"add Reflect to my agent"*, *"give my agent memory"*,
*"wire up `client.trace`"*, *"`ctx.memories` is empty"*, *"set up an LLM judge
for Reflect"*.

Covers 5 Python frameworks: OpenAI Agents SDK, Claude Agent SDK, LangGraph,
Pydantic AI, and a generic-loop fallback.

## Install

```bash
# Add a single skill globally
npx skills add StarlightSearch/reflect-skills@integrate-reflect -g -y
```

Browse all skills: <https://skills.sh/StarlightSearch/reflect-skills>

## What is Reflect

Reflect records what happened on each agent run (task, trajectory, outcome),
distills it into a one-sentence reflection, and re-injects the most useful
past lessons into future runs. Memories are ranked by a blend of semantic
similarity and a learned utility score (q-value). Memories that lead to good
outcomes float up; memories tied to failures sink until later success
rehabilitates them.

📚 Docs: <https://docs.starlight-search.com>
🖥️ Console: <https://reflect.starlight-search.com>
⚡ Hosted API: `https://api.starlight-search.com`

For agents fetching docs: <https://docs.starlight-search.com/llms-full.txt>
(plain text, fetch with `curl`/`urllib`).

## Contributing

Open an issue or PR. Skills here are built TDD-style — each skill should be
backed by baseline scenarios that fail without it and pass with it. See
`integrate-reflect/SKILL.md` for an example of the structure.

## License

MIT — see `LICENSE`.
