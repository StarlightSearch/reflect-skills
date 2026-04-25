"""Smoke test: prove the Reflect integration loop closes end-to-end.

Two calls:
  1. Submit a canned trace with result="pass". Wait for ingest.
  2. Submit a similar canned task and assert at least one memory was retrieved.

Run after every meaningful change to the integration:

    REFLECT_API_KEY=rf_live_... REFLECT_PROJECT_ID=my-project \\
        python templates/smoke_test.py

Exit code 0 = pass; 1 = fail with a structured reason.
"""
from __future__ import annotations

import os
import sys
import time
import uuid

from reflect_sdk import ReflectClient

# Cold-start parameters — see SKILL.md step 5.
COLD_START = dict(limit=5, lambda_=0.1, similarity_threshold=0.2)


def main() -> int:
    base_url = os.environ.get("REFLECT_API_URL", "https://api.starlight-search.com")
    api_key = os.environ["REFLECT_API_KEY"]
    project_id = os.environ["REFLECT_PROJECT_ID"]

    client = ReflectClient(base_url=base_url, api_key=api_key, project_id=project_id)

    # Use a per-run nonce so this test never collides across runs.
    nonce = uuid.uuid4().hex[:8]
    seed_task = f"smoke-test seed task {nonce}: configure exponential backoff for a flaky upstream"
    probe_task = f"smoke-test probe task {nonce}: add retry with backoff to an unreliable HTTP client"

    # Step 1 — seed a memory.
    print(f"[smoke] step 1: seeding a memory with task={seed_task!r}")
    with client.trace(seed_task, **COLD_START) as ctx:
        ctx.set_output(
            trajectory=[
                {"role": "user", "content": seed_task},
                {"role": "assistant", "content": "Use tenacity with exponential backoff and a max_attempts of 5."},
            ],
            final_response="Use tenacity with exponential backoff and a max_attempts of 5.",
            result="pass",
        )
    seed_trace_id = ctx.trace_id
    print(f"[smoke]   seeded trace_id={seed_trace_id}")

    # Step 2 — wait for ingest. Reflect generates the reflection + embeds asynchronously.
    print("[smoke] step 2: waiting for ingest (up to 30s)...")
    deadline = time.time() + 30
    while time.time() < deadline:
        trace = client.get_trace(seed_trace_id)
        if getattr(trace, "ingest_status", None) == "completed":
            break
        time.sleep(1)
    else:
        print("[smoke] FAIL: ingest did not complete within 30s")
        print("        possible causes: worker not running, embedding model down, Q_LEARNING_ALPHA misconfigured")
        return 1
    print("[smoke]   ingest complete")

    # Step 3 — probe with a similar task and assert retrieval works.
    print(f"[smoke] step 3: probing retrieval with task={probe_task!r}")
    with client.trace(probe_task, **COLD_START) as ctx:
        n = len(ctx.memories)
        ctx.set_output(
            trajectory=[{"role": "user", "content": probe_task}, {"role": "assistant", "content": "(probe)"}],
            final_response="(probe)",
            result=None,  # don't pollute q-values from a smoke test
        )

    if n < 1:
        print(f"[smoke] FAIL: probe retrieved 0 memories (expected ≥1)")
        print("        check: cold-start params (lambda_=0.1, similarity_threshold=0.2),")
        print("               project_id scoping (seed and probe use the same project_id?),")
        print("               worker actually wrote the memory (see Reflect dashboard)")
        return 1

    print(f"[smoke] PASS: probe retrieved {n} memory(ies); integration loop closes ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
