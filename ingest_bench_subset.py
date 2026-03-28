#!/usr/bin/env python3
"""Ingest a subset of multi-session benchmark questions into the eval DB."""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

EVAL_API = "http://127.0.0.1:8643"
DATASET = os.path.expanduser(
    "~/Projects/memorybench/data/benchmarks/longmemeval/datasets/longmemeval_s_cleaned.json"
)
QUESTIONS_DIR = os.path.expanduser(
    "~/Projects/memorybench/data/benchmarks/longmemeval/datasets/questions"
)

# 17 multi-session questions to ingest (smallest session counts first)
TARGET_IDS = [
    # Already done: "28dc39ac", "6cb6f249"
    "2318644b",
    "ba358f49",
    "gpt4_31ff4165",
    "5a7937c8",
    "cc06de0d",
    "92a0aa75",
    "1192316e",
    "a3332713",
    "a1cc6108",
    "8979f9ec",
    "6c49646a",
    "0ea62687",
    "0a995998",
    "gpt4_7fce9456",
    "9d25d4e0",
]


def load_question_data(qid):
    """Load full question data including haystack sessions."""
    path = os.path.join(QUESTIONS_DIR, f"{qid}.json")
    if not os.path.exists(path):
        print(f"  ⚠ No question file for {qid}")
        return None
    with open(path) as f:
        return json.load(f)


def ingest_session(messages, container_tag, session_id, date_str=None):
    """Ingest a single session via the API."""
    if not messages:
        return 0

    # Format session text
    lines = []
    if date_str:
        lines.append(f"[Session date: {date_str}]")

    for msg in messages:
        speaker = msg.get("speaker") or msg.get("role", "unknown")
        ts = f" [{msg['timestamp']}]" if msg.get("timestamp") else ""
        lines.append(f"{speaker}{ts}: {msg.get('content', '')}")

    text = "\n".join(lines)
    if not text.strip():
        return 0

    body = {
        "text": text,
        "session_key": f"bench_{container_tag}_{session_id}",
        "agent_id": f"bench_{container_tag}",
        "chunk_size": 800,
        "chunk_overlap": 100,
    }
    if date_str:
        body["document_date"] = date_str

    for attempt in range(3):
        try:
            resp = requests.post(
                f"{EVAL_API}/api/ingest",
                json=body,
                timeout=120,
            )
            if resp.ok:
                data = resp.json()
                return data.get("count", 0)
            else:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
        except Exception as e:
            if attempt < 2:
                print(f"    Retry {attempt + 1}: {e}")
                time.sleep(2 * (attempt + 1))
    return -1


def main():
    # Health check
    try:
        r = requests.get(f"{EVAL_API}/api/health", timeout=5)
        health = r.json()
        print(f"Eval server: {health['memories']} memories, v{health['version']}")
    except Exception as e:
        print(f"Cannot reach eval server at {EVAL_API}: {e}")
        sys.exit(1)

    total_memories = 0
    total_sessions = 0
    failed_questions = []

    for qi, qid in enumerate(TARGET_IDS):
        qdata = load_question_data(qid)
        if not qdata:
            failed_questions.append(qid)
            continue

        sessions = qdata.get("haystack_sessions", [])
        session_ids = qdata.get("haystack_session_ids", [])
        dates = qdata.get("haystack_dates", [])
        container_tag = f"{qid}-eval-llm"

        print(
            f"\n[{qi + 1}/{len(TARGET_IDS)}] {qid}: {len(sessions)} sessions — {qdata.get('question', '')[:60]}",
            flush=True,
        )

        q_memories = 0
        q_done = 0

        def _do_session(si):
            sess_messages = sessions[si]
            sid = session_ids[si] if si < len(session_ids) else f"session-{si}"
            d = dates[si] if si < len(dates) else None
            return ingest_session(sess_messages, container_tag, sid, d)

        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {pool.submit(_do_session, si): si for si in range(len(sessions))}
            for fut in as_completed(futures):
                count = fut.result()
                if count >= 0:
                    q_memories += count
                    total_sessions += 1
                q_done += 1
                if q_done % 10 == 0 or q_done == len(sessions):
                    print(
                        f"  {q_done}/{len(sessions)} sessions ({q_memories} memories)", flush=True
                    )

        total_memories += q_memories
        print(f"  ✓ {qid}: {q_memories} memories from {len(sessions)} sessions", flush=True)

    # Refresh cache
    print("\nRefreshing embedding cache...")
    try:
        requests.post(f"{EVAL_API}/api/cache/refresh", timeout=120)
    except Exception:
        pass

    print(f"\n{'=' * 60}")
    print(f"Done! Ingested {total_sessions} sessions → {total_memories} memories")
    print(f"Failed questions: {failed_questions if failed_questions else 'none'}")

    # Final health
    r = requests.get(f"{EVAL_API}/api/health", timeout=5)
    print(f"Eval DB now has {r.json()['memories']} total memories")


if __name__ == "__main__":
    main()
