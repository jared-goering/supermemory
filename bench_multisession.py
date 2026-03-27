#!/usr/bin/env python3
"""
Multi-Session Aggregate Benchmark

Tests Ultramemory's ability to answer multi-session aggregate questions
(e.g., "How many weddings have I attended?") that require scanning and
combining facts across multiple conversation sessions.

Targets: 3 ingested multi-session questions from LongMemEval_s benchmark.
Also loads the full 133 multi-session questions for reference stats.

Usage:
    # Start the eval server first:
    #   ULTRAMEMORY_DB_PATH=/tmp/memorybench_eval.db ultramemory serve --port 8643
    python3 bench_multisession.py
    python3 bench_multisession.py --strategy event_aggregate --top-k 50
    python3 bench_multisession.py --sweep  # test all strategy combinations
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone

import requests

# ─── Config ───────────────────────────────────────────────────────────────────

EVAL_API = "http://127.0.0.1:8643"
CHECKPOINT = "/Users/jared/Projects/memorybench/data/runs/eval-llm/checkpoint.json"
QUESTIONS_DIR = os.path.expanduser(
    "~/Projects/memorybench/data/benchmarks/longmemeval/datasets/questions/"
)
EXPERIMENTS_LOG = "/Users/jared/Projects/openclaw-memory/experiments_multisession.jsonl"

AUTH_PROFILES_PATH = os.path.expanduser("~/.openclaw/agents/main/agent/auth-profiles.json")
OPENROUTER_KEY_PATH = os.path.expanduser("~/.openclaw/secrets/openrouter-api-key.txt")


def get_google_key():
    with open(AUTH_PROFILES_PATH) as f:
        data = json.load(f)
    profiles = data.get("profiles", data)
    if isinstance(profiles, dict):
        for _key, val in profiles.items():
            if isinstance(val, dict) and val.get("provider") == "google":
                return val.get("token") or val.get("apiKey")
    raise RuntimeError("No Google key found in auth-profiles.json")


def get_openrouter_key():
    with open(OPENROUTER_KEY_PATH) as f:
        return f.read().strip()


GOOGLE_KEY = get_google_key()
OPENROUTER_KEY = get_openrouter_key()


# ─── Questions ────────────────────────────────────────────────────────────────


def load_checkpoint_questions():
    """Load the 3 ingested multi-session questions from checkpoint."""
    with open(CHECKPOINT) as f:
        data = json.load(f)
    questions = []
    for qid, q in data["questions"].items():
        if q.get("questionType") == "multi-session":
            questions.append(
                {
                    "id": qid,
                    "question": q["question"],
                    "ground_truth": q["groundTruth"],
                    "question_type": q["questionType"],
                    "container_tag": q.get("containerTag", qid),
                }
            )
    return questions


def load_all_multisession_questions():
    """Load all 133 multi-session questions from JSON files (for reference)."""
    questions = []
    for f in sorted(os.listdir(QUESTIONS_DIR)):
        if not f.endswith(".json"):
            continue
        with open(os.path.join(QUESTIONS_DIR, f)) as fh:
            q = json.load(fh)
        if q.get("question_type") == "multi-session":
            questions.append(q)
    return questions


# ─── LLM Calls ───────────────────────────────────────────────────────────────


def call_llm(prompt, model="gemini-3-flash-preview", max_tokens=2048):
    """Call LLM via Google Gemini API or OpenRouter."""
    if model.startswith(("anthropic/", "openai/")):
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    effective_max = max_tokens * 3 if "gemini-3" in model else max_tokens
    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GOOGLE_KEY}",
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": effective_max},
        },
        timeout=90,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


# ─── Search Strategies ────────────────────────────────────────────────────────


def search_aggregate(question, session_prefix, top_k=50):
    """Use the aggregate_search endpoint for broad multi-session retrieval."""
    resp = requests.post(
        f"{EVAL_API}/api/aggregate_search",
        json={
            "question": question,
            "session_prefix": session_prefix,
            "top_k": top_k,
            "include_source": True,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["memories"], data.get("event_clusters", []), data


def search_standard(question, top_k=20):
    """Standard vector search."""
    resp = requests.post(
        f"{EVAL_API}/api/search",
        json={"query": question, "top_k": top_k, "include_source": True},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("results", []), [], {}


def search_entity(question, top_k=30, entity_expand_k=50):
    """Entity-aware search with cross-session expansion."""
    resp = requests.post(
        f"{EVAL_API}/api/search_entities",
        json={
            "query": question,
            "top_k": top_k,
            "entity_expand_k": entity_expand_k,
            "include_source": True,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("results", []), [], data


# ─── Answer Strategies ────────────────────────────────────────────────────────


def build_aggregate_prompt(question, memories, event_clusters):
    """Build prompt for multi-session aggregate questions with event cluster context."""
    # Group memories by session
    session_groups = {}
    for m in memories:
        session = m.get("source_session", "unknown")
        session_groups.setdefault(session, []).append(m)

    # Format memories by session (skip source chunks to reduce noise)
    mem_parts = []
    for session, mems in sorted(session_groups.items()):
        lines = [f"[Session: {session}]"]
        seen_content = set()
        for m in mems:
            content = m["content"]
            if content in seen_content:
                continue
            seen_content.add(content)
            date = f" ({m.get('document_date', '')})" if m.get("document_date") else ""
            lines.append(f"  - {content}{date}")
        mem_parts.append("\n".join(lines))

    memory_context = "\n\n".join(mem_parts)

    # Deduplicate event clusters by distinct_key
    seen_keys = set()
    deduped_clusters = []
    for c in event_clusters:
        key = c.get("distinct_key", c.get("canonical_label", ""))
        if key not in seen_keys:
            seen_keys.add(key)
            deduped_clusters.append(c)

    # Format event clusters
    cluster_context = ""
    if deduped_clusters:
        cluster_lines = ["Structured Event Data (pre-extracted from conversations):"]
        for c in deduped_clusters:
            involvement = c.get("user_involvement", "unknown")
            date = c.get("normalized_date", "unknown date")
            participants = c.get("participants", "[]")
            duration = c.get("duration_minutes")
            dur_str = f", duration: {duration}min" if duration else ""
            cluster_lines.append(
                f"  - [{c['event_type']}/{c.get('subtype', '')}] {c['canonical_label']} "
                f"(date: {date}, involvement: {involvement}, participants: {participants}{dur_str})"
            )
        cluster_context = "\n".join(cluster_lines)

    return f"""Question: {question}

You are answering a question that requires gathering and combining information from MULTIPLE conversation sessions. The user had many separate conversations and you need to piece together information scattered across them.

{cluster_context + chr(10) + chr(10) if cluster_context else ""}Memories from conversations (organized by session):
{memory_context}

CRITICAL INSTRUCTIONS FOR COUNTING:
1. The question asks about DISTINCT real-world events/items. Two memories about the same event = ONE event.
2. AGGRESSIVELY DEDUPLICATE: If two entries mention the same person, place, or occasion (even with slightly different wording or from different sessions), they are the SAME event. Count it ONCE.
3. Only count events where the user's involvement matches what the question asks (e.g., "attended" means the user was present, not just discussed or planned).
4. Ignore events the user only discussed, planned, or observed — unless the question specifically asks about those.
5. If the question asks "how much" or duration: find the relevant numbers and compute.
6. Base your answer ONLY on the provided information. When in doubt, err on the side of FEWER items (merge ambiguous entries).

REASONING:
[List each DISTINCT item you're counting. For each, note the key identifying details (who, what, when) and which session(s) mention it. Explicitly note any entries you're MERGING as duplicates.]

ANSWER:
[Your final answer — be specific and concise]"""


def build_standard_prompt(question, memories):
    """Standard search prompt (baseline)."""
    parts = []
    for i, m in enumerate(memories):
        lines = [f"Memory {i + 1} (sim: {m.get('similarity', 0):.3f}):"]
        lines.append(m["content"])
        if m.get("document_date"):
            lines.append(f"Date: {m['document_date']}")
        if m.get("source_chunk"):
            lines.append(f"Source: {m['source_chunk'][:500]}")
        parts.append("\n".join(lines))
    context = "\n---\n".join(parts)

    return f"""You are a question-answering system. Based on the retrieved context below, answer the question.

Question: {question}

Retrieved Context:
{context}

Instructions:
- Think step by step, then provide a clear answer
- If the context doesn't contain enough information, say "I don't know"
- Base your answer ONLY on the provided context
- If this is a counting or aggregation question, list each item before giving the count

Answer:"""


# ─── Judge ────────────────────────────────────────────────────────────────────


JUDGE_PROMPT = """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.

Respond with ONLY a JSON object:
{"score": 1, "label": "correct", "explanation": "..."} if the response contains the correct answer
{"score": 0, "label": "incorrect", "explanation": "..."} if the response does not contain the correct answer"""


def judge_answer(question, ground_truth, hypothesis):
    """Judge answer correctness."""
    prompt = f"""{JUDGE_PROMPT}

Question: {question}
Ground Truth Answer: {ground_truth}
System's Hypothesis: {hypothesis}"""

    result_text = call_llm(prompt, model="gemini-3-flash-preview", max_tokens=256)

    try:
        clean = result_text.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            clean = "\n".join(lines)
        json_match = json.loads(clean[clean.index("{") : clean.rindex("}") + 1])
        return {
            "score": 1 if json_match.get("score") == 1 else 0,
            "label": json_match.get("label", "unknown"),
            "explanation": json_match.get("explanation", ""),
        }
    except (json.JSONDecodeError, ValueError):
        import re

        score_match = re.search(r'"score"\s*:\s*(\d)', result_text)
        if score_match:
            score = int(score_match.group(1))
            return {"score": 1 if score == 1 else 0, "label": "unknown", "explanation": ""}
        return {"score": 0, "label": "unknown", "explanation": "Parse failed: " + result_text[:200]}


# ─── Experiment Runner ────────────────────────────────────────────────────────


STRATEGIES = {
    "aggregate": {
        "description": "Aggregate search (vector + events + keywords) with aggregate prompt",
        "search": "aggregate",
        "top_k": 50,
    },
    "aggregate_large": {
        "description": "Aggregate search with higher top_k for broader coverage",
        "search": "aggregate",
        "top_k": 80,
    },
    "entity": {
        "description": "Entity-aware search with aggregate prompt",
        "search": "entity",
        "top_k": 30,
        "entity_expand_k": 80,
    },
    "standard": {
        "description": "Standard vector search (baseline)",
        "search": "standard",
        "top_k": 20,
    },
    "standard_large": {
        "description": "Standard vector search with high top_k",
        "search": "standard",
        "top_k": 50,
    },
}


def run_question(question, strategy_name, strategy_config, answer_model="gemini-3-flash-preview"):
    """Run a single question with a given strategy."""
    t0 = time.time()

    qid = question["id"]
    # Extract session prefix from container_tag or question id
    # Session keys are like bench_gpt4_2f8be40d-eval-llm_gpt4_2f8be40d-session-0
    container = question.get("container_tag", qid)
    session_prefix = container.split("-eval-llm")[0] if "-eval-llm" in container else qid

    search_type = strategy_config["search"]
    top_k = strategy_config.get("top_k", 50)

    # Search
    if search_type == "aggregate":
        memories, event_clusters, meta = search_aggregate(
            question["question"], session_prefix, top_k
        )
    elif search_type == "entity":
        memories, event_clusters, meta = search_entity(
            question["question"],
            top_k=top_k,
            entity_expand_k=strategy_config.get("entity_expand_k", 50),
        )
    else:
        memories, event_clusters, meta = search_standard(question["question"], top_k)

    search_ms = (time.time() - t0) * 1000

    # Build prompt
    if search_type == "aggregate":
        prompt = build_aggregate_prompt(question["question"], memories, event_clusters)
    else:
        prompt = build_standard_prompt(question["question"], memories)

    # Get answer
    t1 = time.time()
    try:
        answer = call_llm(prompt, model=answer_model, max_tokens=2048)
    except Exception as e:
        answer = f"Error: {e}"
    answer_ms = (time.time() - t1) * 1000

    # Judge
    t2 = time.time()
    judgment = judge_answer(question["question"], question["ground_truth"], answer)
    judge_ms = (time.time() - t2) * 1000

    return {
        "question_id": qid,
        "question": question["question"],
        "ground_truth": question["ground_truth"],
        "answer": answer,
        "correct": judgment["score"] == 1,
        "judgment": judgment,
        "strategy": strategy_name,
        "search_type": search_type,
        "memory_count": len(memories),
        "cluster_count": len(event_clusters),
        "search_ms": round(search_ms, 1),
        "answer_ms": round(answer_ms, 1),
        "judge_ms": round(judge_ms, 1),
    }


def run_benchmark(questions, strategy_name, strategy_config, answer_model="gemini-3-flash-preview"):
    """Run all questions with a given strategy."""
    results = []
    correct = 0

    for q in questions:
        result = run_question(q, strategy_name, strategy_config, answer_model)
        results.append(result)
        if result["correct"]:
            correct += 1

        status = "✓" if result["correct"] else "✗"
        print(f"  {status} [{result['question_id']}] {result['question'][:60]}...")
        print(f"    GT: {str(result['ground_truth'])[:80]}")
        print(f"    Got: {result['answer'][:200]}")
        if not result["correct"]:
            print(f"    Judge: {result['judgment'].get('explanation', '')[:150]}")
        print(
            f"    Memories: {result['memory_count']}, Clusters: {result['cluster_count']}, "
            f"Search: {result['search_ms']:.0f}ms, Answer: {result['answer_ms']:.0f}ms"
        )
        print()

    accuracy = correct / len(questions) if questions else 0
    return {
        "strategy": strategy_name,
        "accuracy": round(accuracy * 100, 2),
        "correct": correct,
        "total": len(questions),
        "results": results,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Multi-Session Aggregate Benchmark")
    parser.add_argument(
        "--strategy",
        default="aggregate",
        choices=list(STRATEGIES.keys()),
        help="Search/answer strategy to use",
    )
    parser.add_argument("--sweep", action="store_true", help="Test all strategies")
    parser.add_argument("--top-k", type=int, default=None, help="Override top_k for search")
    parser.add_argument("--model", default="gemini-3-flash-preview", help="Answer model")
    args = parser.parse_args()

    print("Loading multi-session questions from checkpoint...")
    questions = load_checkpoint_questions()
    print(f"  {len(questions)} ingested multi-session questions found")

    all_ms = load_all_multisession_questions()
    print(
        f"  ({len(all_ms)} total multi-session questions in dataset, {len(all_ms) - len(questions)} not yet ingested)"
    )
    print()

    if not questions:
        print("ERROR: No multi-session questions found in checkpoint!")
        return

    strategies_to_test = list(STRATEGIES.keys()) if args.sweep else [args.strategy]
    all_results = {}

    for strategy_name in strategies_to_test:
        config = STRATEGIES[strategy_name].copy()
        if args.top_k is not None:
            config["top_k"] = args.top_k

        print(f"{'=' * 60}")
        print(f"Strategy: {strategy_name} — {config.get('description', '')}")
        print(f"  search={config['search']}, top_k={config.get('top_k')}")
        print(f"{'=' * 60}")
        print()

        result = run_benchmark(questions, strategy_name, config, args.model)
        all_results[strategy_name] = result

        print(f"Result: {result['correct']}/{result['total']} = {result['accuracy']}%")
        print()

        # Log experiment
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": strategy_name,
            "config": config,
            "model": args.model,
            "accuracy": result["accuracy"],
            "correct": result["correct"],
            "total": result["total"],
            "per_question": [
                {
                    "id": r["question_id"],
                    "correct": r["correct"],
                    "memory_count": r["memory_count"],
                    "cluster_count": r["cluster_count"],
                }
                for r in result["results"]
            ],
        }
        with open(EXPERIMENTS_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # Summary
    if len(all_results) > 1:
        print()
        print(f"{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        for name, result in sorted(all_results.items(), key=lambda x: -x[1]["accuracy"]):
            print(
                f"  {name:20s}: {result['accuracy']:5.1f}% ({result['correct']}/{result['total']})"
            )

        best = max(all_results.items(), key=lambda x: x[1]["accuracy"])
        print(f"\nBest: {best[0]} at {best[1]['accuracy']}%")


if __name__ == "__main__":
    main()
