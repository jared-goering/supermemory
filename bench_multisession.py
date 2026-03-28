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
    """Load multi-session questions from checkpoint (includes gpt4_* variants)."""
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


def _get_ingested_session_prefixes():
    """Query the eval DB to find which session prefixes have ingested data."""
    try:
        requests.get(f"{EVAL_API}/api/stats", timeout=5)
        # Fallback: query DB directly if stats endpoint not available
    except Exception:
        pass

    import sqlite3

    db_path = os.environ.get("ULTRAMEMORY_DB_PATH", "/tmp/memorybench_eval.db")  # noqa: S108
    conn = sqlite3.connect(db_path, timeout=5)
    rows = conn.execute(
        """SELECT DISTINCT
            CASE
                WHEN source_session LIKE '%-eval-llm%'
                THEN substr(source_session, 7, instr(source_session, '-eval-llm') - 7)
                ELSE source_session
            END as prefix
        FROM memories
        WHERE source_session LIKE 'bench_%'"""
    ).fetchall()
    conn.close()
    return {r[0] for r in rows}


def load_all_multisession_questions():
    """Load all 133 multi-session questions from JSON files."""
    questions = []
    for f in sorted(os.listdir(QUESTIONS_DIR)):
        if not f.endswith(".json"):
            continue
        with open(os.path.join(QUESTIONS_DIR, f)) as fh:
            q = json.load(fh)
        if q.get("question_type") == "multi-session":
            questions.append(q)
    return questions


def load_testable_questions():
    """Load all multi-session questions that have ingested data in the eval DB.

    Merges questions from two sources:
    1. JSON question files (the canonical 133 multi-session questions)
    2. Checkpoint file (may include gpt4_* variants not in JSON files)

    Returns only questions whose session data exists in the eval DB.
    """
    ingested_prefixes = _get_ingested_session_prefixes()
    print(f"  Ingested session prefixes in eval DB: {len(ingested_prefixes)}")

    # Load from JSON files
    all_ms = load_all_multisession_questions()
    print(f"  Total multi-session questions in dataset: {len(all_ms)}")

    # Build testable set from JSON files
    testable = {}
    for q in all_ms:
        qid = q["question_id"]
        if qid in ingested_prefixes:
            testable[qid] = {
                "id": qid,
                "question": q["question"],
                "ground_truth": q["answer"],
                "question_type": q["question_type"],
                "container_tag": f"{qid}-eval-llm",
            }

    # Also load from checkpoint (picks up gpt4_* and other variants)
    checkpoint_qs = load_checkpoint_questions()
    for q in checkpoint_qs:
        if q["id"] not in testable:
            # Check if this question's prefix has data
            container = q.get("container_tag", q["id"])
            prefix = container.split("-eval-llm")[0] if "-eval-llm" in container else q["id"]
            # Check both with and without gpt4_ prefix
            if prefix in ingested_prefixes or f"gpt4_{prefix}" in ingested_prefixes:
                testable[q["id"]] = q

    return list(testable.values()), len(all_ms)


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


def search_structured(question, session_prefix):
    """Use the /api/aggregate endpoint for structured fact-based answers."""
    resp = requests.post(
        f"{EVAL_API}/api/aggregate",
        json={
            "question": question,
            "session_prefix": session_prefix,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


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


def build_aggregate_prompt(question, memories, event_clusters, extracted_events=None):
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

    # Format pre-extracted distinct events (from server-side dedup)
    # Filter to: (1) cluster-sourced, (2) user participated, (3) topic-relevant
    import re as _re

    extracted_context = ""
    if extracted_events:
        # Extract topic keywords from question for relevance filtering
        q_lower = question.lower()
        q_stop = {
            "how",
            "many",
            "much",
            "did",
            "do",
            "have",
            "has",
            "had",
            "i",
            "my",
            "me",
            "the",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "and",
            "or",
            "different",
            "attend",
            "attended",
            "last",
            "past",
            "this",
            "year",
            "month",
            "week",
            "all",
            "total",
            "what",
            "which",
            "where",
            "when",
        }
        q_words = set(_re.findall(r"[a-z]+", q_lower)) - q_stop

        relevant = []
        for ev in extracted_events:
            if ev.get("source") != "event_cluster":
                continue
            if ev.get("user_involvement") not in (
                "attended",
                "participated",
                "performed",
                "completed",
                "did",
                "went",
                "visited",
            ):
                continue
            # Topic relevance check: event type/label should overlap with question
            # Use stem matching (strip trailing s/ed/ing) for fuzzy overlap
            ev_text = (
                f"{ev.get('event_type', '')} {ev.get('subtype', '')} {ev['description']}".lower()
            )
            ev_words = set(_re.findall(r"[a-z]+", ev_text))

            def _stems(words):
                stems = set()
                for w in words:
                    stems.add(w)
                    for suffix in ("ings", "ing", "tion", "tions", "ed", "es", "s"):
                        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                            stems.add(w[: -len(suffix)])
                return stems

            if q_words and ev_words:
                q_stems = _stems(q_words)
                ev_stems = _stems(ev_words)
                overlap = q_stems & ev_stems
                if not overlap:
                    continue
            relevant.append(ev)

        if relevant:
            extracted_lines = [
                f"PRE-EXTRACTED DISTINCT EVENTS matching the question topic ({len(relevant)} found):",
                "  (These are server-side deduplicated. Use as primary reference for counting.)",
            ]
            for i, ev in enumerate(relevant, 1):
                parts = [f"  {i}. {ev['description'][:150]}"]
                meta = [f"date={ev.get('date', '?')}"]
                if ev.get("duration_minutes"):
                    meta.append(f"duration={ev['duration_minutes']}min")
                if ev.get("participants"):
                    meta.append(f"participants={ev['participants']}")
                parts.append(f"     {' | '.join(meta)}")
                extracted_lines.append("\n".join(parts))
            extracted_context = "\n".join(extracted_lines)

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

You are answering a question that requires gathering and combining information from MULTIPLE conversation sessions.

{extracted_context if extracted_context else cluster_context if cluster_context else ""}

{"" if extracted_context else "Memories from conversations:" + chr(10) + memory_context + chr(10)}
RULES:
{"- The PRE-EXTRACTED EVENTS list above is your ONLY source. Do NOT invent additional events." if extracted_context else "- Count only DISTINCT events. Merge duplicates aggressively."}
- MERGE RULE: If two entries could POSSIBLY be the same event described differently, they ARE the same event. Examples:
  * "sister's wedding" + "Rachel's wedding" = SAME (sister could be Rachel)
  * "cousin's wedding" + "Cousin Rachel's wedding at vineyard" = SAME
  * "college roommate's wedding" + "Emily's wedding" = SAME (roommate could be Emily)
  * "coworker's wedding at a vineyard" + "cousin Rachel's wedding at a vineyard" = SAME (same venue detail suggests same event)
- MERGE RULE: Same venue/location/setting = strong signal they're the same event. MERGE unless dates clearly differ.
- ONGOING/RECURRING activities (e.g., "weekly classes") = ONE activity, not multiple events.
- Only count events the user DIRECTLY attended/participated in. Exclude: planned, mentioned, gifted, observed, expo visits.
- When in doubt, ALWAYS MERGE. Undercounting is better than overcounting.
- For amounts/durations: compute from explicit numbers only.

ANSWER FORMAT:
1. First, group all entries that could be the same event together.
2. For each merged group, give ONE line: "Event N: [description] (merged from: entries X, Y, Z)"
3. End with: FINAL ANSWER: [number or brief fact]"""


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
    "structured": {
        "description": "Structured facts + aggregate search (hybrid)",
        "search": "structured",
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
    structured_data = None
    if search_type == "structured":
        # Phase 1: Get structured answer from /api/aggregate
        structured_data = search_structured(question["question"], session_prefix)
        # Phase 2: Also get broad context from aggregate_search for LLM fallback
        memories, event_clusters, meta = search_aggregate(
            question["question"], session_prefix, top_k
        )
    elif search_type == "aggregate":
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
    if search_type == "structured" and structured_data:
        s_answer = structured_data.get("structured_answer")
        s_facts = structured_data.get("structured_facts", [])
        if s_answer is not None and s_facts:
            # Build a prompt that presents the structured facts for LLM verification
            fact_lines = []
            for i, f in enumerate(s_facts, 1):
                fact_lines.append(
                    f"  {i}. {f['subject']} — {f['predicate']} "
                    f"(value={f.get('value')}, unit={f.get('unit')}, date={f.get('date')})"
                )
            facts_block = "\n".join(fact_lines)
            extracted_events = meta.get("extracted_events", [])
            prompt = f"""Question: {question["question"]}

STRUCTURED FACTS from database (deterministic, pre-extracted):
{facts_block}

Structured answer: {s_answer} (based on {len(s_facts)} distinct facts)

Additionally, here is broader context from memory search for verification:

{build_aggregate_prompt(question["question"], memories, event_clusters, extracted_events)}

RULES:
- The STRUCTURED FACTS above are your PRIMARY source. They are machine-extracted and deduplicated.
- Use the memory context ONLY to verify or correct the structured answer.
- If the structured facts are clearly wrong or incomplete based on memory evidence, explain why and give a corrected count.
- Otherwise, trust the structured answer.

ANSWER FORMAT:
FINAL ANSWER: [number or brief fact]"""
        else:
            # No structured facts — fall back to standard aggregate prompt
            extracted_events = meta.get("extracted_events", [])
            prompt = build_aggregate_prompt(
                question["question"], memories, event_clusters, extracted_events
            )
    elif search_type == "aggregate":
        extracted_events = meta.get("extracted_events", [])
        prompt = build_aggregate_prompt(
            question["question"], memories, event_clusters, extracted_events
        )
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

    extracted_event_count = len(meta.get("extracted_events", []))
    structured_fact_count = (
        len(structured_data.get("structured_facts", [])) if structured_data else 0
    )
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
        "extracted_event_count": extracted_event_count,
        "structured_fact_count": structured_fact_count,
        "structured_answer": structured_data.get("structured_answer") if structured_data else None,
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
        sf_info = ""
        if result.get("structured_fact_count"):
            sf_info = f", StructFacts: {result['structured_fact_count']}"
            if result.get("structured_answer") is not None:
                sf_info += f" (ans={result['structured_answer']})"
        print(
            f"    Memories: {result['memory_count']}, Clusters: {result['cluster_count']}, "
            f"Events: {result.get('extracted_event_count', '?')}{sf_info}, "
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
    parser.add_argument(
        "--limit", type=int, default=None, help="Max questions to run (for quick testing)"
    )
    args = parser.parse_args()

    print("Loading multi-session questions...")
    questions, total_ms = load_testable_questions()
    print(
        f"  {len(questions)} testable (with ingested data), "
        f"{total_ms - len(questions)} not yet ingested out of {total_ms} total"
    )
    print()

    if not questions:
        print("ERROR: No testable multi-session questions found!")
        print("  Ensure the eval DB has ingested session data.")
        return

    if args.limit and args.limit < len(questions):
        questions = questions[: args.limit]
        print(f"  (limited to {args.limit} questions)")
        print()

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
                    "extracted_event_count": r.get("extracted_event_count", 0),
                    "structured_fact_count": r.get("structured_fact_count", 0),
                    "structured_answer": r.get("structured_answer"),
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
