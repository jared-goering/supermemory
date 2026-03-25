#!/usr/bin/env python3
"""
Autoresearch: Supermemory Benchmark Optimizer

Iterates over search/answer strategies against pre-ingested benchmark data.
Logs every experiment to experiments_bench.jsonl for analysis.

Usage:
    python3 autoresearch_bench.py --max-experiments 100

Requires: eval server running on :8643 with ingested benchmark data.
"""

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ─── Config ───────────────────────────────────────────────────────────────────

EVAL_API = "http://127.0.0.1:8643"
EVAL_DB = "/tmp/memorybench_eval.db"
CHECKPOINT = "/Users/jared/Projects/memorybench/data/runs/eval-llm/checkpoint.json"
EXPERIMENTS_LOG = "/Users/jared/Projects/openclaw-memory/experiments_bench.jsonl"
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

# ─── Search Space ─────────────────────────────────────────────────────────────

SEARCH_SPACE = {
    "top_k": [5, 10, 15, 20, 30, 50],
    "include_source": [True, False],
    "rerank_strategy": [
        "none",
        "similarity_threshold",  # drop below threshold
        "entity_boost",  # boost results sharing entities with query
        "temporal_recency",  # boost recent memories
        "diversity",  # MMR-style diversification
        "category_weight",  # weight by category type
    ],
    "similarity_threshold": [
        0.0,
        0.3,
        0.4,
        0.5,
        0.6,
    ],  # only used with similarity_threshold strategy
    "temporal_weight": [0.0, 0.1, 0.2, 0.3],  # blend with recency
    "answer_strategy": [
        "default",  # current prompt
        "cot_structured",  # chain-of-thought with structure
        "entity_focused",  # group by entity first
        "temporal_narrative",  # present as timeline
        "concise_direct",  # minimal prompt, direct answer
    ],
    "answer_model": [
        "gemini-3-flash-preview",  # latest Gemini 3 Flash - smarter than 2.5
    ],
    "max_context_memories": [3, 5, 10, 15, 20],  # how many memories to include in answer prompt
}


# ─── Questions ────────────────────────────────────────────────────────────────


def load_questions():
    """Load questions from memorybench checkpoint."""
    with open(CHECKPOINT) as f:
        data = json.load(f)
    questions = []
    for qid, q in data["questions"].items():
        questions.append(
            {
                "id": qid,
                "question": q["question"],
                "ground_truth": q["groundTruth"],
                "question_type": q["questionType"],
            }
        )
    return questions


# ─── Judge Prompts ────────────────────────────────────────────────────────────

JUDGE_PROMPTS = {
    "default": """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.

Respond with ONLY a JSON object:
{"score": 1, "label": "correct", "explanation": "..."} if the response contains the correct answer
{"score": 0, "label": "incorrect", "explanation": "..."} if the response does not contain the correct answer""",
    "temporal-reasoning": """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. Do not penalize off-by-one errors for days/weeks/months.

Respond with ONLY a JSON object:
{"score": 1, "label": "correct", "explanation": "..."} if the response contains the correct answer
{"score": 0, "label": "incorrect", "explanation": "..."} if the response does not contain the correct answer""",
    "knowledge-update": """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.

Respond with ONLY a JSON object:
{"score": 1, "label": "correct", "explanation": "..."} if the response contains the correct answer
{"score": 0, "label": "incorrect", "explanation": "..."} if the response does not contain the correct answer""",
    "single-session-preference": """I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.

Respond with ONLY a JSON object:
{"score": 1, "label": "correct", "explanation": "..."} if the response satisfies the rubric
{"score": 0, "label": "incorrect", "explanation": "..."} if the response does not satisfy the rubric""",
}


def get_judge_prompt(question_type):
    for key in JUDGE_PROMPTS:
        if key in question_type:
            return JUDGE_PROMPTS[key]
    return JUDGE_PROMPTS["default"]


# ─── Answer Prompts ───────────────────────────────────────────────────────────


def build_answer_prompt(question, memories, strategy="default"):
    """Build answer prompt based on strategy."""
    if strategy == "default":
        context = format_memories_default(memories)
        return f"""You are a question-answering system. Based on the retrieved context below, answer the question.

Question: {question}

Retrieved Context:
{context}

Instructions:
- Think step by step, then provide a clear answer
- If the context doesn't contain enough information, say "I don't know"
- Base your answer ONLY on the provided context
- When information has been updated (UPDATE relation), use the latest version

Reasoning:
[Your step-by-step reasoning]

Answer:
[Your final answer]"""

    elif strategy == "cot_structured":
        context = format_memories_default(memories)
        return f"""Question: {question}

Memories retrieved:
{context}

Step 1 - Identify relevant memories: Which memories directly relate to the question?
Step 2 - Check for updates: Are any memories superseded by newer versions (UPDATE relations)?
Step 3 - Temporal ordering: What's the chronological order of relevant events?
Step 4 - Synthesize: Combine the relevant, most current information.
Step 5 - Answer: Provide your answer.

Work through each step, then give your final answer after "ANSWER:"."""

    elif strategy == "entity_focused":
        context = format_memories_by_entity(memories)
        return f"""Question: {question}

Memories organized by entity/topic:
{context}

Instructions: Answer based only on these memories. Group related facts by entity to reason about cross-session information. If you don't have enough info, say "I don't know".

Answer:"""

    elif strategy == "temporal_narrative":
        context = format_memories_temporal(memories)
        return f"""Question: {question}

Memory timeline (chronological):
{context}

Instructions: Use this timeline to answer the question. Pay attention to the most recent information, especially if earlier facts were updated. Answer concisely.

Answer:"""

    elif strategy == "concise_direct":
        context = format_memories_minimal(memories)
        return f"""Q: {question}

Context:
{context}

A:"""

    elif strategy == "entity_grouped":
        context = format_memories_by_entity(memories)
        return f"""Question: {question}

Memories organized by entity (each entity group contains all related memories across different sessions):
{context}

Instructions:
1. Each [Entity: ...] group contains ALL known memories about that entity/topic from across multiple conversations.
2. Count distinct entities/events carefully when the question asks "how many."
3. Cross-reference between groups to find connections.
4. Use dates to establish chronological order.
5. If memories have been updated (UPDATE relation), use the latest version.

Think through the entity groups step by step, then provide your answer after "ANSWER:"."""

    return build_answer_prompt(question, memories, "default")


def format_memories_default(memories):
    parts = []
    for i, m in enumerate(memories):
        lines = [f"Memory {i + 1} (sim: {m.get('similarity', 0):.3f}):"]
        lines.append(m["content"])
        if m.get("document_date"):
            lines.append(f"Date: {m['document_date']}")
        if m.get("relations"):
            for rel in m["relations"]:
                lines.append(f"  [{rel['relation']}] {rel['related_content']}")
        if m.get("source_chunk"):
            lines.append(f"Source: {m['source_chunk'][:500]}")
        parts.append("\n".join(lines))
    return "\n---\n".join(parts)


def format_memories_by_entity(memories):
    """Group memories by their entity tags (from entity-aware search)."""
    # Check if memories have entity tags from entity search
    has_entities = any(m.get("entities") for m in memories)

    if has_entities:
        # Group by actual entities
        entity_groups = {}
        ungrouped = []
        for m in memories:
            entities = m.get("entities", [])
            if entities:
                for e in entities:
                    entity_groups.setdefault(e, []).append(m)
            else:
                ungrouped.append(m)

        parts = []
        seen_ids = set()
        # Sort entity groups by number of memories (largest first)
        for entity_name, mems in sorted(entity_groups.items(), key=lambda x: -len(x[1])):
            lines = [f"[Entity: {entity_name} ({len(mems)} memories)]"]
            for m in mems:
                mid = m.get("id", id(m))
                if mid not in seen_ids:
                    seen_ids.add(mid)
                    date_str = f" ({m.get('document_date', '')})" if m.get("document_date") else ""
                    source_tag = (
                        " [entity-expanded]" if m.get("source") == "entity_expansion" else ""
                    )
                    lines.append(f"  - {m['content']}{date_str}{source_tag}")
                    if m.get("relations"):
                        for rel in m["relations"]:
                            lines.append(f"    [{rel['relation']}] {rel['related_content']}")
            parts.append("\n".join(lines))

        if ungrouped:
            lines = ["[Other memories]"]
            for m in ungrouped:
                mid = m.get("id", id(m))
                if mid not in seen_ids:
                    seen_ids.add(mid)
                    lines.append(f"  - {m['content']}")
            parts.append("\n".join(lines))

        return "\n\n".join(parts)
    else:
        # Fallback: group by session
        groups = {}
        for m in memories:
            session = m.get("source_session", "unknown")
            prefix = "-".join(session.split("-")[:-1]) if "-" in session else session
            groups.setdefault(prefix, []).append(m)

        parts = []
        for group_name, mems in groups.items():
            lines = [f"[Group: {group_name}]"]
            for m in mems:
                lines.append(f"  - {m['content']}")
                if m.get("relations"):
                    for rel in m["relations"]:
                        lines.append(f"    [{rel['relation']}] {rel['related_content']}")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)


def format_memories_temporal(memories):
    """Sort by document_date, present as timeline."""
    sorted_mems = sorted(
        memories,
        key=lambda m: m.get("document_date") or "9999",
    )
    parts = []
    for m in sorted_mems:
        date = m.get("document_date", "unknown date")
        parts.append(f"[{date}] {m['content']}")
        if m.get("relations"):
            for rel in m["relations"]:
                parts.append(f"  -> [{rel['relation']}] {rel['related_content']}")
    return "\n".join(parts)


def format_memories_minimal(memories):
    """Minimal format - just content."""
    return "\n".join(f"- {m['content']}" for m in memories)


# ─── Re-ranking ───────────────────────────────────────────────────────────────


def rerank(memories, strategy, config):
    """Apply re-ranking strategy to search results."""
    if strategy == "none":
        return memories

    elif strategy == "similarity_threshold":
        threshold = config.get("similarity_threshold", 0.4)
        return [m for m in memories if m.get("similarity", 0) >= threshold]

    elif strategy == "entity_boost":
        # Boost memories that share entities via relations
        for m in memories:
            entity_bonus = len(m.get("relations", [])) * 0.05
            m["_score"] = m.get("similarity", 0) + entity_bonus
        return sorted(memories, key=lambda m: m.get("_score", 0), reverse=True)

    elif strategy == "temporal_recency":
        # Boost recent memories
        weight = config.get("temporal_weight", 0.1)
        now = datetime.now(timezone.utc)
        for m in memories:
            try:
                doc_date = datetime.fromisoformat(
                    m.get("document_date", "2020-01-01").replace("Z", "+00:00")
                )
                days_ago = (now - doc_date).days
                recency = max(0, 1 - days_ago / 365)  # 0-1 scale, 1 = today
            except (ValueError, TypeError):
                recency = 0.5
            m["_score"] = m.get("similarity", 0) * (1 - weight) + recency * weight
        return sorted(memories, key=lambda m: m.get("_score", 0), reverse=True)

    elif strategy == "diversity":
        # MMR-style: pick top result, then iteratively pick most different
        if len(memories) <= 1:
            return memories
        selected = [memories[0]]
        remaining = memories[1:]
        while remaining and len(selected) < len(memories):
            # Pick the one most different from already selected (by session)
            selected_sessions = {m.get("source_session") for m in selected}
            # Prefer different sessions
            diff = [m for m in remaining if m.get("source_session") not in selected_sessions]
            if diff:
                selected.append(diff[0])
                remaining.remove(diff[0])
            else:
                selected.append(remaining.pop(0))
        return selected

    elif strategy == "category_weight":
        # Weight categories differently
        category_weights = {
            "preference": 1.3,
            "fact": 1.1,
            "event": 1.0,
            "opinion": 0.9,
            "insight": 1.0,
            "recommendation": 0.8,
            "goal": 1.0,
        }
        for m in memories:
            cat = m.get("category", "fact")
            m["_score"] = m.get("similarity", 0) * category_weights.get(cat, 1.0)
        return sorted(memories, key=lambda m: m.get("_score", 0), reverse=True)

    return memories


# ─── API Calls ────────────────────────────────────────────────────────────────


def search_memories(query, top_k=20, include_source=True, entity_search=False, entity_expand_k=50):
    """Search eval DB via API. If entity_search=True, uses entity-aware endpoint."""
    if entity_search:
        resp = requests.post(
            f"{EVAL_API}/api/search_entities",
            json={
                "query": query,
                "top_k": top_k,
                "entity_expand_k": entity_expand_k,
                "include_source": include_source,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", []), data
    else:
        resp = requests.post(
            f"{EVAL_API}/api/search",
            json={"query": query, "top_k": top_k, "include_source": include_source},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("results", []), {}


def call_llm(prompt, model="gemini-2.5-flash", max_tokens=1024):
    """Call LLM via Google Gemini API or OpenRouter (for Anthropic/OpenAI models)."""
    # Route non-Gemini models through OpenRouter
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
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    # Gemini path
    # Gemini 3 Flash uses thinking tokens; bump output budget so answers aren't cut off
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


def judge_answer(question, ground_truth, hypothesis, question_type):
    """Use GPT-4o or Claude as judge."""
    judge_system = get_judge_prompt(question_type)

    is_preference = "preference" in question_type.lower()
    gt_label = "Rubric" if is_preference else "Ground Truth Answer"

    prompt = f"""{judge_system}

Question: {question}
{gt_label}: {ground_truth}
System's Hypothesis: {hypothesis}"""

    result_text = call_llm(prompt, model="gemini-3-flash-preview", max_tokens=256)

    try:
        # Strip markdown code fences that Gemini 3 Flash likes to add
        clean = result_text.strip()
        if clean.startswith("```"):
            # Remove ```json ... ``` wrapper
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
        # Fallback: look for score field directly in text
        import re

        score_match = re.search(r'"score"\s*:\s*(\d)', result_text)
        if score_match:
            score = int(score_match.group(1))
            label_match = re.search(r'"label"\s*:\s*"(\w+)"', result_text)
            label = label_match.group(1) if label_match else "unknown"
            expl_match = re.search(r'"explanation"\s*:\s*"([^"]*)"', result_text)
            explanation = expl_match.group(1) if expl_match else ""
            return {
                "score": 1 if score == 1 else 0,
                "label": label,
                "explanation": explanation,
            }
        is_correct = '"correct"' in result_text.lower() and '"incorrect"' not in result_text.lower()
        return {
            "score": 1 if is_correct else 0,
            "label": "correct" if is_correct else "incorrect",
            "explanation": "Parse failed: " + result_text[:200],
        }


# ─── Experiment Runner ────────────────────────────────────────────────────────


def run_experiment(questions, config, experiment_id):
    """Run one experiment with given config against all questions."""
    results = []
    correct = 0
    total = len(questions)
    by_type = {}

    for q in questions:
        t0 = time.time()

        # 1. Search
        entity_search = config.get("entity_search", False)
        memories, search_meta = search_memories(
            q["question"],
            top_k=config["top_k"],
            include_source=config["include_source"],
            entity_search=entity_search,
            entity_expand_k=config.get("entity_expand_k", 50),
        )
        search_ms = (time.time() - t0) * 1000

        # 2. Rerank
        memories = rerank(memories, config["rerank_strategy"], config)

        # 3. Truncate to max_context_memories
        memories = memories[: config["max_context_memories"]]

        # 4. Build answer prompt
        prompt = build_answer_prompt(q["question"], memories, config["answer_strategy"])

        # 5. Get answer
        t1 = time.time()
        try:
            answer = call_llm(prompt, model=config["answer_model"], max_tokens=1024)
        except Exception as e:
            answer = f"Error: {e}"
        answer_ms = (time.time() - t1) * 1000

        # 6. Judge
        t2 = time.time()
        judgment = judge_answer(q["question"], q["ground_truth"], answer, q["question_type"])
        judge_ms = (time.time() - t2) * 1000

        is_correct = judgment["score"] == 1
        if is_correct:
            correct += 1

        qtype = q["question_type"]
        by_type.setdefault(qtype, {"correct": 0, "total": 0})
        by_type[qtype]["total"] += 1
        if is_correct:
            by_type[qtype]["correct"] += 1

        results.append(
            {
                "question_id": q["id"],
                "question_type": qtype,
                "correct": is_correct,
                "search_ms": round(search_ms, 1),
                "answer_ms": round(answer_ms, 1),
                "judge_ms": round(judge_ms, 1),
                "num_memories": len(memories),
                "explanation": judgment["explanation"][:200],
            }
        )

    accuracy = correct / total if total > 0 else 0
    type_accuracy = {k: v["correct"] / v["total"] for k, v in by_type.items()}

    return {
        "accuracy": round(accuracy * 100, 2),
        "correct": correct,
        "total": total,
        "by_type": type_accuracy,
        "results": results,
    }


def sample_config():
    """Sample a random config from the search space."""
    config = {}
    for key, values in SEARCH_SPACE.items():
        config[key] = random.choice(values)

    return config


def smart_sample_config(history):
    """Sample config biased toward top-performing parameter values."""
    if len(history) < 10:
        return sample_config()

    # Find top 20% experiments
    sorted_hist = sorted(history, key=lambda h: h["accuracy"], reverse=True)
    top_n = max(3, len(sorted_hist) // 5)
    top_configs = [h["config"] for h in sorted_hist[:top_n]]

    config = {}
    for key, values in SEARCH_SPACE.items():
        # 60% chance: pick from top configs, 40%: random exploration
        if random.random() < 0.6 and top_configs:
            config[key] = random.choice(top_configs).get(key, random.choice(values))
        else:
            config[key] = random.choice(values)

    return config


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Autoresearch: Supermemory Benchmark Optimizer")
    parser.add_argument("--max-experiments", type=int, default=100)
    parser.add_argument("--resume", action="store_true", help="Resume from existing log")
    args = parser.parse_args()

    print(f"Loading questions from {CHECKPOINT}...")
    questions = load_questions()
    print(f"  {len(questions)} questions loaded")

    # Load previous experiments
    history = []
    if args.resume and Path(EXPERIMENTS_LOG).exists():
        with open(EXPERIMENTS_LOG) as f:
            for line in f:
                if line.strip():
                    history.append(json.loads(line))
        print(f"  Resuming from {len(history)} previous experiments")

    best_accuracy = max((h["accuracy"] for h in history), default=0)
    print(f"  Best accuracy so far: {best_accuracy}%")
    print(f"  Running up to {args.max_experiments} experiments\n")

    start_id = len(history)

    for i in range(args.max_experiments):
        exp_id = start_id + i
        config = smart_sample_config(history)

        print(
            f"[Exp {exp_id}] top_k={config['top_k']}, rerank={config['rerank_strategy']}, "
            f"answer={config['answer_strategy']}, max_ctx={config['max_context_memories']}, "
            f"src={config['include_source']}, thresh={config['similarity_threshold']}"
        )

        t0 = time.time()
        try:
            result = run_experiment(questions, config, exp_id)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        elapsed = time.time() - t0

        is_best = result["accuracy"] > best_accuracy
        if is_best:
            best_accuracy = result["accuracy"]

        marker = " 🏆 NEW BEST!" if is_best else ""
        print(
            f"  Accuracy: {result['accuracy']}% ({result['correct']}/{result['total']}) "
            f"in {elapsed:.1f}s{marker}"
        )

        # Per-type breakdown
        for qtype, acc in sorted(result["by_type"].items()):
            print(f"    {qtype}: {acc * 100:.0f}%")

        # Log experiment
        entry = {
            "experiment_id": exp_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "accuracy": result["accuracy"],
            "correct": result["correct"],
            "total": result["total"],
            "by_type": result["by_type"],
            "elapsed_s": round(elapsed, 1),
            "is_best": is_best,
        }
        history.append(entry)

        with open(EXPERIMENTS_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")

        print()

    # Summary
    print("=" * 60)
    print(f"AUTORESEARCH COMPLETE: {len(history)} total experiments")
    print(f"Best accuracy: {best_accuracy}%")

    if not history:
        print("No experiments completed.")
        return

    # Find best config
    best = max(history, key=lambda h: h["accuracy"])
    print(f"\nBest config (Exp {best['experiment_id']}):")
    for k, v in best["config"].items():
        print(f"  {k}: {v}")
    print("\nBy type:")
    for qtype, acc in sorted(best["by_type"].items()):
        print(f"  {qtype}: {acc * 100:.0f}%")


if __name__ == "__main__":
    main()
