#!/usr/bin/env python3
"""
Opus-Driven Autoresearch: Supermemory Benchmark Optimizer

Uses Claude Opus as a research agent to design experiments, analyze results,
and iteratively find optimal search/answer configurations.

Instead of random sampling: Opus sees all results so far, identifies patterns,
and proposes targeted experiments to test specific hypotheses.

Usage:
    python3 autoresearch_opus.py --max-rounds 10 --exps-per-round 3
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ─── Config ───────────────────────────────────────────────────────────────────

EVAL_API = "http://127.0.0.1:8643"
CHECKPOINT = "/Users/jared/Projects/memorybench/data/runs/eval-llm/checkpoint.json"
EXPERIMENTS_LOG = "/Users/jared/Projects/openclaw-memory/experiments_opus.jsonl"
RESEARCH_LOG = "/Users/jared/Projects/openclaw-memory/research_opus.md"
AUTH_PROFILES_PATH = os.path.expanduser("~/.openclaw/agents/main/agent/auth-profiles.json")


def get_api_keys():
    with open(AUTH_PROFILES_PATH) as f:
        data = json.load(f)
    profiles = data.get("profiles", data)
    keys = {}
    for _key, val in profiles.items():
        if isinstance(val, dict):
            provider = val.get("provider", "")
            tok = val.get("token") or val.get("apiKey")
            if provider == "google":
                keys["google"] = tok
            elif provider == "anthropic":
                keys["anthropic"] = tok
            elif provider == "openai":
                keys["openai"] = tok
    return keys


API_KEYS = get_api_keys()

# ─── Search Space (for Opus to reference) ─────────────────────────────────────

SEARCH_SPACE = {
    "top_k": [3, 5, 10, 15, 20, 30, 50],
    "include_source": [True, False],
    "rerank_strategy": [
        "none",
        "similarity_threshold",
        "entity_boost",
        "temporal_recency",
        "diversity",
        "category_weight",
    ],
    "similarity_threshold": [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "temporal_weight": [0.0, 0.1, 0.2, 0.3, 0.5],
    "answer_strategy": [
        "default",
        "cot_structured",
        "entity_focused",
        "temporal_narrative",
        "concise_direct",
    ],
    "answer_model": ["gemini-3-flash-preview"],
    "max_context_memories": [3, 5, 10, 15, 20, 30],
}


# ─── Import shared components from autoresearch_bench ─────────────────────────

# Rather than duplicating, import the experiment runner
sys.path.insert(0, str(Path(__file__).parent))
from autoresearch_bench import (
    load_questions,
    run_experiment,
    search_memories,
)

# ─── Opus API ─────────────────────────────────────────────────────────────────


def call_opus(system_prompt, user_prompt, max_tokens=4096):
    """Call o3 (OpenAI) as the research agent — strongest reasoning model available."""
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEYS['openai']}",
            "Content-Type": "application/json",
        },
        json={
            "model": "o3",
            "max_completion_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ─── Research Agent Prompts ───────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an ML research agent optimizing a memory retrieval benchmark.

You're running experiments on a memory system (Supermemory) that:
1. Ingests conversation sessions and extracts atomic facts/memories using an LLM
2. Stores them in SQLite with embeddings for semantic search
3. Has entity profiles, relations (UPDATE/EXTEND/CONTRADICT), categories, temporal metadata
4. Retrieves relevant memories for a question, then uses an LLM to generate an answer
5. A judge LLM evaluates correctness

The benchmark (LongMemEval_s) has 18 questions across 6 types (3 each):
- single-session-assistant: Facts from assistant's responses in one session
- single-session-user: Facts from user's messages in one session
- single-session-preference: User preferences mentioned in conversation
- multi-session-reasoning: Requires combining facts across multiple sessions
- temporal-reasoning: Requires understanding time/order of events
- knowledge-update: Facts that were updated/corrected over time

Your job: Design experiments to maximize accuracy. You control these parameters:
{search_space}

After each round of experiments, you'll see detailed results including:
- Per-question correctness with judge explanations
- Retrieved memory contents and similarity scores
- Which question types succeed/fail and why

Think like a researcher: form hypotheses, design controlled experiments, analyze results."""

DESIGN_PROMPT = """Here are the results so far:

{history_summary}

{detailed_failures}

Based on this analysis:
1. What patterns do you see? Which parameters matter most?
2. What hypotheses do you want to test next?
3. Design exactly {n_experiments} experiments to test those hypotheses.

IMPORTANT: Each experiment must be a valid JSON config with ALL these keys:
- top_k (int)
- include_source (bool)
- rerank_strategy (string)
- similarity_threshold (float)
- temporal_weight (float)
- answer_strategy (string)
- answer_model (string, always "gemini-2.5-flash")
- max_context_memories (int)

Respond with your analysis, then a JSON block with your experiments:

```json
{{
    "hypotheses": ["hypothesis 1", "hypothesis 2", ...],
    "experiments": [
        {{"top_k": 20, "include_source": true, "rerank_strategy": "entity_boost", "similarity_threshold": 0.4, "temporal_weight": 0.1, "answer_strategy": "cot_structured", "answer_model": "gemini-2.5-flash", "max_context_memories": 15}},
        ...
    ]
}}
```"""

INITIAL_DESIGN_PROMPT = """This is the first round. No experiments have been run yet.

The benchmark has 18 questions (3 per type). Previous brute-force runs with random configs
averaged ~30% accuracy with Gemini Flash. The best single run got 55% with Claude Haiku
(top_k=10, entity_boost reranking, cot_structured prompting, max_context=20).

Key insights from prior runs:
- multi-session-reasoning: 0% across ALL configs (the hardest category)
- single-session-preference: 0% across ALL configs (retrieval can't find preferences)
- single-session-assistant: ~67% (easiest, basic retrieval works)
- Retrieval recall is high (Hit@10=94%) but precision low (21%) - lots of noise in results
- The answer model gets the right memories but can't filter signal from noise

Design {n_experiments} experiments for the first round. Include:
1. A strong baseline (your best guess at optimal config)
2. Diagnostic experiments to understand what's broken (e.g., vary only one parameter)
3. At least one aggressive/creative config

{format_instructions}"""


# ─── Detailed Failure Analysis ────────────────────────────────────────────────


def get_detailed_results(questions, config):
    """Run experiment and capture detailed per-question results for analysis."""
    detailed = []
    for q in questions:
        # Search
        memories = search_memories(
            q["question"],
            top_k=config["top_k"],
            include_source=config["include_source"],
        )

        # Import rerank from bench
        from autoresearch_bench import rerank

        memories = rerank(memories, config["rerank_strategy"], config)
        memories = memories[: config["max_context_memories"]]

        # Capture what Opus needs to see
        mem_summary = []
        for i, m in enumerate(memories[:5]):  # Top 5 for analysis
            mem_summary.append(
                {
                    "rank": i + 1,
                    "content": m["content"][:200],
                    "similarity": round(m.get("similarity", 0), 3),
                    "category": m.get("category", "unknown"),
                    "document_date": m.get("document_date", "unknown"),
                    "has_relations": len(m.get("relations", [])) > 0,
                }
            )

        detailed.append(
            {
                "question_id": q["id"],
                "question": q["question"][:200],
                "question_type": q["question_type"],
                "ground_truth": q["ground_truth"][:200],
                "num_retrieved": len(memories),
                "top_memories": mem_summary,
            }
        )

    return detailed


def format_history(history):
    """Format experiment history for Opus."""
    if not history:
        return "No experiments run yet."

    lines = []
    for h in history:
        lines.append(
            f"Exp {h['experiment_id']}: {h['accuracy']}% "
            f"(rerank={h['config']['rerank_strategy']}, "
            f"answer={h['config']['answer_strategy']}, "
            f"top_k={h['config']['top_k']}, "
            f"max_ctx={h['config']['max_context_memories']}, "
            f"src={h['config']['include_source']}, "
            f"thresh={h['config']['similarity_threshold']})"
        )
        for qtype, acc in sorted(h["by_type"].items()):
            lines.append(f"  {qtype}: {acc * 100:.0f}%")
        lines.append("")

    # Summary stats
    accs = [h["accuracy"] for h in history]
    max(history, key=lambda h: h["accuracy"])
    lines.append(
        f"--- Summary: {len(history)} experiments, "
        f"mean={sum(accs) / len(accs):.1f}%, best={max(accs)}% ---"
    )

    return "\n".join(lines)


def format_failure_details(history, questions):
    """Get detailed retrieval info for consistently failing questions."""
    if not history:
        return ""

    # Find questions that ALWAYS fail
    question_scores = {}
    for h in history:
        for r in h.get("detailed_results", []):
            qid = r["question_id"]
            question_scores.setdefault(qid, []).append(r.get("correct", False))

    always_fail = [
        qid
        for qid, scores in question_scores.items()
        if all(not s for s in scores) and len(scores) >= 2
    ]

    if not always_fail:
        return ""

    # Get retrieval details for failing questions using best config
    best = max(history, key=lambda h: h["accuracy"])
    failing_qs = [q for q in questions if q["id"] in always_fail]

    if not failing_qs:
        return ""

    details = get_detailed_results(failing_qs, best["config"])
    lines = ["\n--- Detailed retrieval for ALWAYS-FAILING questions ---"]
    for d in details:
        lines.append(f"\nQ: {d['question']}")
        lines.append(f"Type: {d['question_type']}")
        lines.append(f"Expected: {d['ground_truth']}")
        lines.append(f"Retrieved {d['num_retrieved']} memories. Top 5:")
        for m in d["top_memories"]:
            lines.append(
                f"  #{m['rank']} (sim={m['similarity']}, cat={m['category']}): {m['content']}"
            )

    return "\n".join(lines)


# ─── Main Loop ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Opus-Driven Autoresearch")
    parser.add_argument(
        "--max-rounds", type=int, default=10, help="Max rounds of Opus analysis + experiments"
    )
    parser.add_argument("--exps-per-round", type=int, default=3, help="Experiments per round")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    print(f"Loading questions from {CHECKPOINT}...")
    questions = load_questions()
    print(f"  {len(questions)} questions loaded")

    # Load history
    history = []
    if args.resume and Path(EXPERIMENTS_LOG).exists():
        with open(EXPERIMENTS_LOG) as f:
            for line in f:
                if line.strip():
                    history.append(json.loads(line))
        print(f"  Resuming from {len(history)} previous experiments")

    best_accuracy = max((h["accuracy"] for h in history), default=0)
    print(f"  Best accuracy so far: {best_accuracy}%")
    print(f"  Running {args.max_rounds} rounds × {args.exps_per_round} experiments\n")

    # Initialize research log
    with open(RESEARCH_LOG, "a") as f:
        f.write(
            f"\n\n# Opus Autoresearch Session — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        )

    n_exp = args.exps_per_round
    format_instructions = f"""Respond with your analysis, then a JSON block:

```json
{{
    "hypotheses": ["hypothesis 1", ...],
    "experiments": [
        {{"top_k": 20, "include_source": true, "rerank_strategy": "entity_boost", "similarity_threshold": 0.4, "temporal_weight": 0.1, "answer_strategy": "cot_structured", "answer_model": "gemini-2.5-flash", "max_context_memories": 15}},
        ... (exactly {n_exp} experiments)
    ]
}}
```"""

    for round_num in range(args.max_rounds):
        print(f"{'=' * 60}")
        print(f"ROUND {round_num + 1}/{args.max_rounds}")
        print(f"{'=' * 60}\n")

        # Ask Opus to design experiments
        print("Asking Opus to analyze results and design experiments...")
        t0 = time.time()

        system = SYSTEM_PROMPT.format(search_space=json.dumps(SEARCH_SPACE, indent=2))

        if not history:
            user_prompt = INITIAL_DESIGN_PROMPT.format(
                n_experiments=n_exp,
                format_instructions=format_instructions,
            )
        else:
            history_summary = format_history(history)
            failure_details = format_failure_details(history, questions)
            user_prompt = DESIGN_PROMPT.format(
                history_summary=history_summary,
                detailed_failures=failure_details,
                n_experiments=n_exp,
            )

        try:
            opus_response = call_opus(system, user_prompt)
        except Exception as e:
            print(f"  Opus call failed: {e}")
            print("  Falling back to random experiments")
            opus_response = None

        opus_time = time.time() - t0
        print(f"  Opus responded in {opus_time:.1f}s\n")

        # Parse experiments from Opus response
        experiments = []
        if opus_response:
            # Log Opus analysis
            with open(RESEARCH_LOG, "a") as f:
                f.write(f"## Round {round_num + 1}\n\n")
                f.write(f"### Opus Analysis\n\n{opus_response}\n\n")

            # Print the analysis part (before JSON)
            json_start = opus_response.find("```json")
            if json_start > 0:
                analysis = opus_response[:json_start].strip()
                print(f"  Opus Analysis:\n  {analysis[:500]}...\n")

            # Extract JSON
            try:
                json_str = opus_response[opus_response.index("{") : opus_response.rindex("}") + 1]
                parsed = json.loads(json_str)
                experiments = parsed.get("experiments", [])
                hypotheses = parsed.get("hypotheses", [])
                print("  Hypotheses:")
                for h in hypotheses[:5]:
                    print(f"    - {h}")
                print()
            except (json.JSONDecodeError, ValueError) as e:
                print(f"  Failed to parse Opus JSON: {e}")
                print(f"  Response excerpt: {opus_response[-500:]}")

        # Fallback: random experiments if Opus failed
        if not experiments:
            print("  Using random fallback configs")
            from autoresearch_bench import sample_config

            experiments = [sample_config() for _ in range(n_exp)]

        # Run each experiment
        for _i, config in enumerate(experiments):
            exp_id = len(history)

            # Ensure all required keys exist
            for key in SEARCH_SPACE:
                if key not in config:
                    config[key] = random.choice(SEARCH_SPACE[key])

            # Ensure answer_model is Gemini 3 Flash
            config["answer_model"] = "gemini-3-flash-preview"

            print(
                f"  [Exp {exp_id}] top_k={config['top_k']}, "
                f"rerank={config['rerank_strategy']}, "
                f"answer={config['answer_strategy']}, "
                f"max_ctx={config['max_context_memories']}, "
                f"src={config['include_source']}, "
                f"thresh={config['similarity_threshold']}"
            )

            t1 = time.time()
            try:
                result = run_experiment(questions, config, exp_id)
            except Exception as e:
                print(f"    ERROR: {e}")
                continue
            elapsed = time.time() - t1

            is_best = result["accuracy"] > best_accuracy
            if is_best:
                best_accuracy = result["accuracy"]

            marker = " 🏆 NEW BEST!" if is_best else ""
            print(
                f"    Accuracy: {result['accuracy']}% ({result['correct']}/{result['total']}) "
                f"in {elapsed:.1f}s{marker}"
            )

            for qtype, acc in sorted(result["by_type"].items()):
                print(f"      {qtype}: {acc * 100:.0f}%")

            # Save detailed results for Opus to analyze
            entry = {
                "experiment_id": exp_id,
                "round": round_num,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config": config,
                "accuracy": result["accuracy"],
                "correct": result["correct"],
                "total": result["total"],
                "by_type": result["by_type"],
                "elapsed_s": round(elapsed, 1),
                "is_best": is_best,
                "detailed_results": result["results"],
            }
            history.append(entry)

            with open(EXPERIMENTS_LOG, "a") as f:
                f.write(json.dumps(entry) + "\n")

            print()

        # Log round results
        with open(RESEARCH_LOG, "a") as f:
            round_exps = [h for h in history if h.get("round") == round_num]
            f.write(f"### Round {round_num + 1} Results\n\n")
            for e in round_exps:
                f.write(
                    f"- Exp {e['experiment_id']}: {e['accuracy']}% "
                    f"({e['config']['rerank_strategy']}, {e['config']['answer_strategy']})\n"
                )
            f.write(f"\nBest overall: {best_accuracy}%\n\n")

        # Early stopping: if we've plateaued
        if len(history) >= 15:
            recent = [h["accuracy"] for h in history[-10:]]
            if max(recent) - min(recent) < 3.0:
                print(
                    "\n⚠️  Plateau detected (last 10 runs within 3% of each other). "
                    "May need architectural changes, not just parameter tuning."
                )

    # Final summary
    print("\n" + "=" * 60)
    print(f"OPUS AUTORESEARCH COMPLETE: {len(history)} experiments across {args.max_rounds} rounds")
    print(f"Best accuracy: {best_accuracy}%")

    best = max(history, key=lambda h: h["accuracy"])
    print(f"\nBest config (Exp {best['experiment_id']}):")
    for k, v in best["config"].items():
        if k != "answer_model":
            print(f"  {k}: {v}")
    print("\nBy type:")
    for qtype, acc in sorted(best["by_type"].items()):
        print(f"  {qtype}: {acc * 100:.0f}%")

    print(f"\nResearch log: {RESEARCH_LOG}")
    print(f"Experiments: {EXPERIMENTS_LOG}")


if __name__ == "__main__":
    main()
