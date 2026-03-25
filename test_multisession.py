"""Quick test: multi-session questions with event-aware answering."""
import litellm
import requests

EVAL_API = "http://127.0.0.1:8643"

# Multi-session questions from the benchmark
QUESTIONS = [
    {
        "question": "How many weddings have I attended in this year?",
        "ground_truth": "I attended three weddings. The couples were Rachel and Mike, Emily and Sarah, and Jen and Tom.",
        "session_prefix": "gpt4_2f8be40d-eval-llm",
    },
    {
        "question": "How many different art-related events did I attend in the past month?",
        "ground_truth": "4",
        "session_prefix": "2ce6a0f2-eval-llm",
    },
    {
        "question": "How many hours of jogging and yoga did I do last week?",
        "ground_truth": "0.5 hours",
        "session_prefix": "7024f17c-eval-llm",
    },
]

def get_event_context(question, session_prefix):
    """Get events from aggregate endpoint + regular search for hybrid context."""
    # Get structured events
    agg = requests.post(f"{EVAL_API}/api/aggregate", json={
        "question": question,
        "session_prefix": session_prefix,
    }).json()

    # Also get regular search results scoped to sessions
    search = requests.post(f"{EVAL_API}/api/search", json={
        "query": question,
        "top_k": 20,
    }).json()

    # Filter search results to matching sessions
    filtered = [r for r in search.get("results", [])
                if session_prefix in r.get("source_session", "")]

    return agg, filtered

def answer_with_events(question, agg_result, search_results):
    """Use events + memories as structured context for LLM answer."""
    events = agg_result.get("events", [])

    context = "## Structured Events Found\n"
    for i, e in enumerate(events, 1):
        context += f"\n{i}. **{e['canonical_label']}**\n"
        context += f"   Type: {e['event_type']}/{e.get('subtype', 'N/A')}\n"
        context += f"   Participants: {e.get('participants', '[]')}\n"
        context += f"   Date: {e.get('normalized_date', 'unknown')}\n"
        context += f"   Involvement: {e.get('user_involvement', 'unknown')}\n"
        context += f"   Distinct key: {e['distinct_key']}\n"

    context += "\n\n## Supporting Memories\n"
    for r in search_results[:15]:
        context += f"- [{r.get('source_session', '?')}] {r['content']}\n"

    prompt = f"""Based on the structured events and supporting memories below, answer the question.

IMPORTANT:
- Events may be duplicated (same event extracted from different sessions). Merge events that clearly refer to the same real-world occurrence.
- Pay attention to temporal scope in the question ("this year", "last week", "past month").
- Count only distinct real-world events, not mentions.

CONTEXT:
{context}

QUESTION: {question}

Answer concisely and precisely. If counting, give the exact number and list the distinct events."""

    resp = litellm.completion(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )
    return resp.choices[0].message.content

def judge(question, ground_truth, answer):
    """Simple judge."""
    prompt = f"""Is the following answer correct given the ground truth?

Question: {question}
Ground truth: {ground_truth}
Answer: {answer}

Reply with just "CORRECT" or "INCORRECT" and a brief explanation."""

    resp = litellm.completion(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    return resp.choices[0].message.content

if __name__ == "__main__":
    correct = 0
    for q in QUESTIONS:
        print(f"\n{'='*60}")
        print(f"Q: {q['question']}")
        print(f"GT: {q['ground_truth']}")

        agg, search = get_event_context(q["question"], q["session_prefix"])
        print(f"Events found: {agg.get('answer', '?')} clusters")

        answer = answer_with_events(q["question"], agg, search)
        print(f"Answer: {answer}")

        judgment = judge(q["question"], q["ground_truth"], answer)
        print(f"Judge: {judgment}")

        if "CORRECT" in judgment.upper().split("\n")[0]:
            correct += 1

    print(f"\n{'='*60}")
    print(f"Multi-session accuracy: {correct}/{len(QUESTIONS)} ({correct/len(QUESTIONS)*100:.0f}%)")
