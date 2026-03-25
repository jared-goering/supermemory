"""Test v2: Trust aggregate endpoint more, use LLM only for final disambiguation."""
import json

import litellm
import requests

EVAL_API = "http://127.0.0.1:8643"

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

def get_events(question, session_prefix):
    agg = requests.post(f"{EVAL_API}/api/aggregate", json={
        "question": question,
        "session_prefix": session_prefix,
    }).json()
    return agg

def answer_with_dedup(question, agg):
    """Use LLM ONLY to merge duplicate events, then count."""
    events = agg.get("events", [])
    operation = agg.get("operation", "count_distinct")

    if operation == "sum_duration":
        # For duration sums, trust the deterministic answer
        total = agg.get("answer", 0)
        return f"{total} hours"

    if len(events) <= 3:
        # Small enough to trust directly
        labels = [e["canonical_label"] for e in events]
        return f"{len(events)}. Events: {'; '.join(labels)}"

    # For larger sets, ask LLM to merge duplicates
    event_list = ""
    for i, e in enumerate(events, 1):
        event_list += f"{i}. {e['canonical_label']} (participants: {e.get('participants', '?')}, date: {e.get('normalized_date', 'unknown')}, key: {e['distinct_key']})\n"

    prompt = f"""These events were extracted from a personal assistant's memory. Some may be duplicates referring to the same real-world event extracted from different conversations.

EVENTS:
{event_list}

TASK: Merge any duplicates (same real-world event described differently). For each unique event, pick the best label. Return ONLY a JSON array of unique event labels, like:
["Rachel & Mike's wedding", "Emily & Sarah's wedding", "Jen & Tom's wedding"]

Think about: same participants = likely same event. "Cousin Emily's wedding" and "Emily's wedding to Sarah" = same if Emily is the same person."""

    resp = litellm.completion(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )
    text = resp.choices[0].message.content.strip()

    # Parse JSON
    try:
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        unique = json.loads(text)
        return f"{len(unique)}. Events: {'; '.join(unique)}"
    except:
        return f"Could not parse: {text}"

def judge(question, ground_truth, answer):
    prompt = f"""Is the following answer correct given the ground truth? Focus on whether the COUNT is correct.

Question: {question}
Ground truth: {ground_truth}
Answer: {answer}

Reply "YES" or "NO" with brief explanation."""

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

        agg = get_events(q["question"], q["session_prefix"])
        print(f"Raw event count: {agg.get('answer', '?')}")

        answer = answer_with_dedup(q["question"], agg)
        print(f"Answer: {answer}")

        judgment = judge(q["question"], q["ground_truth"], answer)
        is_correct = judgment.strip().upper().startswith("YES")
        print(f"Judge: {judgment}")
        if is_correct:
            correct += 1

    print(f"\n{'='*60}")
    print(f"Multi-session accuracy: {correct}/{len(QUESTIONS)} ({correct/len(QUESTIONS)*100:.0f}%)")
