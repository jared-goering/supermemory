

# Opus Autoresearch Session — 2026-03-24 11:50



# Opus Autoresearch Session — 2026-03-24 11:53

## Round 1

### Opus Analysis

Analysis & Experiment Design

Current failure modes
1. High recall / very low precision → LLM flooded with irrelevant memories; can’t isolate the needed fact.
2. Preferences & multi-session questions fail → either facts not surfaced (preference extraction) or scattered across sessions without salience boosting.
3. Temporal questions fail → recency/order not emphasised during retrieval or reasoning.

Key levers
• Retrieval precision: top_k, similarity_threshold, rerank_strategy
• Context focus: max_context_memories
• Answer reasoning style: answer_strategy tuned to task type
• Temporal weighting to push the newest / temporally-tagged memories upward

Hypotheses
1. Adding entity_boost reranking PLUS a mild similarity filter (0.3-0.4) will cut noise while preserving recall, improving single-session-user, preference, and multi-session questions.
2. Some wrong answers stem from too much context, not too little. A minimalist retrieval (top_k=5, no rerank, no threshold) will diagnose whether the model can answer when noise is minimal; if accuracy drops, recall is the bigger issue.
3. Temporal-recency reranking with a strong temporal weight and a temporal-narrative answer prompt will improve temporal-reasoning and knowledge-update categories, even at very large top_k (breadth first, then aggressive rerank).

Experiments
1. Strong baseline: balanced recall/precision using entity boost + similarity filter; keep context manageable; CoT structured prompting.
2. Diagnostic: tiny top_k, no rerank, concise answer; isolates effect of retrieval size.
3. Aggressive/creative: very large retrieval, heavy temporal recency rerank, narrative temporal reasoning.

```json
{
    "hypotheses": [
        "Entity-boost reranking with a moderate similarity threshold will raise precision enough to solve preference and multi-session questions while maintaining high recall.",
        "If accuracy improves with a very small top_k and no reranking, the main issue is noise; if it drops, recall is the limiting factor.",
        "For temporal and knowledge-update questions, aggressively surfacing the most recent memories and prompting the model to build a temporal narrative will yield higher accuracy."
    ],
    "experiments": [
        {
            "top_k": 15,
            "include_source": true,
            "rerank_strategy": "entity_boost",
            "similarity_threshold": 0.3,
            "temporal_weight": 0.1,
            "answer_strategy": "cot_structured",
            "answer_model": "gemini-2.5-flash",
            "max_context_memories": 20
        },
        {
            "top_k": 5,
            "include_source": true,
            "rerank_strategy": "none",
            "similarity_threshold": 0.0,
            "temporal_weight": 0.0,
            "answer_strategy": "concise_direct",
            "answer_model": "gemini-2.5-flash",
            "max_context_memories": 5
        },
        {
            "top_k": 50,
            "include_source": true,
            "rerank_strategy": "temporal_recency",
            "similarity_threshold": 0.2,
            "temporal_weight": 0.5,
            "answer_strategy": "temporal_narrative",
            "answer_model": "gemini-2.5-flash",
            "max_context_memories": 30
        }
    ]
}
```

### Round 1 Results

- Exp 0: 33.33% (entity_boost, cot_structured)
- Exp 1: 27.78% (none, concise_direct)
- Exp 2: 22.22% (temporal_recency, temporal_narrative)

Best overall: 33.33%
