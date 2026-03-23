"""
OpenClaw Memory Engine — Integration Tests

Tests the full pipeline: ingest → search → history → profile → temporal queries.
Mocks _llm_call and embedder so tests run without API keys or model downloads.
"""

import json
import os
import re
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np

from supermemory.engine import MemoryEngine


# ── Mock LLM responses ──────────────────────────────────────────────────────

CONVERSATION_1 = """
Jared lives at 742 Evergreen Terrace, Springfield, IL.
He works as a software engineer at Acme Corp.
Jared prefers dark mode for all his IDEs.
His favorite programming language is Python.
"""

EXTRACT_RESPONSE_1 = json.dumps([
    {
        "content": "Jared lives at 742 Evergreen Terrace, Springfield, IL",
        "category": "person",
        "event_date": None,
        "confidence": 1.0,
        "entities": ["Jared"],
    },
    {
        "content": "Jared works as a software engineer at Acme Corp",
        "category": "person",
        "event_date": None,
        "confidence": 1.0,
        "entities": ["Jared", "Acme Corp"],
    },
    {
        "content": "Jared prefers dark mode for all his IDEs",
        "category": "preference",
        "event_date": None,
        "confidence": 1.0,
        "entities": ["Jared"],
    },
    {
        "content": "Jared's favorite programming language is Python",
        "category": "preference",
        "event_date": None,
        "confidence": 0.9,
        "entities": ["Jared"],
    },
])

PROFILE_RESPONSE_1 = json.dumps({
    "static_profile": {
        "name": "Jared",
        "address": "742 Evergreen Terrace, Springfield, IL",
        "employer": "Acme Corp",
        "role": "Software Engineer",
    },
    "dynamic_profile": {
        "ide_preference": "dark mode",
        "favorite_language": "Python",
    },
})

CONVERSATION_2 = """
Jared moved to 123 Main St, Austin, TX last month.
He started a new job as CTO at OpenClaw.
He still loves Python but has been writing a lot of Rust lately.
"""

EXTRACT_RESPONSE_2 = json.dumps([
    {
        "content": "Jared lives at 123 Main St, Austin, TX",
        "category": "person",
        "event_date": "2025-02-01",
        "confidence": 1.0,
        "entities": ["Jared"],
    },
    {
        "content": "Jared works as CTO at OpenClaw",
        "category": "person",
        "event_date": "2025-02-01",
        "confidence": 1.0,
        "entities": ["Jared", "OpenClaw"],
    },
    {
        "content": "Jared has been writing a lot of Rust lately",
        "category": "preference",
        "event_date": None,
        "confidence": 0.8,
        "entities": ["Jared"],
    },
])

PROFILE_RESPONSE_2 = json.dumps({
    "static_profile": {
        "name": "Jared",
        "address": "123 Main St, Austin, TX",
        "employer": "OpenClaw",
        "role": "CTO",
    },
    "dynamic_profile": {
        "ide_preference": "dark mode",
        "favorite_language": "Python",
        "recent_language": "Rust",
    },
})


def make_mock_embedder():
    """Create a mock embedder using bag-of-words hashing for meaningful similarity."""
    mock = MagicMock()

    def encode(texts, normalize_embeddings=True):
        if isinstance(texts, str):
            texts = [texts]
        vecs = []
        for t in texts:
            vec = np.zeros(384, dtype=np.float32)
            # Hash each word into a dimension — texts sharing words will be similar
            words = t.lower().split()
            for word in words:
                idx = hash(word) % 384
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vecs.append(vec)
        return np.array(vecs)

    mock.encode = encode
    return mock


def make_first_ingest_llm(profile_response=PROFILE_RESPONSE_1):
    """LLM mock for first ingestion. Detects prompt type by content."""

    def llm_call(prompt):
        if "Extract atomic memories" in prompt:
            return EXTRACT_RESPONSE_1
        elif "build a profile" in prompt:
            return profile_response
        else:
            # Relate — shouldn't happen on first ingest (no existing memories)
            return "[]"

    return llm_call


def make_second_ingest_llm(memories_1, profile_response=PROFILE_RESPONSE_2):
    """LLM mock for second ingestion. Detects prompt type by content."""

    def llm_call(prompt):
        if "Extract atomic memories" in prompt:
            return EXTRACT_RESPONSE_2
        elif "build a profile" in prompt:
            return profile_response
        elif "NEW MEMORY" in prompt:
            # Relate: find new memory IDs from prompt, match to old memories
            new_ids = re.findall(r'NEW MEMORY \(id: ([a-f0-9-]+)\)', prompt)
            old_address_id = memories_1[0]["id"]
            old_job_id = memories_1[1]["id"]
            relations = []
            for new_id in new_ids:
                idx = prompt.find(new_id)
                context = prompt[idx:idx + 300]
                if "123 Main" in context or "Austin" in context:
                    relations.append({
                        "new_id": new_id,
                        "existing_id": old_address_id,
                        "relation": "updates",
                        "context": "New address replaces old address",
                    })
                elif "CTO" in context or "OpenClaw" in context:
                    relations.append({
                        "new_id": new_id,
                        "existing_id": old_job_id,
                        "relation": "updates",
                        "context": "New job replaces old job",
                    })
            return json.dumps(relations)
        else:
            return "[]"

    return llm_call


class TestMemoryEngine(unittest.TestCase):
    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.engine = MemoryEngine(db_path=self.db_path)
        self.engine._embedder = make_mock_embedder()

    def tearDown(self):
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def test_ingest_and_search(self):
        """Test basic ingestion and search."""
        self.engine._llm_call = make_first_ingest_llm()

        memories = self.engine.ingest(
            CONVERSATION_1, session_key="session-1", agent_id="kit", document_date="2025-01-15"
        )

        self.assertEqual(len(memories), 4)
        self.assertTrue(any("742" in m["content"] for m in memories))
        self.assertTrue(any("Acme Corp" in m["content"] for m in memories))

        # Search should find results
        results = self.engine.search("What is Jared's address?")
        self.assertTrue(len(results) > 0)

        # All should be current
        for r in results:
            self.assertTrue(r["is_current"])

    def test_versioning_and_supersede(self):
        """Test that updated facts supersede old ones."""
        # First ingestion
        self.engine._llm_call = make_first_ingest_llm()
        memories_1 = self.engine.ingest(
            CONVERSATION_1, session_key="session-1", agent_id="kit", document_date="2025-01-15"
        )
        self.assertEqual(len(memories_1), 4)

        # Second ingestion — updates address and job
        self.engine._llm_call = make_second_ingest_llm(memories_1)
        memories_2 = self.engine.ingest(
            CONVERSATION_2, session_key="session-2", agent_id="kit", document_date="2025-03-01"
        )
        self.assertEqual(len(memories_2), 3)

        # Verify superseded memories exist
        stats = self.engine.get_stats()
        self.assertGreater(stats["superseded_memories"], 0)

        # Current-only search should return new address, not old
        results = self.engine.search("Jared address")
        address_results = [r for r in results if "lives" in r["content"].lower()]
        current_addresses = [r for r in address_results if r["is_current"]]
        for ca in current_addresses:
            self.assertIn("Austin", ca["content"])

        # All-versions search should include old address
        all_results = self.engine.search("Jared address", current_only=False)
        all_address = [r for r in all_results if "lives" in r["content"].lower()]
        contents = " ".join(r["content"] for r in all_address)
        self.assertIn("Springfield", contents)
        self.assertIn("Austin", contents)

    def test_temporal_as_of_date(self):
        """Test as_of_date queries return memories that were current at that time."""
        # First ingestion on 2025-01-15
        self.engine._llm_call = make_first_ingest_llm()
        memories_1 = self.engine.ingest(
            CONVERSATION_1, session_key="session-1", agent_id="kit", document_date="2025-01-15"
        )

        # Second ingestion on 2025-03-01
        self.engine._llm_call = make_second_ingest_llm(memories_1)
        self.engine.ingest(
            CONVERSATION_2, session_key="session-2", agent_id="kit", document_date="2025-03-01"
        )

        # Query as of 2025-02-01 — before second ingestion
        results_feb = self.engine.search("Jared address", as_of_date="2025-02-01")
        address_results = [r for r in results_feb if "lives" in r["content"].lower()]
        # Should only get old address (Springfield) — Austin wasn't ingested yet
        for ar in address_results:
            self.assertIn("Springfield", ar["content"])

        # Query as of 2025-04-01 — after second ingestion
        results_apr = self.engine.search("Jared address", as_of_date="2025-04-01")
        address_results = [r for r in results_apr if "lives" in r["content"].lower() and r["is_current"]]
        for ar in address_results:
            self.assertIn("Austin", ar["content"])

    def test_history(self):
        """Test get_history returns version chain."""
        self.engine._llm_call = make_first_ingest_llm()
        self.engine.ingest(CONVERSATION_1, "s1", "kit", "2025-01-15")

        history = self.engine.get_history("Jared")
        self.assertGreater(len(history), 0)
        for entry in history:
            self.assertIn("Jared", entry["content"])

    def test_profile(self):
        """Test profile creation from memories."""
        self.engine._llm_call = make_first_ingest_llm()
        self.engine.ingest(CONVERSATION_1, "s1", "kit", "2025-01-15")

        profile = self.engine.get_profile("Jared")
        self.assertIsNotNone(profile)
        self.assertEqual(profile["entity_name"], "Jared")
        self.assertIn("name", profile["static_profile"])
        self.assertEqual(profile["static_profile"]["name"], "Jared")

    def test_stats(self):
        """Test stats reporting."""
        self.engine._llm_call = make_first_ingest_llm()
        self.engine.ingest(CONVERSATION_1, "s1", "kit", "2025-01-15")

        stats = self.engine.get_stats()
        self.assertEqual(stats["total_memories"], 4)
        self.assertEqual(stats["current_memories"], 4)
        self.assertEqual(stats["sessions"], 1)

    def test_relations(self):
        """Test relation retrieval."""
        # First ingestion
        self.engine._llm_call = make_first_ingest_llm()
        memories_1 = self.engine.ingest(CONVERSATION_1, "s1", "kit", "2025-01-15")

        # Second ingestion with relations
        self.engine._llm_call = make_second_ingest_llm(memories_1)
        memories_2 = self.engine.ingest(CONVERSATION_2, "s2", "kit", "2025-03-01")

        # Find the new address memory and check its relations
        new_address = [m for m in memories_2 if "Austin" in m["content"]]
        self.assertTrue(len(new_address) > 0)

        relations = self.engine.get_relations(new_address[0]["id"])
        self.assertGreater(len(relations), 0)
        self.assertEqual(relations[0]["relation"], "updates")


if __name__ == "__main__":
    unittest.main()
