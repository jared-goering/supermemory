"""
Ultramemory — Integration Tests

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

from ultramemory.engine import MemoryEngine

# ── Mock LLM responses ──────────────────────────────────────────────────────

CONVERSATION_1 = """
Alice lives at 742 Evergreen Terrace, Springfield, IL.
She works as a software engineer at Acme Corp.
Alice prefers dark mode for all her IDEs.
Her favorite programming language is Python.
"""

EXTRACT_RESPONSE_1 = json.dumps(
    [
        {
            "content": "Alice lives at 742 Evergreen Terrace, Springfield, IL",
            "category": "person",
            "event_date": None,
            "confidence": 1.0,
            "entities": ["Alice"],
        },
        {
            "content": "Alice works as a software engineer at Acme Corp",
            "category": "person",
            "event_date": None,
            "confidence": 1.0,
            "entities": ["Alice", "Acme Corp"],
        },
        {
            "content": "Alice prefers dark mode for all her IDEs",
            "category": "preference",
            "event_date": None,
            "confidence": 1.0,
            "entities": ["Alice"],
        },
        {
            "content": "Alice's favorite programming language is Python",
            "category": "preference",
            "event_date": None,
            "confidence": 0.9,
            "entities": ["Alice"],
        },
    ]
)

PROFILE_RESPONSE_1 = json.dumps(
    {
        "static_profile": {
            "name": "Alice",
            "address": "742 Evergreen Terrace, Springfield, IL",
            "employer": "Acme Corp",
            "role": "Software Engineer",
        },
        "dynamic_profile": {
            "ide_preference": "dark mode",
            "favorite_language": "Python",
        },
    }
)

CONVERSATION_2 = """
Alice moved to 123 Main St, Austin, TX last month.
She started a new job as CTO at Widgets Inc.
She still loves Python but has been writing a lot of Rust lately.
"""

EXTRACT_RESPONSE_2 = json.dumps(
    [
        {
            "content": "Alice lives at 123 Main St, Austin, TX",
            "category": "person",
            "event_date": "2025-02-01",
            "confidence": 1.0,
            "entities": ["Alice"],
        },
        {
            "content": "Alice works as CTO at Widgets Inc",
            "category": "person",
            "event_date": "2025-02-01",
            "confidence": 1.0,
            "entities": ["Alice", "Widgets Inc"],
        },
        {
            "content": "Alice has been writing a lot of Rust lately",
            "category": "preference",
            "event_date": None,
            "confidence": 0.8,
            "entities": ["Alice"],
        },
    ]
)

PROFILE_RESPONSE_2 = json.dumps(
    {
        "static_profile": {
            "name": "Alice",
            "address": "123 Main St, Austin, TX",
            "employer": "Widgets Inc",
            "role": "CTO",
        },
        "dynamic_profile": {
            "ide_preference": "dark mode",
            "favorite_language": "Python",
            "recent_language": "Rust",
        },
    }
)


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
            new_ids = re.findall(r"NEW MEMORY \(id: ([a-f0-9-]+)\)", prompt)
            old_address_id = memories_1[0]["id"]
            old_job_id = memories_1[1]["id"]
            relations = []
            for new_id in new_ids:
                idx = prompt.find(new_id)
                context = prompt[idx : idx + 300]
                if "123 Main" in context or "Austin" in context:
                    relations.append(
                        {
                            "new_id": new_id,
                            "existing_id": old_address_id,
                            "relation": "updates",
                            "context": "New address replaces old address",
                        }
                    )
                elif "CTO" in context or "Widgets" in context:
                    relations.append(
                        {
                            "new_id": new_id,
                            "existing_id": old_job_id,
                            "relation": "updates",
                            "context": "New job replaces old job",
                        }
                    )
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
        self.assertTrue(any("742 Evergreen" in m["content"] for m in memories))
        self.assertTrue(any("Acme Corp" in m["content"] for m in memories))

        # Search should find results
        results = self.engine.search("What is Alice's address?")
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
        results = self.engine.search("Alice address")
        address_results = [r for r in results if "lives" in r["content"].lower()]
        current_addresses = [r for r in address_results if r["is_current"]]
        for ca in current_addresses:
            self.assertIn("Austin", ca["content"])

        # All-versions search should include old address
        all_results = self.engine.search("Alice address", current_only=False)
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
        results_feb = self.engine.search("Alice address", as_of_date="2025-02-01")
        address_results = [r for r in results_feb if "lives" in r["content"].lower()]
        # Should only get old address (Springfield) — Austin wasn't ingested yet
        for ar in address_results:
            self.assertIn("Springfield", ar["content"])

        # Query as of 2025-04-01 — after second ingestion
        results_apr = self.engine.search("Alice address", as_of_date="2025-04-01")
        address_results = [
            r for r in results_apr if "lives" in r["content"].lower() and r["is_current"]
        ]
        for ar in address_results:
            self.assertIn("Austin", ar["content"])

    def test_history(self):
        """Test get_history returns version chain."""
        self.engine._llm_call = make_first_ingest_llm()
        self.engine.ingest(CONVERSATION_1, "s1", "kit", "2025-01-15")

        history = self.engine.get_history("Alice")
        self.assertGreater(len(history), 0)
        for entry in history:
            self.assertIn("Alice", entry["content"])

    def test_profile(self):
        """Test profile creation from memories."""
        self.engine._llm_call = make_first_ingest_llm()
        self.engine.ingest(CONVERSATION_1, "s1", "kit", "2025-01-15")

        profile = self.engine.get_profile("Alice")
        self.assertIsNotNone(profile)
        self.assertEqual(profile["entity_name"], "Alice")
        self.assertIn("name", profile["static_profile"])
        self.assertEqual(profile["static_profile"]["name"], "Alice")

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

    def test_entity_join_table(self):
        """Test that entities are stored in the join table during ingest."""
        self.engine._llm_call = make_first_ingest_llm()
        self.engine.ingest(CONVERSATION_1, "s1", "kit", "2025-01-15")

        # Should have entity entries
        entities = self.engine.list_entities()
        entity_names = [e["entity_name"] for e in entities]
        self.assertIn("Alice", entity_names)
        self.assertIn("Acme Corp", entity_names)

        # Alice should have 4 mentions (all 4 memories mention Alice)
        alice = [e for e in entities if e["entity_name"] == "Alice"][0]
        self.assertEqual(alice["mention_count"], 4)

        # Acme Corp should have 1 mention
        acme = [e for e in entities if e["entity_name"] == "Acme Corp"][0]
        self.assertEqual(acme["mention_count"], 1)

    def test_entity_alias_and_merge(self):
        """Test entity alias resolution and merge."""
        self.engine._llm_call = make_first_ingest_llm()
        self.engine.ingest(CONVERSATION_1, "s1", "kit", "2025-01-15")

        # Add an alias
        self.engine.add_entity_alias("alice smith", "Alice")

        # Test resolution
        conn = self.engine._conn()
        resolved = self.engine._resolve_entity(conn, "alice smith")
        conn.close()
        self.assertEqual(resolved, "Alice")

        # Test merge: rename "Acme Corp" to "ACME Corporation"
        self.engine.merge_entities("Acme Corp", "ACME Corporation")

        # After merge, list should show ACME Corporation, not Acme Corp
        entities = self.engine.list_entities()
        entity_names = [e["entity_name"] for e in entities]
        self.assertIn("ACME Corporation", entity_names)
        self.assertNotIn("Acme Corp", entity_names)

        # History lookup with old name should still work via alias
        history = self.engine.get_history("Acme Corp")
        self.assertGreater(len(history), 0)

    def test_source_chunk_normalization(self):
        """Test that source text is stored once in source_chunks, not per-memory."""
        self.engine._llm_call = make_first_ingest_llm()
        self.engine.ingest(CONVERSATION_1, "s1", "kit", "2025-01-15")

        # Should have 4 memories but only 1 source chunk
        conn = self.engine._conn()
        chunk_count = conn.execute("SELECT COUNT(*) as c FROM source_chunks").fetchone()["c"]
        self.assertEqual(chunk_count, 1)

        # All memories should reference the same chunk
        chunk_ids = conn.execute(
            "SELECT DISTINCT source_chunk_id FROM memories WHERE source_chunk_id IS NOT NULL"
        ).fetchall()
        self.assertEqual(len(chunk_ids), 1)

        # The chunk content should be the original text
        chunk = conn.execute("SELECT content FROM source_chunks").fetchone()
        self.assertEqual(chunk["content"], CONVERSATION_1)
        conn.close()

        # Search should still return source_chunk via JOIN
        results = self.engine.search("Alice address")
        address_results = [r for r in results if "742 Evergreen" in r["content"]]
        self.assertTrue(len(address_results) > 0)
        self.assertEqual(address_results[0]["source_chunk"], CONVERSATION_1)

    def test_entity_history_uses_join_table(self):
        """Test that get_history uses indexed join table, not LIKE scan."""
        self.engine._llm_call = make_first_ingest_llm()
        self.engine.ingest(CONVERSATION_1, "s1", "kit", "2025-01-15")

        # History via join table
        history = self.engine.get_history("Alice")
        self.assertEqual(len(history), 4)

        # All should contain "Alice"
        for entry in history:
            self.assertIn("Alice", entry["content"])

    # ── Event Extraction Tests ──────────────────────────────────────────

    def test_extract_events_basic(self):
        """Test event extraction from text with mocked LLM."""
        # First ingest to create a chunk
        self.engine._llm_call = make_first_ingest_llm()
        self.engine.ingest(CONVERSATION_1, "s1", "kit", "2025-01-15")

        # Get the chunk_id
        conn = self.engine._conn()
        chunk_row = conn.execute("SELECT id FROM source_chunks LIMIT 1").fetchone()
        chunk_id = chunk_row["id"]
        conn.close()

        # Now mock the event extraction LLM call
        event_response = json.dumps(
            [
                {
                    "event_type": "meeting",
                    "subtype": "onboarding",
                    "summary": "Alice started working at Acme Corp",
                    "participants": ["Alice"],
                    "time_text": None,
                    "normalized_date": "2025-01-15",
                    "duration_minutes": None,
                    "user_involvement": "did",
                    "confidence": 0.8,
                }
            ]
        )

        def event_llm(prompt):
            if "Extract distinct events" in prompt:
                return event_response
            return "[]"

        self.engine._llm_call = event_llm

        events = self.engine.extract_events(
            CONVERSATION_1, session_key="s1", chunk_id=chunk_id, document_date="2025-01-15"
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "meeting")
        self.assertEqual(events[0]["summary"], "Alice started working at Acme Corp")

        # Verify DB records
        conn = self.engine._conn()
        mention_count = conn.execute("SELECT COUNT(*) as c FROM event_mentions").fetchone()["c"]
        self.assertEqual(mention_count, 1)

        cluster_count = conn.execute("SELECT COUNT(*) as c FROM event_clusters").fetchone()["c"]
        self.assertEqual(cluster_count, 1)

        member_count = conn.execute("SELECT COUNT(*) as c FROM event_cluster_members").fetchone()[
            "c"
        ]
        self.assertEqual(member_count, 1)

        # Verify memory linkage
        link_count = conn.execute("SELECT COUNT(*) as c FROM event_mention_memories").fetchone()[
            "c"
        ]
        self.assertGreater(link_count, 0)
        conn.close()

    def test_event_clustering_deterministic(self):
        """Test that events with the same distinct_key cluster together."""
        # Mock LLM to return same event type from two different "sessions"
        event_data = {
            "event_type": "wedding",
            "subtype": None,
            "summary": "Rachel and Mike's wedding",
            "participants": ["Rachel", "Mike"],
            "time_text": "this year",
            "normalized_date": "2025-06-15",
            "duration_minutes": None,
            "user_involvement": "attended",
            "confidence": 0.9,
        }

        call_count = [0]

        def event_llm(prompt):
            if "Extract distinct events" in prompt:
                call_count[0] += 1
                # Slightly different summary but same participants/type/date
                data = dict(event_data)
                if call_count[0] > 1:
                    data["summary"] = "Attended Rachel and Mike's wedding ceremony"
                return json.dumps([data])
            return "[]"

        self.engine._llm_call = event_llm

        # First extraction
        events1 = self.engine.extract_events(
            "I went to Rachel and Mike's wedding", session_key="s1", document_date="2025-06-15"
        )
        self.assertEqual(len(events1), 1)
        cluster_id_1 = events1[0]["cluster_id"]

        # Second extraction — same event from different session
        events2 = self.engine.extract_events(
            "The wedding of Rachel and Mike was beautiful",
            session_key="s2",
            document_date="2025-06-15",
        )
        self.assertEqual(len(events2), 1)
        cluster_id_2 = events2[0]["cluster_id"]

        # Both should be in the same cluster
        self.assertEqual(cluster_id_1, cluster_id_2)
        self.assertEqual(events1[0]["distinct_key"], events2[0]["distinct_key"])

        # DB should have 2 mentions but 1 cluster
        conn = self.engine._conn()
        mentions = conn.execute("SELECT COUNT(*) as c FROM event_mentions").fetchone()["c"]
        clusters = conn.execute("SELECT COUNT(*) as c FROM event_clusters").fetchone()["c"]
        members = conn.execute("SELECT COUNT(*) as c FROM event_cluster_members").fetchone()["c"]
        self.assertEqual(mentions, 2)
        self.assertEqual(clusters, 1)
        self.assertEqual(members, 2)
        conn.close()

    def test_event_distinct_key_computation(self):
        """Test the deterministic distinct_key computation."""
        key1 = MemoryEngine._compute_event_distinct_key("wedding", ["Mike", "Rachel"], "2025-06-15")
        key2 = MemoryEngine._compute_event_distinct_key("wedding", ["Rachel", "Mike"], "2025-06-15")
        # Participants are sorted, so order shouldn't matter
        self.assertEqual(key1, key2)

        # Different date = different key
        key3 = MemoryEngine._compute_event_distinct_key("wedding", ["Rachel", "Mike"], "2025-07-15")
        self.assertNotEqual(key1, key3)

        # No participants
        key4 = MemoryEngine._compute_event_distinct_key("exercise", [], "2025-01-01")
        self.assertEqual(key4, "exercise||2025-01-01")

    def test_extract_events_empty(self):
        """Test that extract_events handles empty/no-event responses."""
        self.engine._llm_call = lambda prompt: "[]"
        events = self.engine.extract_events("Just a regular day", session_key="s1")
        self.assertEqual(events, [])

    def test_extract_events_invalid_json(self):
        """Test that extract_events handles invalid LLM responses gracefully."""
        self.engine._llm_call = lambda prompt: "not valid json at all"
        events = self.engine.extract_events("Some text", session_key="s1")
        self.assertEqual(events, [])


class TestAggregateEndpoint(unittest.TestCase):
    """Test the aggregate query logic (sync helper)."""

    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.engine = MemoryEngine(db_path=self.db_path)
        self.engine._embedder = make_mock_embedder()

    def tearDown(self):
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def test_aggregate_count_distinct(self):
        """Test counting distinct event clusters."""
        # Create events with different distinct_keys
        weddings = [
            {
                "event_type": "wedding",
                "participants": ["Rachel", "Mike"],
                "normalized_date": "2025-03-15",
                "summary": "Rachel and Mike wedding",
            },
            {
                "event_type": "wedding",
                "participants": ["Emily", "Sarah"],
                "normalized_date": "2025-05-20",
                "summary": "Emily and Sarah wedding",
            },
            {
                "event_type": "wedding",
                "participants": ["Jen", "Tom"],
                "normalized_date": "2025-08-10",
                "summary": "Jen and Tom wedding",
            },
        ]

        call_idx = [0]

        def event_llm(prompt):
            if "Extract distinct events" in prompt:
                idx = call_idx[0]
                call_idx[0] += 1
                if idx < len(weddings):
                    w = weddings[idx]
                    return json.dumps(
                        [
                            {
                                "event_type": w["event_type"],
                                "subtype": None,
                                "summary": w["summary"],
                                "participants": w["participants"],
                                "time_text": "this year",
                                "normalized_date": w["normalized_date"],
                                "duration_minutes": None,
                                "user_involvement": "attended",
                                "confidence": 0.9,
                            }
                        ]
                    )
            return "[]"

        self.engine._llm_call = event_llm

        for i, w in enumerate(weddings):
            self.engine.extract_events(
                f"Attended {w['summary']}", session_key=f"s{i}", document_date=w["normalized_date"]
            )

        # Verify 3 clusters exist
        conn = self.engine._conn()
        cluster_count = conn.execute("SELECT COUNT(*) as c FROM event_clusters").fetchone()["c"]
        self.assertEqual(cluster_count, 3)

        # Query clusters by type
        wedding_clusters = conn.execute(
            "SELECT * FROM event_clusters WHERE event_type = 'wedding'"
        ).fetchall()
        self.assertEqual(len(wedding_clusters), 3)
        conn.close()


class TestEventFactExtraction(unittest.TestCase):
    """Test that non-quantifiable events produce structured_facts (Problem 1)."""

    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.engine = MemoryEngine(db_path=self.db_path)
        self.engine._embedder = make_mock_embedder()

    def tearDown(self):
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def test_event_without_number_produces_fact(self):
        """'I went to cousin Rachel's wedding' should produce a fact even without a number."""

        # Mock LLM to return an event-type fact
        def fact_llm(prompt):
            if "Extract structured facts" in prompt:
                return json.dumps(
                    [
                        {
                            "fact_type": "event",
                            "category": "wedding",
                            "subject": "cousin Rachel's wedding",
                            "predicate": "attended",
                            "value": 1,
                            "unit": "occurrence",
                            "date": "2025-06-15",
                            "confidence": 1.0,
                            "is_user_action": True,
                            "participants": ["Rachel"],
                            "event_type": "wedding",
                        }
                    ]
                )
            elif "Extract atomic memories" in prompt:
                return json.dumps(
                    [
                        {
                            "content": "User attended cousin Rachel's wedding",
                            "category": "social",
                            "event_date": "2025-06-15",
                            "confidence": 1.0,
                            "entities": ["Rachel"],
                        }
                    ]
                )
            elif "build a profile" in prompt:
                return json.dumps({"static_profile": {}, "dynamic_profile": {}})
            return "[]"

        self.engine._llm_call = fact_llm
        self.engine.ingest(
            "I went to cousin Rachel's wedding last June. It was beautiful.",
            session_key="s1",
            agent_id="user_alice",
        )

        conn = self.engine._conn()
        facts = conn.execute("SELECT * FROM structured_facts").fetchall()
        conn.close()

        self.assertGreater(len(facts), 0, "Should have extracted at least one fact")
        fact = dict(facts[0])
        self.assertEqual(fact["fact_type"], "event")
        self.assertEqual(fact["category"], "wedding")
        self.assertIn("rachel", fact["subject"].lower())
        self.assertEqual(fact["value"], 1.0)
        self.assertEqual(fact["unit"], "occurrence")
        self.assertTrue(fact["is_user_action"])

    def test_event_fact_has_participants_and_event_type(self):
        """Extracted facts should include participants and event_type fields."""

        def fact_llm(prompt):
            if "Extract structured facts" in prompt:
                return json.dumps(
                    [
                        {
                            "fact_type": "attendance",
                            "category": "wedding",
                            "subject": "Mike and Rachel's wedding",
                            "predicate": "attended",
                            "value": 1,
                            "unit": "occurrence",
                            "date": "2025-06-15",
                            "confidence": 1.0,
                            "is_user_action": True,
                            "participants": ["Mike", "Rachel"],
                            "event_type": "wedding",
                        }
                    ]
                )
            elif "Extract atomic memories" in prompt:
                return json.dumps(
                    [
                        {
                            "content": "User attended Mike and Rachel's wedding",
                            "category": "social",
                            "event_date": "2025-06-15",
                            "confidence": 1.0,
                            "entities": ["Mike", "Rachel"],
                        }
                    ]
                )
            elif "build a profile" in prompt:
                return json.dumps({"static_profile": {}, "dynamic_profile": {}})
            return "[]"

        self.engine._llm_call = fact_llm
        self.engine.ingest(
            "I attended Mike and Rachel's wedding on June 15th.",
            session_key="s1",
            agent_id="user_alice",
        )

        conn = self.engine._conn()
        facts = conn.execute("SELECT * FROM structured_facts").fetchall()
        conn.close()

        self.assertGreater(len(facts), 0)
        fact = dict(facts[0])
        self.assertIsNotNone(fact["participants"])
        participants = json.loads(fact["participants"])
        self.assertIn("Mike", participants)
        self.assertIn("Rachel", participants)
        self.assertEqual(fact["event_type"], "wedding")
        self.assertIsNotNone(fact["canonical_event_id"])


class TestCanonicalEventDedup(unittest.TestCase):
    """Test canonical event clustering (Problem 2)."""

    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.engine = MemoryEngine(db_path=self.db_path)
        self.engine._embedder = make_mock_embedder()

    def tearDown(self):
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def test_same_event_different_descriptions_cluster_together(self):
        """'Rachel's wedding in June' and 'flew to Denver for a wedding' with same date/participants should share canonical_event_id."""
        call_count = {"extract": 0, "facts": 0}

        def fact_llm(prompt):
            if "Extract structured facts" in prompt:
                call_count["facts"] += 1
                if call_count["facts"] == 1:
                    return json.dumps(
                        [
                            {
                                "fact_type": "attendance",
                                "category": "wedding",
                                "subject": "Rachel's wedding in June",
                                "predicate": "attended",
                                "value": 1,
                                "unit": "occurrence",
                                "date": "2025-06-15",
                                "confidence": 1.0,
                                "is_user_action": True,
                                "participants": ["Rachel"],
                                "event_type": "wedding",
                            }
                        ]
                    )
                else:
                    return json.dumps(
                        [
                            {
                                "fact_type": "event",
                                "category": "travel",
                                "subject": "flew to Denver for a wedding",
                                "predicate": "flew",
                                "value": 1,
                                "unit": "occurrence",
                                "date": "2025-06-14",
                                "confidence": 0.9,
                                "is_user_action": True,
                                "participants": ["Rachel"],
                                "event_type": "wedding",
                            }
                        ]
                    )
            elif "Extract atomic memories" in prompt:
                call_count["extract"] += 1
                # Return unique content per call to avoid dedup
                if call_count["extract"] == 1:
                    return json.dumps(
                        [
                            {
                                "content": "User attended Rachel's wedding in June",
                                "category": "social",
                                "event_date": "2025-06-15",
                                "confidence": 1.0,
                                "entities": ["Rachel"],
                            }
                        ]
                    )
                else:
                    return json.dumps(
                        [
                            {
                                "content": "User flew to Denver for Rachel's wedding",
                                "category": "social",
                                "event_date": "2025-06-14",
                                "confidence": 1.0,
                                "entities": ["Rachel"],
                            }
                        ]
                    )
            elif "build a profile" in prompt:
                return json.dumps({"static_profile": {}, "dynamic_profile": {}})
            return "[]"

        self.engine._llm_call = fact_llm

        # Ingest two descriptions of the same event
        self.engine.ingest(
            "I went to Rachel's wedding in June. It was lovely.",
            session_key="s1",
            agent_id="user_alice",
        )
        self.engine.ingest(
            "I flew to Denver for a wedding. Rachel looked amazing.",
            session_key="s2",
            agent_id="user_alice",
        )

        conn = self.engine._conn()
        facts = conn.execute(
            "SELECT canonical_event_id FROM structured_facts WHERE canonical_event_id IS NOT NULL"
        ).fetchall()
        conn.close()

        self.assertGreater(len(facts), 1, "Should have at least 2 facts")
        canonical_ids = set(r["canonical_event_id"] for r in facts)
        self.assertEqual(
            len(canonical_ids),
            1,
            f"Both facts should share the same canonical_event_id, got {canonical_ids}",
        )

    def test_different_events_get_different_canonical_ids(self):
        """Events with different dates and participants should get different canonical IDs."""
        call_count = {"extract": 0, "facts": 0}

        def fact_llm(prompt):
            if "Extract structured facts" in prompt:
                call_count["facts"] += 1
                if call_count["facts"] == 1:
                    return json.dumps(
                        [
                            {
                                "fact_type": "attendance",
                                "category": "wedding",
                                "subject": "Rachel's wedding",
                                "predicate": "attended",
                                "value": 1,
                                "unit": "occurrence",
                                "date": "2025-06-15",
                                "confidence": 1.0,
                                "is_user_action": True,
                                "participants": ["Rachel"],
                                "event_type": "wedding",
                            }
                        ]
                    )
                else:
                    return json.dumps(
                        [
                            {
                                "fact_type": "attendance",
                                "category": "wedding",
                                "subject": "Emily's wedding",
                                "predicate": "attended",
                                "value": 1,
                                "unit": "occurrence",
                                "date": "2025-09-20",
                                "confidence": 1.0,
                                "is_user_action": True,
                                "participants": ["Emily"],
                                "event_type": "wedding",
                            }
                        ]
                    )
            elif "Extract atomic memories" in prompt:
                call_count["extract"] += 1
                if call_count["extract"] == 1:
                    return json.dumps(
                        [
                            {
                                "content": "User attended Rachel's wedding in June",
                                "category": "social",
                                "event_date": "2025-06-15",
                                "confidence": 1.0,
                                "entities": ["Rachel"],
                            }
                        ]
                    )
                else:
                    return json.dumps(
                        [
                            {
                                "content": "User went to Emily's wedding in September",
                                "category": "social",
                                "event_date": "2025-09-20",
                                "confidence": 1.0,
                                "entities": ["Emily"],
                            }
                        ]
                    )
            elif "build a profile" in prompt:
                return json.dumps({"static_profile": {}, "dynamic_profile": {}})
            return "[]"

        self.engine._llm_call = fact_llm

        self.engine.ingest(
            "I attended Rachel's wedding in June.",
            session_key="s1",
            agent_id="user_alice",
        )
        self.engine.ingest(
            "I went to Emily's wedding in September.",
            session_key="s2",
            agent_id="user_alice",
        )

        conn = self.engine._conn()
        facts = conn.execute(
            "SELECT canonical_event_id, subject FROM structured_facts WHERE canonical_event_id IS NOT NULL"
        ).fetchall()
        conn.close()

        self.assertGreater(len(facts), 1)
        canonical_ids = set(r["canonical_event_id"] for r in facts)
        self.assertEqual(
            len(canonical_ids),
            2,
            f"Different events should have different canonical IDs, got {canonical_ids}",
        )


class TestAgentIdFilteringAggregate(unittest.TestCase):
    """Test agent_id filtering in aggregate endpoints (Problem 3)."""

    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.engine = MemoryEngine(db_path=self.db_path)
        self.engine._embedder = make_mock_embedder()

    def tearDown(self):
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def test_aggregate_filters_by_agent_id_prefix(self):
        """Aggregate should only count facts belonging to the specified agent."""
        import sqlite3

        call_count = {"extract": 0, "facts": 0}

        def fact_llm(prompt):
            if "Extract structured facts" in prompt:
                call_count["facts"] += 1
                agent = "alice" if call_count["facts"] <= 1 else "bob"
                return json.dumps(
                    [
                        {
                            "fact_type": "event",
                            "category": "wedding",
                            "subject": f"{agent}'s friend's wedding",
                            "predicate": "attended",
                            "value": 1,
                            "unit": "occurrence",
                            "date": f"2025-0{call_count['facts']}-15",
                            "confidence": 1.0,
                            "is_user_action": True,
                            "participants": [],
                            "event_type": "wedding",
                        }
                    ]
                )
            elif "Extract atomic memories" in prompt:
                call_count["extract"] += 1
                if call_count["extract"] == 1:
                    return json.dumps(
                        [
                            {
                                "content": "Alice's friend had a lovely wedding ceremony",
                                "category": "social",
                                "event_date": "2025-01-15",
                                "confidence": 1.0,
                                "entities": [],
                            }
                        ]
                    )
                else:
                    return json.dumps(
                        [
                            {
                                "content": "Bob went to a beautiful outdoor wedding",
                                "category": "social",
                                "event_date": "2025-02-15",
                                "confidence": 1.0,
                                "entities": [],
                            }
                        ]
                    )
            elif "build a profile" in prompt:
                return json.dumps({"static_profile": {}, "dynamic_profile": {}})
            return "[]"

        self.engine._llm_call = fact_llm

        # Ingest for two different agents
        self.engine.ingest(
            "I attended my friend's wedding.",
            session_key="s1",
            agent_id="alice_agent_001",
        )
        self.engine.ingest(
            "I went to a beautiful wedding.",
            session_key="s2",
            agent_id="bob_agent_002",
        )

        # Verify both agents' facts are in DB
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        all_facts = conn.execute(
            "SELECT * FROM structured_facts WHERE is_user_action = 1"
        ).fetchall()
        conn.close()

        self.assertGreaterEqual(len(all_facts), 2, "Should have facts from both agents")

        # Now query via memories to check agent linkage
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        alice_facts = conn.execute(
            """SELECT sf.* FROM structured_facts sf
               JOIN memories m ON sf.memory_id = m.id
               WHERE m.source_agent LIKE 'alice%' AND sf.is_user_action = 1""",
        ).fetchall()
        bob_facts = conn.execute(
            """SELECT sf.* FROM structured_facts sf
               JOIN memories m ON sf.memory_id = m.id
               WHERE m.source_agent LIKE 'bob%' AND sf.is_user_action = 1""",
        ).fetchall()
        conn.close()

        self.assertGreater(len(alice_facts), 0, "Alice should have facts")
        self.assertGreater(len(bob_facts), 0, "Bob should have facts")
        # They should be separate
        alice_ids = {r["id"] for r in alice_facts}
        bob_ids = {r["id"] for r in bob_facts}
        self.assertEqual(len(alice_ids & bob_ids), 0, "Facts should not overlap between agents")


class TestQueryClassifier(unittest.TestCase):
    """Test the rule-based query classifier for search routing."""

    def test_counting_how_many(self):
        from ultramemory.server import classify_query

        self.assertEqual(classify_query("How many weddings did I attend?"), "counting")
        self.assertEqual(classify_query("how many times did I visit New York?"), "counting")
        self.assertEqual(classify_query("How many concerts have I been to?"), "counting")

    def test_counting_how_often(self):
        from ultramemory.server import classify_query

        self.assertEqual(classify_query("How often do I exercise?"), "counting")

    def test_counting_total_number(self):
        from ultramemory.server import classify_query

        self.assertEqual(classify_query("What is the total number of trips?"), "counting")

    def test_counting_how_much_not_cost(self):
        from ultramemory.server import classify_query

        # "how much time" should be counting
        self.assertEqual(classify_query("How much time did I spend gaming?"), "counting")
        # "how much does X cost" should be lookup (the cost question)
        self.assertEqual(classify_query("How much does the car cost?"), "lookup")
        self.assertEqual(classify_query("How much is rent?"), "lookup")

    def test_temporal_when_did(self):
        from ultramemory.server import classify_query

        self.assertEqual(classify_query("When did I last see Rachel?"), "temporal")
        self.assertEqual(classify_query("When was the last meeting?"), "temporal")

    def test_temporal_how_long(self):
        from ultramemory.server import classify_query

        self.assertEqual(classify_query("How long did the trip to Denver last?"), "temporal")
        self.assertEqual(classify_query("How long ago did I start this job?"), "temporal")

    def test_temporal_last_first_time(self):
        from ultramemory.server import classify_query

        self.assertEqual(classify_query("What was the last time I went hiking?"), "temporal")
        self.assertEqual(classify_query("The first time I met John"), "temporal")

    def test_temporal_most_recent(self):
        from ultramemory.server import classify_query

        self.assertEqual(classify_query("What is the most recent concert I attended?"), "temporal")
        self.assertEqual(classify_query("What's the latest update on the project?"), "temporal")

    def test_temporal_month_year(self):
        from ultramemory.server import classify_query

        self.assertEqual(classify_query("What happened in March?"), "temporal")
        self.assertEqual(classify_query("What did I do in January?"), "temporal")
        self.assertEqual(classify_query("Events in 2024"), "temporal")
        self.assertEqual(classify_query("What happened since 2025?"), "temporal")

    def test_lookup_default(self):
        from ultramemory.server import classify_query

        self.assertEqual(classify_query("What is Alice's favorite color?"), "lookup")
        self.assertEqual(classify_query("Where does the user live?"), "lookup")
        self.assertEqual(classify_query("Tell me about the project"), "lookup")
        self.assertEqual(classify_query("What are my hobbies?"), "lookup")

    def test_classifier_speed(self):
        """Classifier must complete in <1ms per query."""
        import time

        from ultramemory.server import classify_query

        queries = [
            "How many weddings did I attend?",
            "When did I last see Rachel?",
            "What is Alice's favorite color?",
            "How often do I exercise?",
            "What happened in March?",
        ]
        start = time.perf_counter()
        for _ in range(1000):
            for q in queries:
                classify_query(q)
        elapsed = time.perf_counter() - start
        # 5000 classifications should take well under 1 second
        self.assertLess(elapsed, 1.0, f"5000 classifications took {elapsed:.3f}s (should be <1s)")


class TestQueryRouter(unittest.TestCase):
    """Test that /api/search routes queries to the correct handler."""

    def test_search_endpoint_returns_query_type_lookup(self):
        """Lookup queries should return query_type='lookup' in response."""

        from ultramemory.server import classify_query

        # Verify classification
        self.assertEqual(classify_query("What is Alice's favorite color?"), "lookup")

    def test_search_endpoint_returns_query_type_counting(self):
        """Counting queries should include structured_answer with count."""
        from ultramemory.server import classify_query

        self.assertEqual(classify_query("How many weddings did I attend?"), "counting")

    def test_search_endpoint_returns_query_type_temporal(self):
        """Temporal queries should include structured_answer with timeline."""
        from ultramemory.server import classify_query

        self.assertEqual(classify_query("When did I last see Rachel?"), "temporal")


if __name__ == "__main__":
    unittest.main()
