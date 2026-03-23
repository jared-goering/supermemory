#!/usr/bin/env python3
"""Live auto-ingest: watches OpenClaw session JSONL files for new conversation
content and extracts atomic memories in near-real-time.

Designed for cron (every 15 min) or long-running watch mode.

Tracks per-file byte offsets so only new appended lines are processed.
Filters noise (heartbeats, tool results, system events, thinking).
Batches conversation turns into segments before LLM extraction.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ENGINE_DIR = os.environ.get("MEMORY_ENGINE_DIR", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_DIR)

from config import get_config

_cfg = get_config()
DB_PATH = _cfg["db_path"]
STATE_FILE = _cfg.get("state_file", os.path.expanduser("~/.supermemory/ingest-state.json"))
SESSIONS_ROOT = os.path.expanduser("~/.openclaw/agents")

# ── Filtering ────────────────────────────────────────────────────────────────

NOISE_PATTERNS = [
    "HEARTBEAT_OK",
    "NO_REPLY",
    "Read HEARTBEAT.md",
    "heartbeat poll",
    "[cron:",
    "System: [",
    "Exec failed",
    "Exec completed",
    "Slack message edited",
    "Internal task completion event",
    "<<<BEGIN_UNTRUSTED_CHILD_RESULT>>>",
    "<<<END_UNTRUSTED_CHILD_RESULT>>>",
    "openclaw.inbound_meta",
] + _cfg.get("skip_patterns", [])

# Only extract from these agents (skip noisy automation agents)
INGEST_AGENTS = {
    "main", "builder", "forge", "architect", "designer",
    "researcher", "oracle", "campaign", "crm", "ops",
    "swagco", "video", "marketing", "inbox", "outreach",
    "sage", "atlas",
}

# Minimum segment length worth sending to LLM
MIN_SEGMENT_CHARS = 100

# Max chars per segment (avoid huge LLM calls)
MAX_SEGMENT_CHARS = 6000

# Max age of sessions to scan (skip ancient ones)
MAX_SESSION_AGE_DAYS = 7


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"offsets": {}, "last_run": None, "total_ingested": 0, "runs": 0}


def save_state(state: dict):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def is_noise(text: str) -> bool:
    """Check if text is noise that shouldn't be ingested."""
    stripped = text.strip()
    if len(stripped) < 30:
        return True
    for pattern in NOISE_PATTERNS:
        if pattern in stripped:
            return True
    # Skip if it's mostly JSON/code (tool output)
    if stripped.startswith("{") or stripped.startswith("["):
        return True
    return False


def extract_text_from_message(msg: dict) -> str | None:
    """Extract human-readable text from a JSONL message entry."""
    inner = msg.get("message", {})
    role = inner.get("role", "")

    # Only want user and assistant messages
    if role not in ("user", "assistant"):
        return None

    content = inner.get("content", "")

    # String content
    if isinstance(content, str):
        return content if not is_noise(content) else None

    # Array content (multi-part)
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                ptype = part.get("type", "")
                if ptype == "text":
                    t = part.get("text", "")
                    if t and not is_noise(t):
                        text_parts.append(t)
                # Skip: thinking, toolCall, toolResult, image, etc.
        return "\n".join(text_parts) if text_parts else None

    return None


def scan_session_file(filepath: str, offset: int) -> tuple[list[dict], int]:
    """Read new lines from a JSONL session file starting at byte offset.
    Returns (messages, new_offset)."""
    messages = []
    try:
        size = os.path.getsize(filepath)
        if size <= offset:
            return [], offset

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            f.seek(offset)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("type") == "message":
                        text = extract_text_from_message(entry)
                        if text:
                            messages.append({
                                "text": text,
                                "role": entry["message"]["role"],
                                "timestamp": entry.get("timestamp", ""),
                                "id": entry.get("id", ""),
                            })
                except json.JSONDecodeError:
                    continue

            new_offset = f.tell()
    except (OSError, IOError) as e:
        print(f"  ! Error reading {filepath}: {e}", file=sys.stderr)
        return [], offset

    return messages, new_offset


def batch_into_segments(messages: list[dict]) -> list[str]:
    """Group sequential messages into conversation segments."""
    if not messages:
        return []

    segments = []
    current = []
    current_len = 0

    for msg in messages:
        role_label = "User" if msg["role"] == "user" else "Assistant"
        line = f"{role_label}: {msg['text']}"

        if current_len + len(line) > MAX_SEGMENT_CHARS and current:
            segments.append("\n\n".join(current))
            current = [line]
            current_len = len(line)
        else:
            current.append(line)
            current_len += len(line)

    if current and current_len >= MIN_SEGMENT_CHARS:
        segments.append("\n\n".join(current))

    return segments


def find_active_sessions() -> list[tuple[str, str, str]]:
    """Find active JSONL session files.
    Returns list of (filepath, agent_id, session_id)."""
    results = []
    cutoff = time.time() - (MAX_SESSION_AGE_DAYS * 86400)

    for agent_dir in Path(SESSIONS_ROOT).iterdir():
        if not agent_dir.is_dir():
            continue
        agent_id = agent_dir.name
        if agent_id not in INGEST_AGENTS:
            continue

        sessions_dir = agent_dir / "sessions"
        if not sessions_dir.exists():
            continue

        for fpath in sessions_dir.glob("*.jsonl"):
            # Skip deleted/reset files
            if ".deleted." in fpath.name or ".reset." in fpath.name or ".lock" in fpath.name:
                continue
            # Skip old files
            try:
                if fpath.stat().st_mtime < cutoff:
                    continue
            except OSError:
                continue

            session_id = fpath.stem
            results.append((str(fpath), agent_id, session_id))

    return results


def run_ingest_cycle(state: dict, dry_run: bool = False, verbose: bool = False) -> int:
    """Run one ingest cycle. Returns number of new memories extracted."""
    sessions = find_active_sessions()
    offsets = state.get("offsets", {})
    total_new = 0
    segments_processed = 0

    if verbose:
        print(f"Scanning {len(sessions)} active session files...", file=sys.stderr)

    engine = None
    if not dry_run:
        from memory_engine import MemoryEngine
        engine = MemoryEngine(db_path=DB_PATH)

    for filepath, agent_id, session_id in sessions:
        prev_offset = offsets.get(filepath, 0)
        messages, new_offset = scan_session_file(filepath, prev_offset)

        if not messages:
            offsets[filepath] = new_offset
            continue

        if verbose:
            print(f"  {agent_id}/{session_id}: {len(messages)} new messages", file=sys.stderr)

        segments = batch_into_segments(messages)

        for segment in segments:
            segments_processed += 1
            session_key = f"live:{agent_id}:{session_id[:8]}"

            if dry_run:
                print(f"\n{'='*60}", file=sys.stderr)
                print(f"[DRY RUN] Would ingest segment from {agent_id}/{session_id[:8]}:", file=sys.stderr)
                print(segment[:300] + ("..." if len(segment) > 300 else ""), file=sys.stderr)
                continue

            try:
                memories = engine.ingest(
                    segment,
                    session_key=session_key,
                    agent_id=agent_id,
                    document_date=datetime.now().isoformat()[:10],
                )
                n = len(memories)
                total_new += n
                if verbose and n > 0:
                    for m in memories:
                        cat = m.get("category", "?")
                        print(f"    + [{cat}] {m['content'][:80]}", file=sys.stderr)
            except Exception as e:
                print(f"  ! Ingest error ({agent_id}/{session_id[:8]}): {e}", file=sys.stderr)

        offsets[filepath] = new_offset

    state["offsets"] = offsets
    state["last_run"] = datetime.now().isoformat()
    state["total_ingested"] = state.get("total_ingested", 0) + total_new
    state["runs"] = state.get("runs", 0) + 1

    if verbose or total_new > 0:
        print(f"\nCycle complete: {segments_processed} segments, {total_new} new memories", file=sys.stderr)

    return total_new


def main():
    parser = argparse.ArgumentParser(description="Live auto-ingest from OpenClaw session transcripts")

    parser.add_argument("--dry-run", action="store_true", help="Show what would be ingested without running LLM")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--watch", action="store_true", help="Run continuously (every --interval seconds)")
    parser.add_argument("--interval", type=int, default=_cfg["ingest_interval"], help=f"Watch interval in seconds (default: {_cfg['ingest_interval']})")
    parser.add_argument("--reset", action="store_true", help="Reset offsets (re-process everything)")
    parser.add_argument("--stats", action="store_true", help="Show ingestion stats")

    args = parser.parse_args()

    if args.stats:
        state = load_state()
        print(f"Total memories ingested: {state.get('total_ingested', 0)}")
        print(f"Runs: {state.get('runs', 0)}")
        print(f"Last run: {state.get('last_run', 'never')}")
        print(f"Files tracked: {len(state.get('offsets', {}))}")
        return

    state = load_state()

    if args.reset:
        state["offsets"] = {}
        save_state(state)
        print("Offsets reset. Next run will process all recent sessions.", file=sys.stderr)
        if not args.watch:
            return

    if args.watch:
        print(f"Live ingest watching every {args.interval}s...", file=sys.stderr)
        while True:
            try:
                n = run_ingest_cycle(state, dry_run=args.dry_run, verbose=args.verbose)
                save_state(state)
                # Refresh API embedding cache if new memories were added
                if n > 0 and not args.dry_run:
                    try:
                        import urllib.request
                        req = urllib.request.Request(
                            f"http://localhost:{_cfg['api_port']}/api/cache/refresh",
                            method="POST",
                            headers={"Content-Type": "application/json"},
                            data=b"{}",
                        )
                        urllib.request.urlopen(req, timeout=10)
                        print(f"  API cache refreshed ({n} new memories)", file=sys.stderr)
                    except Exception:
                        pass  # API might not be running
            except Exception as e:
                print(f"Cycle error: {e}", file=sys.stderr)
            time.sleep(args.interval)
    else:
        n = run_ingest_cycle(state, dry_run=args.dry_run, verbose=args.verbose)
        save_state(state)
        if not args.dry_run:
            print(f"Ingested {n} new memories")


if __name__ == "__main__":
    main()
