#!/usr/bin/env python3
"""
Overnight pipeline: Ingest all remaining multi-session benchmark questions,
backfill structured facts, re-run benchmark, then backfill production DB.

Steps:
  1. Ingest remaining ~118 multi-session benchmark sessions into eval DB
  2. Backfill structured facts on new data
  3. Re-run benchmark on all testable questions
  4. Backfill production memory.db with structured facts
"""

import glob
import json
import os
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ─── Config ────────────────────────────────────────────────────────────
EVAL_API = os.environ.get("EVAL_API", "http://127.0.0.1:8901")
PROD_API = os.environ.get("PROD_API", "http://127.0.0.1:8643")
EVAL_DB = "/tmp/memorybench_eval.db"
PROD_DB = os.path.expanduser("~/Projects/openclaw-memory/memory.db")
QUESTIONS_DIR = os.path.expanduser(
    "~/Projects/memorybench/data/benchmarks/longmemeval/datasets/questions"
)
WORKERS = 10  # concurrent sessions per question
INGEST_TIMEOUT = 300  # seconds per session ingest


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def check_server(url, name):
    try:
        r = requests.get(f"{url}/api/health", timeout=5)
        r.raise_for_status()
        log(f"✅ {name} healthy at {url}")
        return True
    except Exception as e:
        log(f"❌ {name} unreachable at {url}: {e}")
        return False


def get_ingested_sessions(db_path):
    """Get set of source_session tags already in the DB."""
    conn = sqlite3.connect(db_path, timeout=30)
    rows = conn.execute(
        "SELECT DISTINCT source_session FROM memories WHERE source_session LIKE 'bench_%'"
    ).fetchall()
    conn.close()
    return {r[0] for r in rows}


def get_ingested_qids(db_path):
    """Get set of question IDs already ingested in the DB (heuristic)."""
    sessions = get_ingested_sessions(db_path)
    qids = set()
    for src in sessions:
        # Extract qid from bench_<qid>-<rest>
        after_bench = src[len("bench_"):]
        # qid is the first part before the first "-" unless it starts with "gpt4_"
        if after_bench.startswith("gpt4_"):
            parts = after_bench.split("-", 1)
            qids.add(parts[0])
        else:
            parts = after_bench.split("-", 1)
            qids.add(parts[0])
    return qids


def load_multi_session_questions():
    """Load all multi-session questions from dataset."""
    questions = []
    for f in glob.glob(os.path.join(QUESTIONS_DIR, "*.json")):
        q = json.load(open(f))
        if q.get("question_type") == "multi-session":
            questions.append(q)
    return questions


def ingest_session(api_url, messages, container_tag, session_id, date=None):
    """Ingest one session via API."""
    text_parts = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
        elif isinstance(msg, str):
            role = "user"
            content = msg
        else:
            continue
        text_parts.append(f"{role}: {content}")

    text = "\n".join(text_parts)
    if not text.strip():
        return 0

    payload = {
        "text": text,
        "session_key": container_tag,
        "source": "longmemeval-benchmark",
    }
    if date:
        payload["metadata"] = {"date": date}

    try:
        r = requests.post(
            f"{api_url}/api/ingest", json=payload, timeout=INGEST_TIMEOUT
        )
        r.raise_for_status()
        data = r.json()
        return data.get("count", data.get("memories_created", 0))
    except Exception as e:
        log(f"    ⚠ Session {session_id} failed: {e}")
        return -1


def ingest_question(api_url, question, existing_sessions=None):
    """Ingest all sessions for one benchmark question, concurrently.
    Skips sessions whose container_tag already exists in the DB."""
    qid = question["question_id"]
    sessions = question.get("haystack_sessions", [])
    session_ids = question.get("haystack_session_ids", [])
    dates = question.get("haystack_dates", [])

    existing_sessions = existing_sessions or set()
    total_memories = 0
    failed = 0
    skipped = 0

    def _do_one(i):
        sess = sessions[i]
        sid = session_ids[i] if i < len(session_ids) else f"session-{i}"
        d = dates[i] if i < len(dates) else None
        container_tag = f"bench_{qid}-{sid}"

        if container_tag in existing_sessions:
            return 0  # Already ingested

        count = ingest_session(api_url, sess, container_tag, sid, d)
        if count < 0:
            time.sleep(2)
            count = ingest_session(api_url, sess, container_tag, sid, d)
        return count

    # Check how many will be skipped
    for i in range(len(sessions)):
        sid = session_ids[i] if i < len(session_ids) else f"session-{i}"
        tag = f"bench_{qid}-{sid}"
        if tag in existing_sessions:
            skipped += 1

    if skipped == len(sessions):
        return 0, 0  # All sessions already ingested

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(_do_one, i): i for i in range(len(sessions))}
        for future in as_completed(futures):
            count = future.result()
            if count < 0:
                failed += 1
            else:
                total_memories += count

    return total_memories, failed


# ─── Step 1: Ingest remaining benchmark sessions ──────────────────────
def step1_ingest():
    log("=" * 60)
    log("STEP 1: Ingesting remaining multi-session benchmark data")
    log("=" * 60)

    if not check_server(EVAL_API, "Eval server"):
        log("Eval server not running! Starting it...")
        # Try to start it
        env_str = (
            f"GOOGLE_API_KEY={os.environ.get('GOOGLE_API_KEY', '')} "
            f"GEMINI_API_KEY={os.environ.get('GEMINI_API_KEY', '')} "
            f"ULTRAMEMORY_DB_PATH={EVAL_DB} "
            "ULTRAMEMORY_EMBEDDING_PROVIDER=local "
            "ULTRAMEMORY_EMBEDDING_MODEL=all-MiniLM-L6-v2 "
            "ULTRAMEMORY_MODEL=gemini/gemini-2.5-flash "
            "ULTRAMEMORY_FAST_INGEST=1"
        )
        subprocess.Popen(
            f"{env_str} uvicorn ultramemory.server:app --port 8901",
            shell=True,
            stdout=open("/tmp/eval_server.log", "a"),
            stderr=subprocess.STDOUT,
            cwd=os.path.expanduser("~/Projects/openclaw-memory"),
        )
        time.sleep(10)
        if not check_server(EVAL_API, "Eval server"):
            log("Failed to start eval server. Aborting.")
            return False

    all_questions = load_multi_session_questions()
    ingested = get_ingested_qids(EVAL_DB)
    remaining = [q for q in all_questions if q["question_id"] not in ingested]
    remaining.sort(key=lambda q: len(q.get("haystack_sessions", [])))

    log(f"Total multi-session questions: {len(all_questions)}")
    log(f"Already ingested: {len(ingested)}")
    log(f"Remaining to ingest: {len(remaining)}")

    total_sessions = sum(len(q.get("haystack_sessions", [])) for q in remaining)
    log(f"Total sessions to process: {total_sessions}")

    # Pre-load existing sessions for dedup
    existing_sessions = get_ingested_sessions(EVAL_DB)
    log(f"Existing session tags in DB: {len(existing_sessions)}")

    start = time.time()
    completed = 0
    total_new_memories = 0

    for q in remaining:
        qid = q["question_id"]
        n_sessions = len(q.get("haystack_sessions", []))
        log(f"  [{completed + 1}/{len(remaining)}] Ingesting {qid} ({n_sessions} sessions)...")

        memories, failures = ingest_question(EVAL_API, q, existing_sessions)
        total_new_memories += memories
        completed += 1

        elapsed = time.time() - start
        rate = completed / (elapsed / 60) if elapsed > 0 else 0
        eta = (len(remaining) - completed) / rate if rate > 0 else 0
        log(
            f"    → {memories} memories, {failures} failures "
            f"({completed}/{len(remaining)}, {rate:.1f} q/min, ETA: {eta:.0f}min)"
        )

        # Refresh cache every 10 questions
        if completed % 10 == 0:
            try:
                requests.post(f"{EVAL_API}/api/cache/refresh", timeout=120)
            except Exception:
                pass

    elapsed = time.time() - start
    log(f"Step 1 complete: {total_new_memories} new memories in {elapsed / 60:.1f}min")
    return True


# ─── Step 2: Backfill structured facts on eval DB ─────────────────────
def step2_backfill_eval():
    log("=" * 60)
    log("STEP 2: Backfilling structured facts on eval DB")
    log("=" * 60)

    # Use the existing backfill script
    env = os.environ.copy()
    env["ULTRAMEMORY_DB_PATH"] = EVAL_DB
    env["ULTRAMEMORY_EMBEDDING_PROVIDER"] = "local"
    env["ULTRAMEMORY_EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"
    env["ULTRAMEMORY_MODEL"] = "gemini/gemini-2.5-flash"
    result = subprocess.run(
        [sys.executable, "backfill_facts.py"],
        cwd=os.path.expanduser("~/Projects/openclaw-memory"),
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    log(f"Backfill stdout (last 500 chars): {result.stdout[-500:]}")
    if result.returncode != 0:
        log(f"Backfill stderr: {result.stderr[-500:]}")
    return result.returncode == 0


# ─── Step 3: Re-run benchmark ─────────────────────────────────────────
def step3_benchmark():
    log("=" * 60)
    log("STEP 3: Running benchmark on all testable questions")
    log("=" * 60)

    env = os.environ.copy()
    env["ULTRAMEMORY_DB_PATH"] = EVAL_DB
    env["ULTRAMEMORY_EMBEDDING_PROVIDER"] = "local"
    env["ULTRAMEMORY_EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"
    env["ULTRAMEMORY_MODEL"] = "gemini/gemini-2.5-flash"

    result = subprocess.run(
        [
            sys.executable,
            "bench_multisession.py",
            "--strategy",
            "structured",
        ],
        cwd=os.path.expanduser("~/Projects/openclaw-memory"),
        env=env,
        capture_output=True,
        text=True,
        timeout=7200,
    )
    log("Benchmark output (last 2000 chars):")
    log(result.stdout[-2000:])
    if result.returncode != 0:
        log(f"Benchmark stderr: {result.stderr[-500:]}")
    return result.returncode == 0


# ─── Step 4: Backfill production DB ───────────────────────────────────
def step4_backfill_prod():
    log("=" * 60)
    log("STEP 4: Backfilling structured facts on production DB")
    log("=" * 60)

    if not check_server(PROD_API, "Production server"):
        log("Production server not running — skipping prod backfill.")
        log("Run manually: ULTRAMEMORY_DB_PATH=memory.db python backfill_facts.py")
        return False

    env = os.environ.copy()
    env["ULTRAMEMORY_DB_PATH"] = PROD_DB
    env["ULTRAMEMORY_EMBEDDING_PROVIDER"] = "litellm"
    env["ULTRAMEMORY_EMBEDDING_MODEL"] = "gemini/gemini-embedding-2-preview"
    env["ULTRAMEMORY_MODEL"] = "gemini/gemini-2.5-flash"
    result = subprocess.run(
        [sys.executable, "backfill_facts.py"],
        cwd=os.path.expanduser("~/Projects/openclaw-memory"),
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    log(f"Prod backfill stdout (last 500 chars): {result.stdout[-500:]}")
    if result.returncode != 0:
        log(f"Prod backfill stderr: {result.stderr[-500:]}")
    return result.returncode == 0


# ─── Step 5: Full memorybench eval with Gemini embeddings ─────────────
def step5_full_eval():
    log("=" * 60)
    log("STEP 5: Running full memorybench evaluation suite")
    log("=" * 60)

    memorybench_dir = os.path.expanduser("~/Projects/memorybench")
    if not os.path.isdir(memorybench_dir):
        log("memorybench directory not found! Skipping.")
        return False

    # Start a fresh eval server on port 8643 (memorybench default) with normal mode
    # Using a FRESH eval DB so memorybench does its own ingestion
    eval_db_fresh = "/tmp/memorybench_fulleval.db"
    log(f"Starting eval server on :8643 with fresh DB: {eval_db_fresh}")

    # Kill any existing server on 8643
    subprocess.run("lsof -ti :8643 | xargs kill -9 2>/dev/null", shell=True)
    time.sleep(1)

    google_key = os.environ.get("GOOGLE_API_KEY", "")
    server_env = os.environ.copy()
    server_env.update({
        "ULTRAMEMORY_DB_PATH": eval_db_fresh,
        "ULTRAMEMORY_EMBEDDING_PROVIDER": "litellm",
        "ULTRAMEMORY_EMBEDDING_MODEL": "gemini/gemini-embedding-2-preview",
        "ULTRAMEMORY_MODEL": "gemini/gemini-2.5-flash",
        "GOOGLE_API_KEY": google_key,
        "GEMINI_API_KEY": google_key,
    })
    # Remove fast ingest flag — we want full pipeline for eval
    server_env.pop("ULTRAMEMORY_FAST_INGEST", None)
    server_env.pop("ULTRAMEMORY_SKIP_FACTS", None)
    server_env.pop("ULTRAMEMORY_SKIP_PROFILES", None)

    server_proc = subprocess.Popen(
        ["uvicorn", "ultramemory.server:app", "--port", "8643"],
        cwd=os.path.expanduser("~/Projects/openclaw-memory"),
        env=server_env,
        stdout=open("/tmp/eval_fulleval_server.log", "w"),
        stderr=subprocess.STDOUT,
    )
    time.sleep(10)

    if not check_server("http://127.0.0.1:8643", "Full eval server"):
        log("Failed to start full eval server!")
        server_proc.kill()
        return False

    # Run memorybench for each benchmark: longmemeval, locomo, convomem
    # Use gemini-2.5-flash as judge (cheaper than GPT-4o)
    benchmarks = ["longmemeval"]  # start with longmemeval; add others if available
    run_id = "eval-gemini-emb2"
    judge = "gemini-2.5-flash"
    answering_model = "gemini-2.5-flash"

    bench_env = os.environ.copy()
    bench_env.update({
        "OPENCLAW_SUPERMEMORY_URL": "http://127.0.0.1:8643",
        "GOOGLE_API_KEY": google_key,
        "GEMINI_API_KEY": google_key,
    })

    all_ok = True
    for bench in benchmarks:
        log(f"Running {bench} benchmark (run: {run_id})...")
        result = subprocess.run(
            [
                "bun", "run", "src/index.ts", "run",
                "-p", "openclaw-supermemory",
                "-b", bench,
                "-j", judge,
                "-m", answering_model,
                "-r", run_id,
            ],
            cwd=memorybench_dir,
            env=bench_env,
            capture_output=True,
            text=True,
            timeout=14400,  # 4 hours per benchmark
        )
        log(f"{bench} output (last 3000 chars):")
        log(result.stdout[-3000:])
        if result.returncode != 0:
            log(f"{bench} stderr: {result.stderr[-1000:]}")
            all_ok = False

    # Clean up server
    server_proc.terminate()
    try:
        server_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server_proc.kill()

    return all_ok


# ─── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log("🌙 Starting overnight pipeline")
    log(f"Eval DB: {EVAL_DB}")
    log(f"Prod DB: {PROD_DB}")

    results = {}

    results["step1"] = step1_ingest()
    results["step2"] = step2_backfill_eval()
    results["step3"] = step3_benchmark()
    results["step4"] = step4_backfill_prod()
    results["step5"] = step5_full_eval()

    log("=" * 60)
    log("🌅 OVERNIGHT PIPELINE COMPLETE")
    for step, ok in results.items():
        status = "✅" if ok else "❌"
        log(f"  {status} {step}")
    log("=" * 60)
