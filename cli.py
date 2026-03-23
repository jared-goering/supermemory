"""
OpenClaw Memory Engine — CLI Interface
"""

import json
import os
import sys

import click

from config import get_config, default_config_yaml, ensure_dirs
from memory_engine import MemoryEngine


def get_engine(db: str) -> MemoryEngine:
    return MemoryEngine(db_path=db)


@click.group()
@click.option("--db", default=None, help="Path to SQLite database file")
@click.pass_context
def cli(ctx, db):
    """OpenClaw Memory Engine — local-first structured agent memory."""
    ctx.ensure_object(dict)
    cfg = get_config()
    ctx.obj["db"] = db or cfg["db_path"]


@cli.command()
@click.option("--text", help="Text to ingest")
@click.option("--file", "filepath", help="File to ingest")
@click.option("--session", required=True, help="Session key")
@click.option("--agent", required=True, help="Agent ID")
@click.option("--date", default=None, help="Document date (ISO format, default: today)")
@click.pass_context
def ingest(ctx, text, filepath, session, agent, date):
    """Ingest text or file into memory."""
    if not text and not filepath:
        click.echo("Error: provide --text or --file", err=True)
        sys.exit(1)

    if filepath:
        with open(filepath, "r") as f:
            text = f.read()

    engine = get_engine(ctx.obj["db"])
    click.echo(f"Ingesting text ({len(text)} chars) from session '{session}'...")

    memories = engine.ingest(text, session_key=session, agent_id=agent, document_date=date)

    click.echo(f"\nExtracted {len(memories)} memories:")
    for m in memories:
        status = "●" if m.get("confidence", 1.0) >= 0.8 else "○"
        click.echo(f"  {status} [{m.get('category', '?')}] {m['content']}")

    click.echo(f"\nDone. Memories stored in {ctx.obj['db']}")


@cli.command()
@click.argument("query")
@click.option("--top-k", default=10, help="Number of results")
@click.option("--all-versions", is_flag=True, help="Include superseded memories")
@click.option("--as-of", default=None, help="Search as of date (ISO format)")
@click.pass_context
def search(ctx, query, top_k, all_versions, as_of):
    """Search memories by semantic similarity."""
    engine = get_engine(ctx.obj["db"])
    results = engine.search(
        query, top_k=top_k, current_only=not all_versions, as_of_date=as_of
    )

    if not results:
        click.echo("No results found.")
        return

    click.echo(f"Found {len(results)} results for: \"{query}\"\n")
    for i, r in enumerate(results, 1):
        current = "✓" if r["is_current"] else "✗"
        click.echo(f"  {i}. [{current}] [{r['category']}] {r['content']}")
        click.echo(f"     similarity: {r['similarity']:.3f} | confidence: {r['confidence']} | date: {r['document_date']} | v{r['version']}")
        if r.get("relations"):
            for rel in r["relations"]:
                click.echo(f"     ↳ {rel['relation']}: {rel['related_content']}")
        click.echo()


@cli.command()
@click.argument("entity_name")
@click.pass_context
def history(ctx, entity_name):
    """Show version history for an entity."""
    engine = get_engine(ctx.obj["db"])
    entries = engine.get_history(entity_name)

    if not entries:
        click.echo(f"No history found for '{entity_name}'.")
        return

    click.echo(f"History for '{entity_name}' ({len(entries)} entries):\n")
    for e in entries:
        current = "CURRENT" if e["is_current"] else "SUPERSEDED"
        click.echo(f"  [{current}] v{e['version']} ({e['document_date']})")
        click.echo(f"    {e['content']}")
        if e["superseded_by"]:
            click.echo(f"    → superseded by {e['superseded_by'][:8]}...")
        click.echo()


@cli.command()
@click.argument("entity_name")
@click.pass_context
def profile(ctx, entity_name):
    """Show profile for an entity."""
    engine = get_engine(ctx.obj["db"])
    p = engine.get_profile(entity_name)

    if not p:
        click.echo(f"No profile found for '{entity_name}'.")
        return

    click.echo(f"Profile: {p['entity_name']}")
    click.echo(f"Updated: {p['updated_at']}\n")

    click.echo("Static (core facts):")
    click.echo(json.dumps(p["static_profile"], indent=2))
    click.echo("\nDynamic (evolving facts):")
    click.echo(json.dumps(p["dynamic_profile"], indent=2))


@cli.command()
@click.pass_context
def stats(ctx):
    """Show memory database statistics."""
    engine = get_engine(ctx.obj["db"])
    s = engine.get_stats()

    click.echo("Memory Engine Stats")
    click.echo("=" * 40)
    click.echo(f"  Total memories:      {s['total_memories']}")
    click.echo(f"  Current memories:    {s['current_memories']}")
    click.echo(f"  Superseded memories: {s['superseded_memories']}")
    click.echo(f"  Relations:           {s['relations']}")
    click.echo(f"  Profiles:            {s['profiles']}")
    click.echo(f"  Sessions:            {s['sessions']}")
    click.echo()
    if s["categories"]:
        click.echo("  Categories:")
        for cat, count in sorted(s["categories"].items(), key=lambda x: -x[1]):
            click.echo(f"    {cat or 'uncategorized'}: {count}")


@cli.command()
@click.pass_context
def init(ctx):
    """Initialize ~/.supermemory/ with default config and empty database."""
    from pathlib import Path

    supermemory_dir = Path.home() / ".supermemory"
    config_path = supermemory_dir / "config.yaml"
    cfg = get_config()
    db_path = cfg["db_path"]

    # Create directory
    supermemory_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"Created {supermemory_dir}/")

    # Write default config if it doesn't exist
    if config_path.exists():
        click.echo(f"Config already exists: {config_path}")
    else:
        config_path.write_text(default_config_yaml())
        click.echo(f"Wrote default config: {config_path}")

    # Create empty database
    ensure_dirs(cfg)
    engine = MemoryEngine(db_path=db_path)
    click.echo(f"Database ready: {db_path}")

    stats = engine.get_stats()
    click.echo(f"  {stats['total_memories']} memories, {stats['relations']} relations")
    click.echo("\nSupermemory initialized. Edit ~/.supermemory/config.yaml to customize.")


if __name__ == "__main__":
    cli()
