"""CLI entry point for Literature Cortex.

Usage:
    lcortex init                  Initialize vault + seed library
    lcortex search <query>        Phase A: search papers
    lcortex analyze               Phase B: 4C+L scoring
    lcortex synthesize            Phase E: synthesis
    lcortex evolve                Phase F-2: conflict detection
    lcortex export                Phase G: export Obsidian vault
    lcortex status                Show graph stats
    lcortex run <query>           Full pipeline A→G
    lcortex seed                  Seed library management

All commands support --help for detailed options.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import click

from lcortex import __version__
from lcortex.core.config import get_config, Config, ensure_config_dir
from lcortex.core.engine import run_pipeline
from lcortex.core.state import (
    PHASES,
    PHASE_LABELS,
    PipelineState,
    PhaseStatus,
    load_state,
    save_state,
    checkpoint_marker,
    resume_from_checkpoint,
)

# ── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lcortex.cli")


# ── Helpers ─────────────────────────────────────────────────────────────────
def _resolve_workspace(config: Optional[Config] = None) -> Path:
    """Get workspace path for the current command."""
    if config is None:
        config = get_config()
    return Path(config.workspace.path)


def _llm_skip(phase: str) -> None:
    """Print LLM-not-configured skip message."""
    label = PHASE_LABELS.get(phase, phase)
    click.secho(
        f"⚠️  LLM not configured — skipping Phase {phase} ({label})",
        fg="yellow",
    )


def _check_llm(config: Config) -> bool:
    """Check if LLM is configured, print message if not."""
    if not config.llm.is_configured:
        click.secho("⚠️  LLM not configured. Set LCORTEX_LLM_PROVIDER and LCORTEX_LLM_API_KEY.", fg="yellow")
        click.echo("   See: lcortex init --help")
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Main CLI group
# ═══════════════════════════════════════════════════════════════════════════
@click.group()
@click.version_option(version=__version__, prog_name="lcortex")
@click.pass_context
def main(ctx: click.Context):
    """Literature Cortex — AI-powered literature research workflow.

    A standalone literature cognition system with knowledge graph,
    structural analogy, and double-loop learning.

    \b
    Quick start:
      lcortex init                     # Initialize vault + seeds
      lcortex run "FxLMS vibration"    # Full pipeline A→G
      lcortex status                   # Check progress
    """
    ctx.ensure_object(dict)
    ctx.obj["config"] = get_config()


# ═══════════════════════════════════════════════════════════════════════════
# lcortex init
# ═══════════════════════════════════════════════════════════════════════════
@main.command("init")
@click.option("--workspace", "-w", default=None,
              help="Path to workspace/vault directory (default: ~/lcortex-vault)")
@click.option("--with-seeds/--no-seeds", default=True,
              help="Initialize seed library (default: --with-seeds)")
@click.option("--seed-dir", default=None,
              help="Additional user seeds directory to merge")
@click.option("--force", "-f", is_flag=True,
              help="Re-initialize even if vault already exists")
@click.pass_context
def cmd_init(ctx: click.Context, workspace: Optional[str],
             with_seeds: bool, seed_dir: Optional[str], force: bool):
    """Initialize vault + seed library.

    Creates the workspace directory and populates it with seed concepts
    (axions, mathematics, algorithms, physical models) that form the
    initial knowledge graph.

    \b
    Examples:
      lcortex init
      lcortex init --workspace /data/research-vault
      lcortex init --force          # Re-initialize existing vault
      lcortex init --seed-dir ~/my-seeds
    """
    config = ctx.obj.get("config", get_config())
    ws = Path(workspace) if workspace else _resolve_workspace(config)

    if ws.exists() and any(ws.iterdir()) and not force:
        click.secho(f"Vault already exists at {ws}", fg="yellow")
        click.echo("Use --force to re-initialize (will not delete existing data).")
        return

    ws.mkdir(parents=True, exist_ok=True)
    click.secho(f"📁 Initializing vault at {ws}", fg="green")

    # Create standard subdirectories
    dirs = [
        "00-meta",
        "papers",
        "external/phase-c",
        "external/phase-d",
        "layer2-full/meta",
    ]
    for d in dirs:
        (ws / d).mkdir(parents=True, exist_ok=True)
    click.echo(f"   Created vault directory structure ({len(dirs)} dirs)")

    # Initialize seeds
    if with_seeds:
        click.echo("   🌱 Initializing seed library...")
        _init_seeds(ws, user_seeds_dir=seed_dir)

    # Write initial graph.json stub
    graph_file = ws / "layer2-full" / "graph.json"
    if not graph_file.exists():
        import json
        graph_file.write_text(json.dumps({
            "nodes": [],
            "edges": [],
            "ontology_snapshot": {},
            "ontology_evolution": [],
            "created_at": None,
            "version": "0.1.0",
        }, indent=2), encoding="utf-8")

    # Write meta-policy.json from built-in seeds
    _copy_meta_policy(ws)

    # Config file template
    config_dir = ensure_config_dir()
    config_file = config_dir / "config.yaml"
    if not config_file.exists():
        config_file.write_text(
            "# Literature Cortex Configuration\n"
            "# See: https://github.com/literature-cortex\n\n"
            "llm:\n"
            "  provider: deepseek     # none | openai | claude | deepseek | ollama | kimi\n"
            "  api_key: ${DEEPSEEK_API_KEY}\n"
            "  model: deepseek-v4-pro\n"
            "  fallback: ollama       # fallback if primary is unavailable\n\n"
            "workspace:\n"
            f"  path: {ws}\n",
            encoding="utf-8",
        )
        click.echo(f"   ⚙️  Created config template at {config_file}")

    click.secho(f"✅ Vault initialized: {ws}", fg="green")


def _init_seeds(ws: Path, user_seeds_dir: Optional[str] = None):
    """Copy built-in seeds into vault and merge user seeds."""
    from lcortex.seeds.manager import init_seeds
    try:
        init_seeds(str(ws))
    except Exception as e:
        click.secho(f"   ⚠️  Seed initialization skipped: {e}", fg="yellow")

    if user_seeds_dir:
        from lcortex.seeds.manager import load_user_seeds
        try:
            load_user_seeds(user_seeds_dir, str(ws))
            click.echo(f"   📦 Merged user seeds from {user_seeds_dir}")
        except Exception as e:
            click.secho(f"   ⚠️  User seed merge failed: {e}", fg="yellow")


def _copy_meta_policy(ws: Path):
    """Copy built-in meta/control_policy.json into vault."""
    import shutil
    import os

    # Find the built-in control_policy.json
    this_dir = Path(__file__).resolve().parent
    builtin = this_dir / "seeds" / "meta" / "control_policy.json"
    target = ws / "layer2-full" / "meta" / "control_policy.json"

    if builtin.exists() and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(builtin), str(target))
        click.echo("   📋 Copied meta-policy.json to vault")
    elif not builtin.exists():
        click.secho("   ⚠️  Built-in control_policy.json not found", fg="yellow")


# ═══════════════════════════════════════════════════════════════════════════
# lcortex search
# ═══════════════════════════════════════════════════════════════════════════
@main.command("search")
@click.argument("query")
@click.option("--max", "max_results", default=10, show_default=True,
              help="Maximum number of papers to retrieve")
@click.option("--year-min", type=int, default=None,
              help="Minimum publication year")
@click.option("--year-max", type=int, default=None,
              help="Maximum publication year")
@click.option("--sources", default="arxiv",
              help="Comma-separated sources: arxiv, openalex, s2")
@click.option("--stream", is_flag=True,
              help="Stream results as JSONL instead of batch")
@click.option("--workspace", "-w", default=None,
              help="Workspace path override")
@click.option("--dedup/--no-dedup", default=True,
              help="Enable DOI→arXiv→title deduplication")
@click.pass_context
def cmd_search(ctx: click.Context, query: str, max_results: int,
               year_min: Optional[int], year_max: Optional[int],
               sources: str, stream: bool, workspace: Optional[str],
               dedup: bool):
    """Phase A: Search for papers across academic sources.

    \b
    Examples:
      lcortex search "FxLMS vibration control"
      lcortex search "Koopman operator" --max 20 --year-min 2020
      lcortex search "neural ODE" --sources arxiv,s2 --stream
    """
    config = ctx.obj.get("config", get_config())
    ws = Path(workspace) if workspace else _resolve_workspace(config)

    click.secho(f"🔍 Searching: {query}", fg="cyan")
    click.echo(f"   Sources: {sources} | Max: {max_results} "
               f"| Years: {year_min or '—'}–{year_max or '—'}")

    if dedup:
        click.echo(f"   Dedup: enabled (DOI → arXiv → title)")

    # Delegate to the unified run pipeline
    click.secho("🔍 Phase A: Searching (via unified pipeline)", fg="cyan")
    ctx.invoke(cmd_run, query=query, pipeline_mode="dry", max_results=max_results,
               year_min=year_min, year_max=year_max, sources=sources,
               export_format="obsidian", workspace=workspace, resume=False, use_monitor=True)


# ═══════════════════════════════════════════════════════════════════════════
# lcortex analyze
# ═══════════════════════════════════════════════════════════════════════════
@main.command("analyze")
@click.option("--mode", "analyze_mode", default="lite",
              type=click.Choice(["lite", "auto", "deep-read"]),
              show_default=True,
              help="Scoring mode: lite (4C+L only), auto (4C+L + limitations), "
                   "deep-read (full analysis)")
@click.option("--workspace", "-w", default=None,
              help="Workspace path override")
@click.option("--input-file", default=None,
              help="JSONL file with candidate papers (default: auto-detect)")
@click.pass_context
def cmd_analyze(ctx: click.Context, analyze_mode: str,
                workspace: Optional[str], input_file: Optional[str]):
    """Phase B: 4C+L scoring of candidate papers.

    Scores each paper on:
      C1 - Completeness
      C2 - Correctness
      C3 - Clarity
      C4 - Comparison
      L  - Limitations

    Requires an LLM adapter to be configured.
    """
    config = ctx.obj.get("config", get_config())
    if not _check_llm(config):
        return

    ws = Path(workspace) if workspace else _resolve_workspace(config)
    click.secho(f"🤖 Phase B: 4C+L Scoring (via unified pipeline)", fg="cyan")
    click.secho("⚠️  Use 'lcortex run <query> --mode lite' for the full pipeline", fg="yellow")


# ═══════════════════════════════════════════════════════════════════════════
# lcortex synthesize
# ═══════════════════════════════════════════════════════════════════════════
@main.command("synthesize")
@click.option("--workspace", "-w", default=None,
              help="Workspace path override")
@click.option("--output", "-o", default=None,
              help="Output file for review (default: workspace/review.md)")
@click.option("--style", default="academic",
              type=click.Choice(["academic", "bullet", "extended"]),
              show_default=True,
              help="Review style")
@click.pass_context
def cmd_synthesize(ctx: click.Context, workspace: Optional[str],
                   output: Optional[str], style: str):
    """Phase E: Synthesize a structured literature review.

    Generates a comprehensive review from scored papers, including:
      - Research landscape overview
      - Method comparison matrix
      - Cross-paper insights
      - Identified gaps and future directions

    Requires an LLM adapter to be configured.
    """
    config = ctx.obj.get("config", get_config())
    if not _check_llm(config):
        return

    ws = Path(workspace) if workspace else _resolve_workspace(config)
    out = Path(output) if output else ws / "review.md"

    click.secho(f"📝 Phase E: Synthesis (via unified pipeline)", fg="cyan")
    click.secho("⚠️  Use 'lcortex run <query> --mode lite' for the full pipeline", fg="yellow")


# ═══════════════════════════════════════════════════════════════════════════
# lcortex evolve
# ═══════════════════════════════════════════════════════════════════════════
@main.command("evolve")
@click.option("--workspace", "-w", default=None,
              help="Workspace path override")
@click.option("--mode", "evolve_mode", default="auto",
              type=click.Choice(["auto", "force-double-loop", "single-loop-only"]),
              show_default=True,
              help="Evolution mode: auto (let Meta decide), force (always double-loop), "
                   "single (never trigger double-loop)")
@click.option("--dry-run", is_flag=True,
              help="Detect conflicts without making changes")
@click.pass_context
def cmd_evolve(ctx: click.Context, workspace: Optional[str],
               evolve_mode: str, dry_run: bool):
    """Phase F-2: Conflict detection & ontology evolution.

    Runs the Double-Loop learning process:
      1. Deconstruct papers into first-principles atoms
      2. Retrieve best-matching concepts from knowledge graph
      3. Reconstruct using existing concepts
      4. Detect conflicts and assess ontology changes

    Requires an LLM adapter to be configured.
    """
    config = ctx.obj.get("config", get_config())
    if not _check_llm(config):
        return

    ws = Path(workspace) if workspace else _resolve_workspace(config)

    click.secho(f"🔄 Phase F-2: Conflict Detection (via unified pipeline)", fg="cyan")
    click.secho("⚠️  Use 'lcortex run <query> --mode full' for the complete pipeline with conflict detection", fg="yellow")


# ═══════════════════════════════════════════════════════════════════════════
# lcortex export
# ═══════════════════════════════════════════════════════════════════════════
@main.command("export")
@click.option("--format", "export_format", default="obsidian",
              type=click.Choice(["obsidian", "json"]),
              show_default=True,
              help="Export format")
@click.option("--workspace", "-w", default=None,
              help="Workspace path override")
@click.option("--output", "-o", default=None,
              help="Output directory (default: vault root)")
@click.option("--graph-only", is_flag=True,
              help="Export only graph.json, skip paper .md files")
@click.option("--causal-map/--no-causal-map", default=True,
              help="Generate causal map Mermaid diagram")
@click.option("--lang", "target_lang", default=None,
              help="Translate output vault: zh (Chinese), ja, ko, etc. Uses LLM if available.")
@click.option("--lang-mode", "lang_mode", default="generate",
              type=click.Choice(["generate", "glossary", "none"]),
              help="Translation mode: generate (LLM, full) | glossary (LLM, terms only) | none (copy)")
@click.pass_context
def cmd_export(ctx: click.Context, export_format: str,
               workspace: Optional[str], output: Optional[str],
               graph_only: bool, causal_map: bool,
               target_lang: Optional[str], lang_mode: str):
    """Phase G: Export knowledge graph as Obsidian vault or JSON.

    Reads from the workspace's graph.db (SQLite GraphStore) and generates:
      - Paper .md files with wikilinks and structure annotations
      - Knowledge tree (hierarchical ontology view)
      - Causal map (Mermaid diagram of edges)
      - graph.json (complete serialized graph)

    \b
    Examples:
      lcortex export                          # Full Obsidian vault export
      lcortex export --graph-only             # Only graph.json
      lcortex export --format json -o out/    # JSON output to out/
      lcortex export -w /path/to/workspace    # Custom workspace
    """
    from lcortex.graph.store import GraphStore
    from lcortex.export.obsidian import export_graph_json, generate_vault_from_graphstore

    config = ctx.obj.get("config", get_config())
    ws = Path(workspace) if workspace else _resolve_workspace(config)
    out = Path(output) if output else ws

    # Resolve graph.db path — check workspace root and subdirectories
    graph_db_path = ws / "graph.db"
    if not graph_db_path.exists():
        # Try subdirectories (e.g., active-vibration-suppression-fxlms-hybrid/ inside ws)
        for candidate in sorted(ws.glob("*/graph.db")):
            graph_db_path = candidate
            break

    if not graph_db_path.exists():
        click.secho(f"❌ No graph.db found in workspace: {ws}", fg="red")
        click.echo("   Run pipeline phases A→B first to populate the graph.")
        return

    click.secho(f"📊 Phase G: Export", fg="cyan")
    click.echo(f"   Graph DB: {graph_db_path}")
    click.echo(f"   Format: {export_format}")
    click.echo(f"   Output: {out}")
    if graph_only:
        click.echo(f"   Graph-only mode (no paper .md files)")

    # Open GraphStore
    store = GraphStore(str(graph_db_path))
    stats = store.stats()
    click.echo(f"   Graph: {stats['node_count']} nodes, {stats['edge_count']} edges "
               f"({stats['causal_edges']} causal)")

    try:
        if graph_only or export_format == "json":
            # Export graph.json only
            out_path = out / "graph.json" if out.is_dir() else out
            if out.suffix != ".json":
                out_path = out / "graph.json"
            result = export_graph_json(store, out_path)
            s = result["stats"]
            click.secho(
                f"✅ Exported {s['papers']} papers, {s['edges']} edges → {out_path}",
                fg="green",
            )
        else:
            # Full Obsidian vault export
            result = generate_vault_from_graphstore(store, out, query="")
            click.secho(
                f"✅ Exported {result['papers_exported']} papers, "
                f"{result['meta_files']} meta files → {out}",
                fg="green",
            )
            click.echo(f"   Vault: {out / 'vault' if not str(out).endswith('vault') else out}")
            if result.get("errors"):
                click.secho(f"   ⚠️  {len(result['errors'])} errors (see log)", fg="yellow")

        # ── Language conversion ──────────────────────────────
        if target_lang and target_lang != "en" and not graph_only:
            vault_output = out / "vault" if (out / "vault").exists() else out
            click.echo(f"   🌐 Converting vault to {target_lang} (mode={lang_mode})...")
            
            # Get adapter for LLM-powered translation
            adapter = None
            if lang_mode in ("generate", "glossary"):
                try:
                    from lcortex.intelligence.factory import get_adapter
                    adapter = get_adapter(config)
                    if not adapter.is_available():
                        click.secho(f"   ⚠️  LLM not available — using glossary mode", fg="yellow")
                        lang_mode = "glossary" if lang_mode == "generate" else lang_mode
                except Exception:
                    pass
            
            from lcortex.export.language import convert_vault_language, estimate_translation_tokens
            
            # Quick token estimate first
            total_chars = sum(
                len(f.read_text(encoding="utf-8"))
                for f in vault_output.glob("papers/*.md") if f.is_file()
            )
            est = estimate_translation_tokens("x" * total_chars)
            click.echo(f"   📊 Estimated LLM tokens: ~{est['total']:,} ({lang_mode} mode)")
            
            if lang_mode != "none":
                lang_stats = convert_vault_language(
                    vault_output, target_lang, adapter, mode=lang_mode,
                )
                click.echo(
                    f"   ✅ {lang_stats['files_translated']} files → {lang_stats['lang_dir']}"
                )
                if lang_stats.get("tokens_used", 0) > 0:
                    click.echo(f"   📊 Tokens used: ~{lang_stats['tokens_used']:,}")

    except Exception as exc:
        click.secho(f"❌ Export failed: {exc}", fg="red")
        logger.exception("Export failed")


# ═══════════════════════════════════════════════════════════════════════════
# lcortex status
# ═══════════════════════════════════════════════════════════════════════════
@main.command("status")
@click.option("--workspace", "-w", default=None,
              help="Workspace path override")
@click.option("--verbose", "-v", is_flag=True,
              help="Show detailed status including per-phase info")
@click.pass_context
def cmd_status(ctx: click.Context, workspace: Optional[str], verbose: bool):
    """Show knowledge graph stats and pipeline status.

    Displays:
      - Node count (total, by type, by knowledge level)
      - Edge count (correlation vs causation)
      - Recent ontology evolution events
      - Current pipeline phase status
    """
    config = ctx.obj.get("config", get_config())
    ws = Path(workspace) if workspace else _resolve_workspace(config)

    click.secho(f"📊 Literature Cortex Status", fg="cyan", bold=True)
    click.echo(f"   Workspace: {ws}")

    if not ws.exists():
        click.secho("   Vault not initialized. Run 'lcortex init' first.", fg="yellow")
        return

    # Try to load state — first check exact path, then subdirectories
    state = None
    if (ws / "state.json").exists():
        state = load_state(str(ws))
    if state is None:
        for state_file in sorted(ws.glob("*/state.json")):
            s = load_state(str(state_file.parent))
            if s is not None:
                state = s
                break

    if state is not None:
        click.echo(f"\n   Active Pipeline: {state.slug}")
        click.echo(f"   Query: {state.query}")
        click.echo(f"   Mode: {state.mode}")
        click.echo(f"   Started: {state.started_at or '—'}")
        click.echo(f"   Papers: {state.passed_count} passed / {state.paper_count} total")

        click.echo(f"\n   Phase Progress:")
        for p in PHASES:
            status = state.phases[p]
            icon_map = {
                PhaseStatus.COMPLETED: "✅",
                PhaseStatus.IN_PROGRESS: "🔄",
                PhaseStatus.PENDING: "⏳",
                PhaseStatus.SKIPPED: "⏭️",
                PhaseStatus.ERROR: "❌",
            }
            icon = icon_map.get(status, "❓")
            label = PHASE_LABELS.get(p, p)
            click.echo(f"     {icon} {p}: {label} — {status.value}")

        if state.notes and verbose:
            click.echo(f"\n   Notes:")
            for note in state.notes:
                click.echo(f"     • {note}")
    else:
        click.echo(f"\n   No active pipeline state found.")

    # Graph stats from graph.json
    graph_file = ws / "layer2-full" / "graph.json"
    if graph_file.exists():
        try:
            import json
            graph = json.loads(graph_file.read_text(encoding="utf-8"))
            nodes = len(graph.get("nodes", []))
            edges = len(graph.get("edges", []))
            evo = len(graph.get("ontology_evolution", []))
            click.echo(f"\n   Knowledge Graph:")
            click.echo(f"     Nodes: {nodes}")
            click.echo(f"     Edges: {edges}")
            click.echo(f"     Ontology evolutions: {evo}")
        except Exception:
            click.echo(f"\n   Knowledge Graph: graph.json found but unreadable")

    # Seed stats
    seeds_dir = ws / "00-meta"
    if seeds_dir.exists():
        seed_count = len(list(seeds_dir.glob("*.md")))
        click.echo(f"   Seeds: {seed_count} in 00-meta/")


# ═══════════════════════════════════════════════════════════════════════════
# lcortex run
# ═══════════════════════════════════════════════════════════════════════════
@main.command("run")
@click.argument("query")
@click.option("--mode", "pipeline_mode", default="lite",
              type=click.Choice(["dry", "lite", "full"]),
              show_default=True,
              help="Pipeline mode: dry (smoke test, no LLM), lite (A→G single-loop), "
                   "full (A→G with structure extraction + synthesis)")
@click.option("--max", "max_results", default=10, show_default=True,
              help="Maximum papers to retrieve")
@click.option("--year-min", type=int, default=None,
              help="Minimum publication year")
@click.option("--year-max", type=int, default=None,
              help="Maximum publication year")
@click.option("--sources", default="arxiv",
              help="Comma-separated sources")
@click.option("--stream", is_flag=True,
              help="Stream results as JSONL")
@click.option("--export-format", default="obsidian",
              type=click.Choice(["obsidian", "json"]),
              show_default=True,
              help="Export format for Phase G")
@click.option("--workspace", "-w", default=None,
              help="Workspace path override")
@click.option("--resume", is_flag=True,
              help="Resume from last checkpoint")
@click.option("--monitor/--no-monitor", "use_monitor", default=True,
              show_default=True,
              help="Enable/disable pipeline monitor (default: --monitor)")
@click.pass_context
def cmd_run(ctx: click.Context, query: str, pipeline_mode: str,
            max_results: int, year_min: Optional[int], year_max: Optional[int],
            sources: str, stream: bool, export_format: str,
            workspace: Optional[str], resume: bool, use_monitor: bool):
    """Run the full A→G literature research pipeline.

    \b
    Pipeline modes:
      dry    Smoke test — Phase A + C/D + G only (no LLM calls)
      lite   Default — A + B + C/D + E + F + G (single-loop)
      full   Complete — A + B + C/D + E + F + G (full analysis)

    \b
    Examples:
      lcortex run "active vibration suppression" --mode dry
      lcortex run "active vibration suppression FxLMS" --mode lite
      lcortex run "Koopman operator control" --mode full --max 20
      lcortex run "neural ODE stability" --mode lite --year-min 2023
      lcortex run "transformer attention" --resume
      lcortex run "FxLMS" --no-monitor
    """
    config = ctx.obj.get("config", get_config())
    if workspace:
        config.workspace.path = workspace

    # ── Determine LLM adapter ────────────────────────────────────
    from lcortex.intelligence.factory import get_adapter
    adapter = get_adapter(config)

    # ── For dry mode, force NoOpAdapter ─────────────────────────
    if pipeline_mode == "dry":
        from lcortex.intelligence.adapters.noop import NoOpAdapter
        adapter = NoOpAdapter(config)
        click.secho(
            "🔧 Dry mode — no LLM calls. Phases B/E/F/F-2 will be skipped.",
            fg="cyan",
        )
    elif not config.llm.is_configured:
        click.secho(
            "⚠️  LLM not configured. Phases B/E/F/F-2 will be skipped.\n"
            "   Set LCORTEX_LLM_PROVIDER + LCORTEX_LLM_API_KEY for full functionality.",
            fg="yellow",
        )

    # ── Initialize Monitor ──────────────────────────────────────
    if use_monitor:
        from lcortex.monitor import PipelineMonitor
        ws_path = Path(config.workspace.path)
        monitor = PipelineMonitor(ws_path, mode=pipeline_mode)
    else:
        monitor = None

    # ── Run via RunController ───────────────────────────────────
    if use_monitor and monitor is not None:
        from lcortex.monitor import RunController
        controller = RunController(config, adapter, monitor)
        print(f"⚙️  Resource profile: {controller._rp.profile_summary}")
        summary = controller.run_full_pipeline(
            query=query,
            mode=pipeline_mode,
            max_results=max_results,
            year_min=year_min,
            year_max=year_max,
            sources=sources,
            export_format=export_format,
        )

        # Print final summary table
        if summary:
            click.echo()
            click.secho("State saved to workspace.", fg="green")
    else:
        # Fallback: use legacy engine when monitor is disabled
        run_pipeline(
            query=query,
            mode=pipeline_mode,
            config=config,
            max_results=max_results,
            year_min=year_min,
            year_max=year_max,
            sources=sources,
            stream=stream,
            export_format=export_format,
            resume=resume,
        )


# ═══════════════════════════════════════════════════════════════════════════
# lcortex seed — seed library subcommands
# ═══════════════════════════════════════════════════════════════════════════
@main.group("seed")
def cmd_seed():
    """Manage seed library (built-in + user extensions).

    \b
    Subcommands:
      list     Show all available seeds by level
      add      Add a user seed
      remove   Remove a user seed
      show     Display a seed's full content
    """


@cmd_seed.command("list")
@click.option("--level", "-l", default=None,
              help="Filter by knowledge level (L1-L6)")
@click.option("--workspace", "-w", default=None,
              help="Workspace path override")
@click.pass_context
def cmd_seed_list(ctx: click.Context, level: Optional[str], workspace: Optional[str]):
    """List all available seeds by knowledge level."""
    config = ctx.obj.get("config", get_config())
    ws = Path(workspace) if workspace else _resolve_workspace(config)

    click.secho("🌱 Seed Library", fg="cyan", bold=True)

    from lcortex.seeds.manager import list_seeds
    try:
        seeds = list_seeds(str(ws), level_filter=level)
        if not seeds:
            click.echo("   No seeds found. Run 'lcortex init' first.")
            return

        current_level = None
        for s in seeds:
            if s["level"] != current_level:
                current_level = s["level"]
                click.secho(f"\n  {current_level}:", fg="green", bold=True)
            builtin = "📦" if s.get("builtin") else "👤"
            click.echo(f"    {builtin} {s['name']}")
    except Exception as e:
        click.secho(f"⚠️  Error listing seeds: {e}", fg="yellow")


@cmd_seed.command("add")
@click.argument("path", type=click.Path(exists=True))
@click.option("--workspace", "-w", default=None,
              help="Workspace path override")
@click.pass_context
def cmd_seed_add(ctx: click.Context, path: str, workspace: Optional[str]):
    """Add a user seed from a Markdown file.

    The file should have YAML frontmatter with:
      level: L1-L6     (knowledge level)
      name: string      (seed name)
    """
    config = ctx.obj.get("config", get_config())
    ws = Path(workspace) if workspace else _resolve_workspace(config)

    src = Path(path)
    if not src.suffix == ".md":
        click.secho("Seed files must be .md with YAML frontmatter", fg="red")
        return

    # Copy to user seeds dir
    user_dir = ws / "seeds" / "user"
    user_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    dest = user_dir / src.name
    shutil.copy2(str(src), str(dest))
    click.secho(f"✅ Added seed: {src.name} → {user_dir}", fg="green")


@cmd_seed.command("remove")
@click.argument("name")
@click.option("--workspace", "-w", default=None,
              help="Workspace path override")
@click.pass_context
def cmd_seed_remove(ctx: click.Context, name: str, workspace: Optional[str]):
    """Remove a user seed by name."""
    ws = Path(workspace) if workspace else _resolve_workspace(config)
    user_dir = ws / "seeds" / "user"
    if not user_dir.exists():
        click.secho("No user seeds found.", fg="yellow")
        return

    found = False
    for f in user_dir.glob("*.md"):
        if f.stem == name or f.name == name:
            f.unlink()
            click.secho(f"✅ Removed seed: {f.name}", fg="green")
            found = True
            break

    if not found:
        click.secho(f"Seed '{name}' not found in user seeds.", fg="yellow")


@cmd_seed.command("show")
@click.argument("name")
@click.option("--workspace", "-w", default=None,
              help="Workspace path override")
@click.pass_context
def cmd_seed_show(ctx: click.Context, name: str, workspace: Optional[str]):
    """Display a seed's full content."""
    config = ctx.obj.get("config", get_config())
    ws = Path(workspace) if workspace else _resolve_workspace(config)

    # Search user seeds first, then built-in
    search_dirs = [
        ws / "seeds" / "user",
        ws / "seeds",
    ]

    found = False
    for d in search_dirs:
        if not d.exists():
            continue
        for f in d.rglob(f"{name}*.md"):
            click.echo(f.read_text(encoding="utf-8"))
            found = True
            # Also show which level dirs contain it
            for level_dir in sorted(d.glob("L*")):
                for lf in level_dir.glob(f"{name}*.md"):
                    if lf != f:
                        click.echo(lf.read_text(encoding="utf-8"))
            break
        if found:
            break

    if not found:
        click.secho(f"Seed '{name}' not found.", fg="yellow")


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
