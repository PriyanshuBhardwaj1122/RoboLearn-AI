"""
tests/dual_model_kg_builder.py
═══════════════════════════════════════════════════════════════════
Dual-model KG builder that maximises daily Groq free tier tokens:

  Phase 1: llama-3.1-8b-instant   → 500k tokens/day → ~560 chunks
  Phase 2: llama-3.3-70b-versatile → 100k tokens/day → ~112 chunks
  
  Total per day: ~672 chunks (vs 560 with 8b alone)
  
Both models write to the SAME Neo4j graph.
MERGE prevents duplicates — safe to overlap or retry.

Progress is tracked in data/processed/kg_progress.json so you
never process the same chunk twice across runs or model switches.

Usage:
    python tests/dual_model_kg_builder.py             # full dual run
    python tests/dual_model_kg_builder.py --8b-only   # 8b model only
    python tests/dual_model_kg_builder.py --70b-only  # 70b model only
    python tests/dual_model_kg_builder.py --status    # show progress
    python tests/dual_model_kg_builder.py --reset     # reset progress
═══════════════════════════════════════════════════════════════════
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.settings import settings
from ingestion.document_loader import DocumentLoader
from graph.kg_builder import KnowledgeGraphBuilder
from generation.llm_client import LLMClient

# ── Colours ───────────────────────────────────────────────────────────────────
GRN = "\033[92m"; BLU = "\033[94m"; YLW = "\033[93m"
CYN = "\033[96m"; RED = "\033[91m"; GRY = "\033[90m"
BLD = "\033[1m";  RST = "\033[0m"
SEP = GRY + "─" * 68 + RST

# ── Model configs ─────────────────────────────────────────────────────────────
# ── Model configs ─────────────────────────────────────────────────────────────
# When using Ollama: both "8b" and "70b" keys use the same local model
# When using Groq:   "8b" = llama-3.1-8b-instant, "70b" = llama-3.3-70b-versatile
def _get_models():
    provider = settings.llm_provider.lower()
    if provider == "ollama":
        # Ollama: use the same model for both phases — no rate limits
        return {
            "8b": {
                "id":               settings.llm_model,
                "tpd":              999_999_999,   # unlimited
                "rpm":              999,            # unlimited
                "tokens_per_chunk": 892,
                "chunks_per_day":   999_999,        # unlimited — run all remaining
                "color":            BLU,
                "label":            f"{settings.llm_model} (Ollama local)",
            },
            "70b": {
                "id":               settings.llm_model,
                "tpd":              0,              # skip 70b phase for Ollama
                "rpm":              0,
                "tokens_per_chunk": 892,
                "chunks_per_day":   0,              # 0 = skip this phase
                "color":            CYN,
                "label":            f"{settings.llm_model} (Ollama — skip duplicate phase)",
            },
        }
    else:
        # Groq: original dual-model strategy
        return {
            "8b": {
                "id":               "llama-3.1-8b-instant",
                "tpd":              500_000,
                "rpm":              30,
                "tokens_per_chunk": 892,
                "chunks_per_day":   560,
                "color":            BLU,
                "label":            "llama-3.1-8b-instant (Groq)",
            },
            "70b": {
                "id":               "llama-3.3-70b-versatile",
                "tpd":              100_000,
                "rpm":              30,
                "tokens_per_chunk": 892,
                "chunks_per_day":   112,
                "color":            CYN,
                "label":            "llama-3.3-70b-versatile (Groq)",
            },
        }

MODELS = _get_models()

PROGRESS_FILE = Path("data/processed/kg_progress.json")


# ══════════════════════════════════════════════════════════════════════════════
#  PROGRESS TRACKER
# ══════════════════════════════════════════════════════════════════════════════

def load_progress() -> dict:
    """Load which chunks have already been processed."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {
        "processed_chunk_ids": [],
        "total_processed":     0,
        "runs": [],
    }


def save_progress(progress: dict) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def print_status(progress: dict, total_chunks: int) -> None:
    processed = progress["total_processed"]
    remaining  = total_chunks - processed
    pct        = (processed / total_chunks * 100) if total_chunks else 0

    print(f"\n{BLD}KG Build Progress{RST}")
    print(SEP)
    print(f"  Total chunks      : {total_chunks}")
    print(f"  Processed         : {GRN}{processed}{RST} ({pct:.1f}%)")
    print(f"  Remaining         : {YLW}{remaining}{RST}")
    print(f"  Est. days left    : {remaining/672:.1f} days (dual-model)")
    print()

    if progress["runs"]:
        print(f"  {BLD}Run history:{RST}")
        for run in progress["runs"][-5:]:
            print(f"    {GRY}{run['date']}  model={run['model']:30s}  "
                  f"chunks={run['chunks_done']:4d}  "
                  f"entities={run['entities']:4d}  "
                  f"relations={run['relations']:4d}{RST}")


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE-MODEL RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_model(model_key: str, all_chunks: list, progress: dict,
              dry_run: bool = False) -> dict:
    """
    Run extraction with one model until its daily token budget is exhausted
    or all remaining chunks are processed.
    Returns updated run stats.
    """
    cfg = _get_models()[model_key]
    c   = cfg["color"]

    # Filter to unprocessed chunks only
    done_ids   = set(progress["processed_chunk_ids"])
    remaining  = [ch for ch in all_chunks if ch.chunk_id not in done_ids]
    budget     = cfg["chunks_per_day"]
    to_process = remaining[:budget]

    print(f"\n{c}{BLD}{'═'*68}{RST}")
    print(f"{c}{BLD}  Model: {cfg['label']}{RST}")
    print(f"{c}  Daily budget  : {cfg['chunks_per_day']} chunks  "
          f"({cfg['tpd']:,} tokens/day){RST}")
    print(f"{c}  Unprocessed   : {len(remaining)} chunks remaining{RST}")
    print(f"{c}  Will process  : {len(to_process)} chunks this run{RST}")
    print(f"{c}{BLD}{'═'*68}{RST}\n")

    if not to_process:
        print(f"  {GRN}Nothing to do — all chunks already processed ✓{RST}")
        return {"chunks_done": 0, "entities": 0, "relations": 0,
                "skipped": 0, "model": cfg["id"]}

    if dry_run:
        print(f"  {YLW}DRY RUN — not making API calls{RST}")
        return {"chunks_done": len(to_process), "entities": 0,
                "relations": 0, "skipped": 0, "model": cfg["id"]}

    # Override model in settings for this run
    original_model = settings.llm_model
    settings.llm_model = cfg["id"]

    # Build with rate-limit awareness
    total_entities  = 0
    total_relations = 0
    skipped         = 0
    newly_done      = []

    try:
        from graph.kg_builder import KnowledgeGraphBuilder, KGEntity, KGRelation
        import re, json as _json

        builder = KnowledgeGraphBuilder()
        llm     = builder.llm

        for i, chunk in enumerate(to_process, 1):

            # Rate limit: 30 RPM → sleep between calls
            if i > 1:
                # Ollama is local — no rate limit needed
                # Groq — sleep to stay under 30 RPM
                if settings.llm_provider.lower() != "ollama":
                    time.sleep(2.1)
                else:
                    time.sleep(0.1)

            try:
                # Extract
                entities, relations = builder._extract(chunk)

                # Write to Neo4j
                builder._upsert_entities(entities)
                builder._upsert_relations(relations)

                total_entities  += len(entities)
                total_relations += len(relations)
                newly_done.append(chunk.chunk_id)

                # Progress line
                bar_len  = 30
                filled   = int(bar_len * i / len(to_process))
                bar      = "█" * filled + "░" * (bar_len - filled)
                pct      = i / len(to_process) * 100
                sys.stdout.write(
                    f"\r  {c}[{bar}]{RST} {pct:5.1f}%  "
                    f"chunk {i}/{len(to_process)}  "
                    f"ent={total_entities}  rel={total_relations}  "
                )
                sys.stdout.flush()

            except RuntimeError as e:
                # Daily token limit hit — stop cleanly
                if "daily token limit" in str(e).lower() or "tpd" in str(e).lower():
                    print(f"\n\n  {YLW}⚠ Daily token limit reached after "
                          f"{i-1} chunks{RST}")
                    break
                else:
                    print(f"\n  {RED}Error on {chunk.chunk_id}: {e}{RST}")
                    skipped += 1
                    continue

            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    # Per-minute limit — wait and continue
                    wait_match = __import__('re').search(
                        r"try again in ([\d.]+)s", str(e))
                    wait = float(wait_match.group(1)) if wait_match else 62
                    wait = min(wait + 2, 65)
                    print(f"\n  {YLW}Rate limited — waiting {wait:.0f}s...{RST}")
                    time.sleep(wait)
                    # Retry once
                    try:
                        entities, relations = builder._extract(chunk)
                        builder._upsert_entities(entities)
                        builder._upsert_relations(relations)
                        total_entities  += len(entities)
                        total_relations += len(relations)
                        newly_done.append(chunk.chunk_id)
                    except Exception:
                        skipped += 1
                else:
                    skipped += 1

        builder.close()
        print()  # newline after progress bar

    finally:
        settings.llm_model = original_model

    # Update progress
    progress["processed_chunk_ids"].extend(newly_done)
    progress["total_processed"] += len(newly_done)

    run_record = {
        "date":       datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model":      cfg["id"],
        "chunks_done":len(newly_done),
        "entities":   total_entities,
        "relations":  total_relations,
        "skipped":    skipped,
    }
    progress["runs"].append(run_record)
    save_progress(progress)

    print(f"\n  {GRN}✓ Done:{RST} {len(newly_done)} chunks  "
          f"| {total_entities} entities  "
          f"| {total_relations} relations  "
          f"| {skipped} skipped")

    return run_record


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Dual-model KG builder — maximises Groq free tier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Daily schedule:
  Run once per day. The script automatically:
    1. Uses llama-3.1-8b-instant first (~560 chunks, 500k tokens)
    2. Switches to llama-3.3-70b-versatile (~112 chunks, 100k tokens)
    3. Saves progress so you never re-process a chunk
    4. Total: ~672 chunks/day → full 2202-chunk KG in ~4 days
        """
    )
    parser.add_argument("--8b-only",  action="store_true",
                        help="Only use 8b model (560 chunks/day)")
    parser.add_argument("--70b-only", action="store_true",
                        help="Only use 70b model (112 chunks/day)")
    parser.add_argument("--status",   action="store_true",
                        help="Show build progress only")
    parser.add_argument("--reset",    action="store_true",
                        help="Reset progress tracker (does NOT clear Neo4j)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Show what would be processed without API calls")
    args = parser.parse_args()

    print(f"\n{BLD}{'═'*68}")
    print(f"  Dual-Model KG Builder")
    print(f"  8b: 560 chunks/day  +  70b: 112 chunks/day  =  672 chunks/day")
    print(f"{'═'*68}{RST}")

    # Load chunks
    print(f"\n  Loading chunks...")
    loader  = DocumentLoader()
    chunks  = loader.load_chunks()
    print(f"  Total chunks : {GRN}{len(chunks)}{RST}")

    # Load progress
    progress = load_progress()

    if args.reset:
        if input("\n  Reset progress? (y/n): ").lower() == "y":
            progress = {"processed_chunk_ids":[], "total_processed":0, "runs":[]}
            save_progress(progress)
            print(f"  {GRN}Progress reset ✓  (Neo4j graph NOT cleared){RST}")
        return

    if args.status:
        print_status(progress, len(chunks))
        return

    # Print current status
    print_status(progress, len(chunks))

    # ── Phase 1: 8b model ─────────────────────────────────────────────────────
    if not getattr(args, "70b_only", False):
        run_8b = run_model("8b", chunks, progress,
                           dry_run=args.dry_run)

    # ── Phase 2: 70b model (skipped for Ollama — same model, no extra budget) ──
    if not getattr(args, "8b_only", False):
        models = _get_models()
        if models["70b"]["chunks_per_day"] > 0:
            print(f"\n  {YLW}Switching to 70b model for remaining budget...{RST}")
            run_70b = run_model("70b", chunks, progress,
                                dry_run=args.dry_run)
        else:
            print(f"\n  {GRY}70b phase skipped (Ollama — single model, no rate limits){RST}")

    # ── Final summary ─────────────────────────────────────────────────────────
    total_done = progress["total_processed"]
    remaining  = len(chunks) - total_done
    pct        = total_done / len(chunks) * 100

    print(f"\n{BLD}{'═'*68}")
    print(f"  Today's run complete")
    print(f"{'═'*68}{RST}")
    print(f"  Total processed : {GRN}{total_done}/{len(chunks)} ({pct:.1f}%){RST}")
    print(f"  Remaining       : {YLW}{remaining} chunks{RST}")
    if remaining > 0:
        print(f"  Est. days left  : {remaining/672:.1f} days")
        print(f"\n  {GRY}Run again tomorrow after midnight UTC to continue.{RST}")
    else:
        print(f"\n  {GRN}{BLD}✓ Knowledge graph fully built!{RST}")
        print(f"  Run: python tests/kg_build_test.py --verify")


if __name__ == "__main__":
    main()