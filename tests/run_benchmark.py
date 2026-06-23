"""
tests/run_benchmark.py
Runs the 25-query benchmark against the live FAISS + Neo4j + LLM system.

Usage:
    # Retrieval only — NO LLM calls, instant, free
    python tests/run_benchmark.py --level 1
    python tests/run_benchmark.py --all

    # With LLM answers (uses Ollama or Groq per .env)
    python tests/run_benchmark.py --level 1 --answer
    python tests/run_benchmark.py --level 3 --answer
    python tests/run_benchmark.py --all --answer

    # Single query
    python tests/run_benchmark.py --id L3Q1
    python tests/run_benchmark.py --id L4Q1 --answer

    # Quick smoke test — one query per level
    python tests/run_benchmark.py --smoke
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings.vector_store import VectorStore
from graph.kg_retriever import KGRetriever
from retrieval.hybrid_retriever import HybridRetriever
from embeddings.bm25_store import BM25Store

GRN = "\033[92m"; BLU = "\033[94m"; YLW = "\033[93m"
CYN = "\033[96m"; RED = "\033[91m"; GRY = "\033[90m"
BLD = "\033[1m";  RST = "\033[0m"
SEP  = GRY + "─" * 68 + RST
SEP2 = BLD + "═" * 68 + RST

BENCHMARK_FILE = os.path.join(os.path.dirname(__file__), "benchmark_queries.json")

LEVEL_LABELS = {
    1: "Factual Lookup",
    2: "Single Relationship",
    3: "Multi-Entity Cross-Relationship",
    4: "Multi-hop Chain Reasoning",
    5: "Cross-Document Synthesis",
}


# ── Load benchmark ────────────────────────────────────────────────────────────
def load_queries(level=None, query_id=None):
    with open(BENCHMARK_FILE) as f:
        data = json.load(f)
    all_queries = []
    for lvl_key, lvl_data in data["levels"].items():
        lvl_num = int(lvl_key.split("_")[1])
        for q in lvl_data["queries"]:
            q["level"] = lvl_num
            q["level_label"] = LEVEL_LABELS[lvl_num]
            all_queries.append(q)
    if query_id:
        result = [q for q in all_queries if q["id"] == query_id]
        if not result:
            print(f"\n  {RED}Query ID '{query_id}' not found.{RST}")
            print(f"  Valid IDs: {[q['id'] for q in all_queries]}")
            sys.exit(1)
        return result
    if level:
        return [q for q in all_queries if q["level"] == level]
    return all_queries


# ── Score retrieval ───────────────────────────────────────────────────────────
def score_retrieval(context, query_meta):
    scores = {}

    # Source recall — did we retrieve from the right PDF(s)?
    expected = query_meta["expected_source"]
    if isinstance(expected, str):
        expected = [expected]
    retrieved_sources = {c.source_file for c, _ in context.chunks}
    hits = sum(1 for s in expected if any(s in r for r in retrieved_sources))
    scores["source_recall"] = round(hits / max(len(expected), 1), 2)

    # Entity recall — did expected entities appear in chunks or graph?
    expected_ents = [e.lower() for e in query_meta["expected_entities"]]
    combined_text = (
        " ".join(c.text.lower() for c, _ in context.chunks) + " " +
        " ".join(f"{t.source} {t.target}".lower() for t in context.triples)
    )
    ent_hits = sum(1 for e in expected_ents if e in combined_text)
    scores["entity_recall"] = round(ent_hits / max(len(expected_ents), 1), 2)

    # Graph stats
    scores["graph_fired"]  = len(context.triples) > 0
    scores["triple_count"] = len(context.triples)
    scores["chunk_count"]  = len(context.chunks)
    scores["top_score"]    = round(context.chunks[0][1], 4) if context.chunks else 0.0

    return scores


# ── Print one query result ────────────────────────────────────────────────────
def run_query(q, retriever, generator=None):
    print(f"\n  {BLD}{q['id']}{RST}  [{GRY}{q['category']}{RST}]")
    print(f"  {BLD}Query:{RST} \"{q['query']}\"")
    print(f"  {SEP}")

    # Retrieval
    t0 = time.time()
    context = retriever.retrieve(q["query"])
    ms = (time.time() - t0) * 1000

    scores = score_retrieval(context, q)

    # Vector results
    print(f"\n  {BLU}📦 VECTOR{RST}  {scores['chunk_count']} chunks · {ms:.0f}ms · top={scores['top_score']}")
    for i, (chunk, score) in enumerate(context.chunks, 1):
        print(f"    [{i}] {score:.4f}  {chunk.source_file}  page {chunk.page_number}")
        preview = chunk.text[:160].replace("\n", " ").strip()
        print(f"         {GRY}{preview}...{RST}")

    # Graph results
    print(f"\n  {CYN}🔗 GRAPH{RST}   {scores['triple_count']} triples · fired={GRN if scores['graph_fired'] else RED}{scores['graph_fired']}{RST}")
    for t in context.triples:
        print(f"    {CYN}{t.to_string()}{RST}")
    if not context.triples:
        print(f"    {GRY}(none — Neo4j offline or no matching entities){RST}")

    # Scores
    print(f"\n  {YLW}📊 SCORES{RST}")
    src_bar  = "█" * int(scores['source_recall'] * 10) + "░" * (10 - int(scores['source_recall'] * 10))
    ent_bar  = "█" * int(scores['entity_recall'] * 10) + "░" * (10 - int(scores['entity_recall'] * 10))
    print(f"    source_recall  [{src_bar}] {scores['source_recall']:.0%}  (expected: {q['expected_source']})")
    print(f"    entity_recall  [{ent_bar}] {scores['entity_recall']:.0%}  ({len(q['expected_entities'])} entities expected)")
    print(f"    retrieval_type : {q['retrieval_type']}")

    # Ground truth
    print(f"\n  {GRY}📖 GROUND TRUTH:{RST}")
    gt = q["ground_truth"]
    for line in (gt[:300] + "..." if len(gt) > 300 else gt).split("\n"):
        if line.strip():
            print(f"    {GRY}{line}{RST}")

    # LLM answer
    if generator:
        print(f"\n  {GRN}💬 LLM ANSWER{RST}")
        print(f"  {SEP}")
        try:
            ans = generator.generate(q["query"], context)
            for line in ans.answer.split("\n"):
                print(f"  {line}")
            if ans.source_pages:
                print(f"\n  {YLW}📄 CITED SOURCES:{RST}")
                for sp in ans.source_pages:
                    print(f"    [{sp['n']}] {sp['file']} — page {sp['page']}")
        except Exception as e:
            print(f"  {RED}LLM error: {e}{RST}")

    print(f"\n  {GRY}tags: {', '.join(q['metadata_tags'][:6])}{RST}")
    return scores


# ── Print level header ────────────────────────────────────────────────────────
def print_level_header(level, queries):
    label = LEVEL_LABELS.get(level, "")
    print(f"\n{SEP2}")
    print(f"{BLD}  Level {level} — {label}  ({len(queries)} queries){RST}")
    print(SEP2)


# ── Print summary ─────────────────────────────────────────────────────────────
def print_summary(all_scores):
    if not all_scores:
        return
    print(f"\n{SEP2}")
    print(f"{BLD}  BENCHMARK SUMMARY{RST}")
    print(SEP2)

    # Per-level breakdown
    levels = sorted(set(s["level"] for s in all_scores))
    print(f"\n  {'Level':<8} {'Queries':<8} {'Src Recall':<12} {'Ent Recall':<12} {'Graph Fired':<12} {'Avg Top Score'}")
    print(f"  {'-'*64}")
    for lvl in levels:
        lvl_scores = [s for s in all_scores if s["level"] == lvl]
        avg_src  = sum(s["source_recall"]  for s in lvl_scores) / len(lvl_scores)
        avg_ent  = sum(s["entity_recall"]  for s in lvl_scores) / len(lvl_scores)
        graph_ct = sum(1 for s in lvl_scores if s["graph_fired"])
        avg_top  = sum(s["top_score"] for s in lvl_scores) / len(lvl_scores)
        label    = LEVEL_LABELS.get(lvl, "")[:22]
        print(f"  L{lvl} {label:<24} {len(lvl_scores):<6}   "
              f"{GRN}{avg_src:.0%}{RST}        "
              f"{GRN}{avg_ent:.0%}{RST}        "
              f"{GRN}{graph_ct}/{len(lvl_scores)}{RST}          "
              f"{avg_top:.4f}")

    # Overall
    print(f"  {'-'*64}")
    avg_src   = sum(s["source_recall"]  for s in all_scores) / len(all_scores)
    avg_ent   = sum(s["entity_recall"]  for s in all_scores) / len(all_scores)
    graph_ct  = sum(1 for s in all_scores if s["graph_fired"])
    avg_top   = sum(s["top_score"] for s in all_scores) / len(all_scores)
    print(f"  {'OVERALL':<30} {len(all_scores):<6}   "
          f"{BLD}{GRN}{avg_src:.0%}{RST}        "
          f"{BLD}{GRN}{avg_ent:.0%}{RST}        "
          f"{BLD}{GRN}{graph_ct}/{len(all_scores)}{RST}          "
          f"{avg_top:.4f}")

    print(f"\n  {GRY}source_recall  = correct PDF retrieved / expected PDFs{RST}")
    print(f"  {GRY}entity_recall  = expected entities found in chunks+graph / total expected{RST}")
    print(f"  {GRY}graph_fired    = queries where Neo4j returned at least 1 triple{RST}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Run 25-query benchmark against FAISS + Neo4j",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Retrieval only (no LLM, instant)
  python tests/run_benchmark.py --level 1
  python tests/run_benchmark.py --all

  # With LLM answers
  python tests/run_benchmark.py --level 3 --answer
  python tests/run_benchmark.py --id L4Q1 --answer

  # Quick smoke test — 1 query per level
  python tests/run_benchmark.py --smoke
        """
    )
    parser.add_argument("--level",  type=int, choices=[1,2,3,4,5],
                        help="Run one level (1-5)")
    parser.add_argument("--all",    action="store_true",
                        help="Run all 25 queries")
    parser.add_argument("--smoke",  action="store_true",
                        help="Quick test — first query of each level only")
    parser.add_argument("--id",     type=str,
                        help="Run one specific query e.g. L3Q1")
    parser.add_argument("--answer",  action="store_true",
                        help="Call LLM and show answers (uses .env provider)")
    parser.add_argument("--agentic", action="store_true",
                        help="Use ReAct agentic loop instead of standard retrieval (L4-L5 only)")
    args = parser.parse_args()

    print(f"\n{SEP2}")
    print(f"{BLD}  RoboLearn AI — Benchmark Runner{RST}")
    print(f"  25 queries · 5 levels · FAISS + Neo4j + optional LLM")
    print(SEP2)

    # ── Load systems ──────────────────────────────────────────────────────────
    print(f"\n  Loading FAISS index...")
    try:
        vs = VectorStore.load()
        print(f"  {GRN}FAISS loaded — {vs.index.ntotal} vectors{RST}")
    except Exception as e:
        print(f"  {RED}FAISS not found: {e}{RST}")
        print(f"  Run: python pipeline.py --ingest-only")
        sys.exit(1)

    print(f"  Connecting to Neo4j...")
    try:
        kg = KGRetriever()
        stats = kg.summarize_graph()
        print(f"  {GRN}Neo4j: {stats['nodes']} nodes · {stats['relations']} relations{RST}")
    except Exception as e:
        print(f"  {YLW}Neo4j offline ({e}) — graph retrieval disabled{RST}")
        from unittest.mock import MagicMock
        kg = MagicMock()
        kg.search.return_value = []

    print(f"  Loading BM25 index...")
    try:
        bm25 = BM25Store.load(vs.chunks)
        print(f"  {GRN}BM25 loaded — {len(bm25.chunks)} docs{RST}")
    except Exception as e:
        print(f"  {YLW}BM25 not loaded ({e}) — keyword retrieval disabled{RST}")
        bm25 = None
    retriever = HybridRetriever(vs, kg, bm25_store=bm25)

    generator = None
    if args.answer:
        print(f"  Loading LLM...")
        from generation.answer_generator import AnswerGenerator
        from config.settings import settings
        generator = AnswerGenerator()
        print(f"  {GRN}LLM: {settings.llm_provider} / {settings.llm_model}{RST}")

    if args.agentic:
        from retrieval.agent import ReactAgent
        print(f"  Loading agentic ReAct loop...")
        _agent = ReactAgent(vs, kg)
        run_query._agentic = True
        run_query._agent   = _agent
        print(f"  {GRN}Agentic mode: enabled (max 4 iterations){RST}")
    else:
        run_query._agentic = False

    # ── Select queries ────────────────────────────────────────────────────────
    if args.id:
        queries = load_queries(query_id=args.id)
    elif args.level:
        queries = load_queries(level=args.level)
    elif args.smoke:
        all_q = load_queries()
        queries = [next(q for q in all_q if q["level"] == lvl) for lvl in [1,2,3,4,5]]
    elif args.all:
        queries = load_queries()
    else:
        parser.print_help()
        print(f"\n  {YLW}Tip: run --smoke for a quick 5-query test across all levels{RST}")
        print(f"  {YLW}     run --all  for the full 25-query benchmark{RST}\n")
        try:
            kg.close()
        except Exception:
            pass
        return

    # ── Run ───────────────────────────────────────────────────────────────────
    all_scores = []
    current_level = None

    for q in queries:
        if q["level"] != current_level:
            current_level = q["level"]
            level_qs = [x for x in queries if x["level"] == current_level]
            print_level_header(current_level, level_qs)

        scores = run_query(q, retriever, generator)
        scores["id"]    = q["id"]
        scores["level"] = q["level"]
        all_scores.append(scores)

    print_summary(all_scores)

    try:
        kg.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()