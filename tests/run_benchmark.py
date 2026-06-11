"""
tests/run_benchmark.py
Runs the 25-query benchmark against the live system and scores results.

Usage:
    python tests/run_benchmark.py --level 1           # run level 1 only
    python tests/run_benchmark.py --level 3 --answer  # retrieval + LLM answer
    python tests/run_benchmark.py --all               # all 25 queries
    python tests/run_benchmark.py --id L3Q1           # single query by ID
"""
import argparse
import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from embeddings.vector_store import VectorStore
from graph.kg_retriever import KGRetriever
from retrieval.hybrid_retriever import HybridRetriever
from generation.answer_generator import AnswerGenerator

GRN = "\033[92m"; BLU = "\033[94m"; YLW = "\033[93m"
CYN = "\033[96m"; RED = "\033[91m"; GRY = "\033[90m"
BLD = "\033[1m";  RST = "\033[0m"
SEP = GRY + "─" * 70 + RST

BENCHMARK_FILE = os.path.join(os.path.dirname(__file__), "benchmark_queries.json")


def load_benchmark():
    with open(BENCHMARK_FILE) as f:
        return json.load(f)


def score_retrieval(result, query_meta):
    """Score how well retrieval matched expected entities and sources."""
    scores = {}

    # Source match — did we retrieve from the right PDF?
    expected_sources = query_meta["expected_source"]
    if isinstance(expected_sources, str):
        expected_sources = [expected_sources]
    retrieved_sources = set(c.source_file for c, _ in result.chunks)
    source_hits = sum(1 for s in expected_sources if any(s in r for r in retrieved_sources))
    scores["source_recall"] = source_hits / max(len(expected_sources), 1)

    # Entity match — did graph retrieve expected entities?
    expected_entities = [e.lower() for e in query_meta["expected_entities"]]
    graph_text = " ".join(
        f"{t.source} {t.target}".lower() for t in result.triples
    )
    chunk_text = " ".join(c.text.lower() for c, _ in result.chunks)
    entity_hits = sum(
        1 for e in expected_entities
        if e in graph_text or e in chunk_text
    )
    scores["entity_recall"] = entity_hits / max(len(expected_entities), 1)

    # Graph fired — did we get any triples?
    scores["graph_fired"] = len(result.triples) > 0
    scores["triple_count"] = len(result.triples)
    scores["chunk_count"] = len(result.chunks)
    scores["top_score"] = result.chunks[0][1] if result.chunks else 0.0

    return scores


def run_query(query_meta, retriever, generator=None):
    """Run one benchmark query and print full diagnostics."""
    qid = query_meta["id"]
    query = query_meta["query"]
    level = query_meta.get("level", "?")

    print(f"\n  {BLD}{qid}{RST}  [{query_meta['category']}]")
    print(f"  {BLD}Query:{RST} {query}")
    print(f"  {SEP}")

    t0 = time.time()
    context = retriever.retrieve(query)
    ms = (time.time() - t0) * 1000

    scores = score_retrieval(context, query_meta)

    # Vector results
    print(f"\n  {BLU}📦 VECTOR{RST} ({scores['chunk_count']} chunks · {ms:.0f}ms · top_score={scores['top_score']:.4f})")
    for i, (chunk, score) in enumerate(context.chunks, 1):
        print(f"    [{i}] {score:.4f} | {chunk.source_file} | page {chunk.page_number}")
        preview = chunk.text[:150].replace("\n", " ").strip()
        print(f"         {GRY}{preview}...{RST}")

    # Graph results
    print(f"\n  {CYN}🔗 GRAPH{RST} ({scores['triple_count']} triples · fired={scores['graph_fired']})")
    for t in context.triples:
        print(f"    {t.to_string()}")
    if not context.triples:
        print(f"    {GRY}(none){RST}")

    # Scores
    print(f"\n  {YLW}📊 RETRIEVAL SCORES{RST}")
    print(f"    source_recall  : {scores['source_recall']:.2f}  (expected: {query_meta['expected_source']})")
    print(f"    entity_recall  : {scores['entity_recall']:.2f}  (expected: {query_meta['expected_entities']})")
    print(f"    graph_fired    : {GRN if scores['graph_fired'] else RED}{scores['graph_fired']}{RST}")
    print(f"    retrieval_type : {query_meta['retrieval_type']}")

    # Expected answer
    print(f"\n  {GRY}📖 GROUND TRUTH:{RST}")
    gt = query_meta["ground_truth"]
    for line in gt.split(". "):
        print(f"    {GRY}{line.strip()}.{RST}" if line.strip() else "")

    # LLM answer
    if generator:
        print(f"\n  {GRN}💬 LLM ANSWER{RST}")
        try:
            ans = generator.generate(query, context)
            for line in ans.answer.split("\n"):
                print(f"  {line}")
            print(f"\n  {YLW}📄 SOURCES:{RST}")
            for sp in ans.source_pages:
                print(f"    [{sp['n']}] {sp['file']} — page {sp['page']}")
        except Exception as e:
            print(f"  {RED}LLM error: {e}{RST}")

    print(f"\n  {GRY}tags: {query_meta['metadata_tags']}{RST}")
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=[1,2,3,4,5])
    parser.add_argument("--all",   action="store_true")
    parser.add_argument("--id",    type=str, help="e.g. L3Q1")
    parser.add_argument("--answer",action="store_true")
    args = parser.parse_args()

    bench = load_benchmark()

    print(f"\n{BLD}{'═'*70}")
    print(f"  RoboLearn AI — Benchmark Runner")
    print(f"  Sources: {len(bench['benchmark_metadata']['sources'])} PDFs · {bench['benchmark_metadata']['total_queries']} queries · 5 levels")
    print(f"{'═'*70}{RST}")

    print("\n  Loading FAISS...")
    vs = VectorStore.load()
    print("  Connecting to Neo4j...")
    try:
        kg = KGRetriever()
        stats = kg.summarize_graph()
        print(f"  {GRN}Neo4j: {stats['nodes']} nodes, {stats['relations']} relations{RST}")
    except Exception as e:
        print(f"  {YLW}Neo4j offline — graph retrieval disabled{RST}")
        from unittest.mock import MagicMock
        kg = MagicMock(); kg.search.return_value = []

    retriever = HybridRetriever(vs, kg)
    generator = AnswerGenerator() if args.answer else None

    # Collect queries to run
    all_queries = []
    for lvl_key, lvl_data in bench["levels"].items():
        lvl_num = int(lvl_key.split("_")[1])
        for q in lvl_data["queries"]:
            q["level"] = lvl_num
            all_queries.append(q)

    if args.id:
        queries = [q for q in all_queries if q["id"] == args.id]
        if not queries:
            print(f"  {RED}ID not found: {args.id}{RST}")
            print(f"  Valid IDs: {[q['id'] for q in all_queries]}")
            sys.exit(1)
    elif args.level:
        queries = [q for q in all_queries if q["level"] == args.level]
    elif args.all:
        queries = all_queries
    else:
        parser.print_help()
        print(f"\n  {YLW}Tip: run --all to run all 25 queries (retrieval only, no Groq calls){RST}")
        return

    # Run and collect scores
    all_scores = []
    current_level = None
    for q in queries:
        if q["level"] != current_level:
            current_level = q["level"]
            lvl_info = bench["levels"][f"level_{current_level}"]
            print(f"\n{'═'*70}")
            print(f"{BLD}  Level {current_level}: {lvl_info['label']}{RST}")
            print(f"  {GRY}{lvl_info['description']}{RST}")
            print('═'*70)

        scores = run_query(q, retriever, generator)
        scores["id"] = q["id"]
        scores["level"] = q["level"]
        all_scores.append(scores)

    # Summary
    print(f"\n{'═'*70}")
    print(f"{BLD}  BENCHMARK SUMMARY{RST}")
    print('═'*70)
    avg_src  = sum(s["source_recall"]  for s in all_scores) / len(all_scores)
    avg_ent  = sum(s["entity_recall"]  for s in all_scores) / len(all_scores)
    graph_ct = sum(1 for s in all_scores if s["graph_fired"])
    print(f"  Queries run        : {len(all_scores)}")
    print(f"  Avg source recall  : {GRN}{avg_src:.2f}{RST}")
    print(f"  Avg entity recall  : {GRN}{avg_ent:.2f}{RST}")
    print(f"  Graph fired        : {GRN}{graph_ct}/{len(all_scores)}{RST}")
    print(f"  Avg top-1 score    : {sum(s['top_score'] for s in all_scores)/len(all_scores):.4f}")

    try:
        kg.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
