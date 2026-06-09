"""
tests/query_tester.py
Runs queries at 4 complexity levels and shows exactly what
vector DB and knowledge graph retrieved before the LLM sees it.

Usage:
    python tests/query_tester.py                        # all 4 levels, retrieval only
    python tests/query_tester.py --level 3              # only level 3
    python tests/query_tester.py --level 4 --answer     # level 4 + LLM answer
    python tests/query_tester.py --query "your question" --answer
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from embeddings.vector_store import VectorStore
from graph.kg_retriever import KGRetriever
from retrieval.hybrid_retriever import HybridRetriever
from generation.answer_generator import AnswerGenerator

QUERY_LEVELS = {
    1: {
        "label": "Level 1 — Single concept (one word)",
        "description": "Tests if FAISS finds relevant chunks for bare terms. Graph rarely fires.",
        "queries": ["Arduino", "ROS", "PWM", "sensor"],
    },
    2: {
        "label": "Level 2 — Simple question (one relationship)",
        "description": "Tests single-hop graph retrieval + relevant chunk scoring.",
        "queries": [
            "what is ROS",
            "what does Arduino do",
            "what is PWM used for",
            "how does a sensor work",
        ],
    },
    3: {
        "label": "Level 3 — Multi-entity (cross-relationship)",
        "description": "Tests graph traversal across 2 nodes + chunk fusion boost.",
        "queries": [
            "how does ROS communicate with Arduino",
            "relationship between Arduino and DC motor",
            "how does a sensor send data to Arduino",
            "explain ROS nodes and topics",
        ],
    },
    4: {
        "label": "Level 4 — Multi-hop reasoning (full chain)",
        "description": "Tests multi-hop graph traversal + complex chunk retrieval.",
        "queries": [
            "trace the full data flow from sensor to motor in a ROS system",
            "what happens when ROS sends a command to move a robot arm",
            "how are sensors microcontrollers and actuators connected",
            "explain the complete chain from ROS master to motor output",
        ],
    },
}

SEP  = "─" * 68
SEP2 = "═" * 68


def print_retrieval(query: str, retriever: HybridRetriever):
    print(f"\n  Query: \"{query}\"")
    print(f"  {SEP}")

    context = retriever.retrieve(query)

    # ── Vector results ────────────────────────────────────────────
    print(f"\n  📦 VECTOR  ({len(context.chunks)} chunks retrieved)")
    for i, (chunk, score) in enumerate(context.chunks, 1):
        print(f"\n    [{i}] score={score:.4f}  |  {chunk.source_file}  |  page {chunk.page_number}")
        print(f"        chunk_id : {chunk.chunk_id}")
        preview = chunk.text[:220].replace("\n", " ").strip()
        print(f"        preview  : {preview}...")

    if not context.chunks:
        print("    (none)")

    # ── Graph results ─────────────────────────────────────────────
    print(f"\n  🔗 GRAPH   ({len(context.triples)} triples retrieved)")
    for t in context.triples:
        print(f"    {t.to_string()}")
    if not context.triples:
        print("    (none — Neo4j offline or KG not built)")

    return context


def run_level(level: int, retriever, generator, show_answer: bool):
    info = QUERY_LEVELS[level]
    print(f"\n{SEP2}")
    print(f"  {info['label']}")
    print(f"  {info['description']}")
    print(SEP2)

    for query in info["queries"]:
        context = print_retrieval(query, retriever)

        if show_answer:
            print(f"\n  💬 LLM ANSWER  {SEP}")
            try:
                ans = generator.generate(query, context)
                for line in ans.answer.split("\n"):
                    print(f"  {line}")
            except Exception as e:
                print(f"  ⚠️  {e}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level",  type=int, choices=[1,2,3,4])
    parser.add_argument("--query",  type=str)
    parser.add_argument("--answer", action="store_true")
    args = parser.parse_args()

    print("\nLoading vector store...")
    vs = VectorStore.load()
    print("Connecting to Neo4j...")
    try:
        kg = KGRetriever()
    except Exception as e:
        print(f"  ⚠️  Neo4j offline: {e}")
        from unittest.mock import MagicMock
        kg = MagicMock()
        kg.search.return_value = []

    retriever = HybridRetriever(vs, kg)
    generator = AnswerGenerator() if args.answer else None

    if args.query:
        print(f"\n{SEP2}\n  Custom query\n{SEP2}")
        context = print_retrieval(args.query, retriever)
        if args.answer and generator:
            print(f"\n  💬 LLM ANSWER  {SEP}")
            try:
                ans = generator.generate(args.query, context)
                for line in ans.answer.split("\n"):
                    print(f"  {line}")
            except Exception as e:
                print(f"  ⚠️  {e}")
    elif args.level:
        run_level(args.level, retriever, generator, args.answer)
    else:
        for lvl in [1, 2, 3, 4]:
            run_level(lvl, retriever, generator, args.answer)

    try:
        kg.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
