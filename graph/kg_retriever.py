"""
graph/kg_retriever.py
Retrieves relevant subgraph context from Neo4j for a user query.

Two strategies:
  1. Entity-match  — find nodes whose names appear in the query text.
  2. Multi-hop     — expand 1-2 hops from matched nodes via Cypher.

Returns a list of human-readable triple strings for prompt injection.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List

from neo4j import GraphDatabase
from loguru import logger

from config.settings import settings


@dataclass
class GraphTriple:
    source: str
    relation: str
    target: str
    score: float = 1.0   # simple presence score; extend with embedding sim later

    def to_string(self) -> str:
        return f"{self.source} --[{self.relation}]--> {self.target}"


class KGRetriever:
    """
    Queries Neo4j for triples relevant to *query*.

    Usage:
        retriever = KGRetriever()
        triples = retriever.search("How does ROS communicate with Arduino?")
    """

    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    def close(self) -> None:
        self.driver.close()

    # ── Public ──────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int | None = None) -> List[GraphTriple]:
        """
        1. Extract candidate entity names from the query.
        2. Find matching nodes in Neo4j.
        3. Expand 1–2 hops and collect triples.
        4. Return top_k unique triples ranked by centrality (degree).
        """
        top_k = top_k or settings.top_k_graph
        candidates = self._tokenize_query(query)
        if not candidates:
            return []

        triples = []
        with self.driver.session() as session:
            for candidate in candidates:
                triples.extend(self._fetch_triples(session, candidate, hops=2))

        # Deduplicate
        seen = set()
        unique = []
        for t in triples:
            key = (t.source, t.relation, t.target)
            if key not in seen:
                seen.add(key)
                unique.append(t)

        # Rank by degree (nodes that appear more often are more central)
        from collections import Counter
        degree = Counter()
        for t in unique:
            degree[t.source] += 1
            degree[t.target] += 1
        unique.sort(key=lambda t: degree[t.source] + degree[t.target], reverse=True)

        return unique[:top_k]

    def get_entity_context(self, entity_name: str) -> List[GraphTriple]:
        """All direct relationships for a single entity — useful for deep dives."""
        with self.driver.session() as session:
            return self._fetch_triples(session, entity_name, hops=1)

    def summarize_graph(self) -> dict:
        """Quick stats for the UI health panel."""
        with self.driver.session() as session:
            n_nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            n_rels  = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        return {"nodes": n_nodes, "relations": n_rels}

    # ── Cypher helpers ──────────────────────────────────────────────────────

    def _tokenize_query(self, query: str) -> List[str]:
        """
        Extract multi-word technical terms from the query.
        Keeps noun phrases of 1-3 tokens; filters stop words.
        """
        stop = {"how", "does", "what", "is", "the", "a", "an", "and", "or",
                "to", "in", "of", "for", "with", "can", "do", "i", "me", "give"}
        words = re.findall(r"[A-Za-z0-9_\-\.]+", query)
        # Include individual words and bigrams
        terms = set()
        for i, w in enumerate(words):
            if w.lower() not in stop and len(w) > 1:
                terms.add(w)
            if i < len(words) - 1:
                bigram = f"{words[i]} {words[i+1]}"
                if not all(t.lower() in stop for t in [words[i], words[i+1]]):
                    terms.add(bigram)
        return list(terms)

    def _fetch_triples(self, session, candidate: str, hops: int = 2) -> List[GraphTriple]:
        """Fuzzy-match candidate against node names, then expand *hops* away."""
        # Case-insensitive partial match
        cypher = """
        MATCH (n)
        WHERE toLower(n.name) CONTAINS toLower($candidate)
        WITH n LIMIT 5
        CALL apoc.path.subgraphAll(n, {maxLevel: $hops})
        YIELD relationships
        UNWIND relationships AS rel
        RETURN
            startNode(rel).name AS source,
            type(rel)           AS relation,
            endNode(rel).name   AS target
        LIMIT 50
        """
        triples = []
        try:
            results = session.run(cypher, candidate=candidate, hops=hops)
            for row in results:
                triples.append(
                    GraphTriple(
                        source=row["source"],
                        relation=row["relation"],
                        target=row["target"],
                    )
                )
        except Exception:
            # Fallback: APOC may not be installed — use manual 1-hop Cypher
            triples = self._fetch_triples_no_apoc(session, candidate)
        return triples

    def _fetch_triples_no_apoc(self, session, candidate: str) -> List[GraphTriple]:
        """Fallback 1-hop retrieval without APOC."""
        cypher = """
        MATCH (a)-[r]->(b)
        WHERE toLower(a.name) CONTAINS toLower($candidate)
           OR toLower(b.name) CONTAINS toLower($candidate)
        RETURN a.name AS source, type(r) AS relation, b.name AS target
        LIMIT 30
        """
        triples = []
        try:
            for row in session.run(cypher, candidate=candidate):
                triples.append(
                    GraphTriple(
                        source=row["source"],
                        relation=row["relation"],
                        target=row["target"],
                    )
                )
        except Exception as e:
            logger.warning(f"Graph retrieval failed for '{candidate}': {e}")
        return triples
