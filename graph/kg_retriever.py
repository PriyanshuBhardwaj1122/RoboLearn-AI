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

        # Sort candidates — technical terms first so their triples
        # are collected before hub nodes like "Arduino UNO R3"
        candidate_set_lower = {c.lower() for c in candidates}
        technical = [c for c in candidates if c.lower() in self.TECHNICAL_TERMS]
        hub       = [c for c in candidates if c.lower() in self.HUB_TERMS]
        general   = [c for c in candidates if c.lower() not in self.TECHNICAL_TERMS and c.lower() not in self.HUB_TERMS]
        ordered_candidates = technical + general + hub

        # Limit per-candidate to 5 results so high-degree nodes
        # (like Arduino UNO R3 with 500+ edges) don't crowd out
        # specific technical term results (like I2C, SDA, SCL)
        triples = []
        with self.driver.session() as session:
            for candidate in ordered_candidates:
                candidate_triples = self._fetch_triples(session, candidate, hops=3)
                # Deduplicate within this candidate's results before adding
                seen_local = set()
                for t in candidate_triples:
                    key = (t.source, t.relation, t.target)
                    if key not in seen_local:
                        seen_local.add(key)
                        triples.append(t)
                    if len(seen_local) >= 5:
                        break

        # Deduplicate
        seen = set()
        unique = []
        for t in triples:
            key = (t.source, t.relation, t.target)
            if key not in seen:
                seen.add(key)
                unique.append(t)

        # Rank by candidate relevance score:
        # Triples that directly mention a query candidate score higher
        # This prevents high-degree hub nodes (like Arduino UNO R3) from
        # crowding out specific triples (like I2C pins)
        from collections import Counter
        candidate_set = {c.lower() for c in candidates}

        def relevance(t: GraphTriple) -> int:
            score = 0
            src_l = t.source.lower()
            tgt_l = t.target.lower()
            # Direct candidate match in source or target — highest priority
            for c in candidate_set:
                if c in src_l: score += 10
                if c in tgt_l: score += 10
            # Secondary: degree centrality (tiebreaker only)
            return score

        unique.sort(key=relevance, reverse=True)

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

    # Technical terms that must always be kept as candidates
    TECHNICAL_TERMS = {
        "i2c", "spi", "uart", "pwm",
        "sda", "scl", "mosi", "miso", "sck", "gpio", "adc", "dac",
        "rosserial", "tcp", "udp", "mqtt", "pid",
        "a4", "a5", "d9", "d10", "d11", "a0", "a1", "a2",
    }
    # Hub nodes — high-degree entities that should NOT get priority
    # They get processed last so specific terms fill top slots first
    HUB_TERMS = {"arduino", "ros", "atmel", "atmega", "usb", "can"}

    def _tokenize_query(self, query: str) -> List[str]:
        """
        Extract candidate entity terms from query.
        Always preserves technical terms (I2C, SDA, PWM etc.)
        regardless of stop word rules.
        """
        stop = {"how", "does", "what", "is", "the", "a", "an", "and", "or",
                "to", "in", "of", "for", "with", "can", "do", "i", "me",
                "give", "using", "use", "used", "about", "explain"}
        words = re.findall(r"[A-Za-z0-9_\-\.]+", query)
        terms = set()
        for i, w in enumerate(words):
            # Always keep technical terms regardless of stop word list
            if w.lower() in self.TECHNICAL_TERMS:
                terms.add(w)
            elif w.lower() not in stop and len(w) > 1:
                terms.add(w)
            # Bigrams
            if i < len(words) - 1:
                bigram = f"{words[i]} {words[i+1]}"
                if not all(t.lower() in stop for t in [words[i], words[i+1]]):
                    terms.add(bigram)
        return list(terms)

    def _fetch_triples(self, session, candidate: str, hops: int = 3) -> List[GraphTriple]:
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
        """Fast 1-hop retrieval — indexed node name lookup."""
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
                triples.append(GraphTriple(
                    source=row["source"],
                    relation=row["relation"],
                    target=row["target"],
                ))
        except Exception as e:
            logger.warning(f"Graph retrieval failed for '{candidate}': {e}")
        return triples