"""
retrieval/hybrid_retriever.py
Three-way hybrid retrieval with Reciprocal Rank Fusion (RRF):
  vector  (FAISS)  — semantic similarity
  sparse  (BM25)   — keyword / exact token match (great for tables)
  graph   (Neo4j)  — entity-relation traversal

Fusion formula (RRF):
  RRF_score = 1/(k + rank_faiss)
             + 1/(k + rank_bm25)
             + graph_weight × 1/(k + rank_graph_boost)

  k=60 (standard), graph_weight=0.5 (graph is supplementary signal)

Why RRF over weighted score fusion:
  Raw scores from FAISS, BM25, and graph are on different scales and
  distributions — adding them directly is meaningless. RRF uses only
  rank positions, making the fusion scale-invariant and more stable.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

from loguru import logger

from config.settings import settings
from embeddings.vector_store import VectorStore
from embeddings.bm25_store import BM25Store
from graph.kg_retriever import KGRetriever, GraphTriple
from ingestion.document_loader import DocumentChunk
from generation.llm_client import LLMClient


@dataclass
class RetrievedContext:
    query:   str
    chunks:  List[Tuple[DocumentChunk, float]] = field(default_factory=list)
    triples: List[GraphTriple]                 = field(default_factory=list)


class HybridRetriever:

    # RRF constant — standard value, prevents top-rank dominance
    RRF_K = 60

    # Graph contributes as supplementary signal, not primary retrieval
    GRAPH_RRF_WEIGHT = 0.5

    def __init__(
        self,
        vector_store: VectorStore,
        kg_retriever:  KGRetriever,
        bm25_store:    BM25Store | None = None,
        reranker=None,
    ):
        self.vs       = vector_store
        self.kg       = kg_retriever
        self.bm25     = bm25_store
        self.reranker = reranker
        self.llm      = LLMClient()

    # ── Source hint mapping ──────────────────────────────────────────────────

    SOURCE_HINTS = {
        "rosserial":    ["rosserial - ROS Wiki", "Programming"],
        "ros":          ["Programming"],
        "l298n":        ["L298N"],
        "l298":         ["L298N"],
        "motor driver": ["L298N"],
        "h-bridge":     ["L298N"],
        "i2c":          ["I2C", "ARDUINO"],
        "kuka":         ["KUKA"],
        "atmega":       ["Atmel", "ARDUINO"],
        "timer":        ["Atmel", "ARDUINO"],
        "adc":          ["Atmel", "ARDUINO"],
        "arduino":      ["ARDUINO"],
        "automation":   ["ROBOTICS"],
        "actuator":     ["ROBOTICS"],
        "manipulator":  ["ROBOTICS"],
        "pwm":          ["ARDUINO", "Atmel"],
        "uart":         ["ARDUINO"],
        "spi":          ["ARDUINO"],
        "sda":          ["ARDUINO"],
        "scl":          ["ARDUINO"],
        "vin":          ["ARDUINO"],
        "ldo":          ["ARDUINO"],
        "nodehandle":   ["rosserial - ROS Wiki"],
        "publisher":    ["rosserial - ROS Wiki", "Programming"],
        "subscriber":   ["rosserial - ROS Wiki", "Programming"],
        "cmd_vel":      ["Programming"],
        "topic":        ["Programming"],
        "sensor":       ["ROBOTICS", "ARDUINO"],
        "robot":        ["ROBOTICS"],
    }

    def _detect_source_hints(self, query: str) -> set:
        q = query.lower()
        hints = set()
        for term, sources in self.SOURCE_HINTS.items():
            if term in q:
                if isinstance(sources, list):
                    hints.update(sources)
                else:
                    hints.add(sources)
        logger.debug(f"  Source hints detected: {hints}")
        return hints

    def _expand_query(self, query: str) -> List[str]:
        """
        For complex multi-hop queries, decompose into 2-3 focused sub-queries.
        Simple queries (under 10 words) are returned as-is.
        Uses temperature=0.0 for deterministic output — eliminates run-to-run variance.
        """
        word_count = len(query.split())
        if word_count < 10:
            return [query]

        prompt = (
            "Break this question into 2-3 specific sub-questions that together "
            "cover the full answer. Each sub-question should target a different "
            "document or concept. Return only the sub-questions, one per line, "
            "no numbering, no extra text.\n\n"
            f"Question: {query}"
        )
        try:
            from openai import OpenAI
            client = OpenAI(api_key=settings.openai_api_key)
            resp = client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            raw = resp.choices[0].message.content
            sub_queries = [
                q.strip() for q in raw.strip().split('\n')
                if q.strip() and len(q.strip()) > 10
            ][:3]
            if not sub_queries:
                return [query]
            logger.debug(
                f"  Query expanded into {len(sub_queries)} sub-queries: {sub_queries}"
            )
            return [query] + sub_queries
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return [query]

    # ── RRF fusion ───────────────────────────────────────────────────────────

    def _rrf_score(self, rank: int, weight: float = 1.0) -> float:
        """Reciprocal Rank Fusion score for a single ranklist."""
        return weight / (self.RRF_K + rank)

    def _fuse_rrf(
        self,
        vec_results:   List[Tuple[DocumentChunk, float]],
        bm25_results:  Dict[str, Tuple[DocumentChunk, float]],
        graph_ranked:  List[str],
        all_chunks:    Dict[str, DocumentChunk],
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Combine three ranked lists using Reciprocal Rank Fusion.

        vec_results   — FAISS results already sorted by score (rank 1 = best)
        bm25_results  — BM25 results dict {chunk_id: (chunk, score)}
        graph_ranked  — chunk_ids ranked by graph entity hit count
        all_chunks    — all candidate chunks {chunk_id: chunk}

        Returns list of (chunk, rrf_score) sorted descending.
        """
        rrf: Dict[str, float] = {}

        # FAISS ranks (already sorted best-first)
        for rank, (chunk, _) in enumerate(vec_results, start=1):
            cid = chunk.chunk_id
            rrf[cid] = rrf.get(cid, 0.0) + self._rrf_score(rank)

        # BM25 ranks — sort by score descending to get ranks
        bm25_sorted = sorted(
            bm25_results.values(), key=lambda x: x[1], reverse=True
        )
        for rank, (chunk, _) in enumerate(bm25_sorted, start=1):
            cid = chunk.chunk_id
            rrf[cid] = rrf.get(cid, 0.0) + self._rrf_score(rank)

        # Graph boost ranks (weighted at 0.5 — supplementary signal)
        for rank, cid in enumerate(graph_ranked, start=1):
            rrf[cid] = rrf.get(cid, 0.0) + self._rrf_score(
                rank, weight=self.GRAPH_RRF_WEIGHT
            )

        # Build sorted result list
        fused = []
        for cid, score in rrf.items():
            if cid in all_chunks:
                fused.append((all_chunks[cid], score))

        fused.sort(key=lambda x: x[1], reverse=True)
        return fused

    # ── Public ───────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> RetrievedContext:
        logger.debug(f"Hybrid retrieval for: {query!r}")

        top_k_vec   = getattr(settings, "top_k_vector", 8)
        top_k_graph = getattr(settings, "top_k_graph",  10)

        # ── 0. Query expansion ────────────────────────────────────────────
        queries = self._expand_query(query)

        # ── 1. Vector retrieval (FAISS) — run on all sub-queries ──────────
        vec_seen: Dict[str, Tuple[DocumentChunk, float]] = {}
        for q in queries:
            for chunk, score in self.vs.search(q, top_k=top_k_vec):
                cid = chunk.chunk_id
                if cid not in vec_seen or score > vec_seen[cid][1]:
                    vec_seen[cid] = (chunk, score)
        # Sort best-first so FAISS ranks are correct for RRF
        vec_results = sorted(
            vec_seen.values(), key=lambda x: x[1], reverse=True
        )

        # ── 2. BM25 retrieval — run on all sub-queries ────────────────────
        bm25_results: Dict[str, Tuple[DocumentChunk, float]] = {}
        if self.bm25:
            for q in queries:
                for chunk, score in self.bm25.search(q, top_k=top_k_vec * 2):
                    cid = chunk.chunk_id
                    if cid not in bm25_results or score > bm25_results[cid][1]:
                        bm25_results[cid] = (chunk, score)

        # ── 3. Graph retrieval (Neo4j) — original query only ──────────────
        triples: List[GraphTriple] = []
        try:
            triples = self.kg.search(query, top_k=top_k_graph)
        except Exception as e:
            logger.warning(f"Graph retrieval failed: {e}")

        # ── 4. Graph entity names → rank chunks by entity hit count ───────
        graph_entity_names = set()
        for t in triples:
            graph_entity_names.add(t.source.lower())
            graph_entity_names.add(t.target.lower())

        # Collect all candidates
        all_chunks: Dict[str, DocumentChunk] = {}
        for chunk, _ in vec_results:
            all_chunks[chunk.chunk_id] = chunk
        for cid, (chunk, _) in bm25_results.items():
            all_chunks[cid] = chunk

        # Rank all candidates by graph entity hit count (for graph RRF list)
        def graph_hits(chunk: DocumentChunk) -> int:
            if not graph_entity_names:
                return 0
            text_lower = chunk.text.lower()
            return sum(1 for e in graph_entity_names if e in text_lower)

        graph_ranked = sorted(
            all_chunks.keys(),
            key=lambda cid: graph_hits(all_chunks[cid]),
            reverse=True,
        )

        # ── 5. RRF fusion ─────────────────────────────────────────────────
        candidates = self._fuse_rrf(
            vec_results, bm25_results, graph_ranked, all_chunks
        )
        candidates = candidates[:top_k_vec * 2]

        logger.debug(
            f"  RRF fusion: {len(vec_results)} vec + "
            f"{len(bm25_results)} bm25 + "
            f"{len(graph_ranked)} graph → "
            f"{len(candidates)} candidates"
        )

        # ── 6. Source diversity ───────────────────────────────────────────
        hints = self._detect_source_hints(query)
        if hints:
            hinted, rest = [], []
            seen_hint_sources = set()
            for chunk, score in candidates:
                src = chunk.source_file
                matched_hint = any(h in src for h in hints)
                if matched_hint and src not in seen_hint_sources:
                    hinted.append((chunk, score))
                    seen_hint_sources.add(src)
                else:
                    rest.append((chunk, score))
            candidates = (hinted + rest)[:top_k_vec * 2]

        # ── 7. Parent lookup — swap children for parent chunks ────────────
        candidates = self._fetch_parents(candidates)

        # ── 8. Rerank (cross-encoder) ─────────────────────────────────────
        if self.reranker and len(candidates) > 1:
            top_chunks = self.reranker.rerank(
                query, candidates, top_k=top_k_vec, source_hints=hints
            )
        else:
            top_chunks = candidates[:top_k_vec]

        logger.debug(
            f"  → {len(top_chunks)} chunks, {len(triples)} triples "
            f"(from {len(queries)} queries, {len(all_chunks)} unique candidates)"
        )
        return RetrievedContext(query=query, chunks=top_chunks, triples=triples)

    # ── Parent-child retrieval ────────────────────────────────────────────────

    def _fetch_parents(
        self,
        chunks: List[Tuple[DocumentChunk, float]],
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Given child chunks from FAISS/BM25, return their parent chunks.
        Parents carry full section context — sent to LLM instead of children.
        Falls back to child if parent not found.
        """
        if not hasattr(self, "_parent_store"):
            try:
                from ingestion.document_loader import DocumentLoader
                loader = DocumentLoader()
                self._parent_store = loader.load_parents()
                logger.debug(
                    f"Loaded {len(self._parent_store)} parents into retriever cache"
                )
            except Exception as ex:
                logger.warning(f"Could not load parents: {ex}")
                self._parent_store = {}

        if not self._parent_store:
            return chunks

        seen_parents: set = set()
        result: List[Tuple[DocumentChunk, float]] = []

        for child, score in chunks:
            pid = child.parent_id
            if pid and pid in self._parent_store:
                if pid not in seen_parents:
                    seen_parents.add(pid)
                    result.append((self._parent_store[pid], score))
            else:
                if child.chunk_id not in seen_parents:
                    seen_parents.add(child.chunk_id)
                    result.append((child, score))

        return result