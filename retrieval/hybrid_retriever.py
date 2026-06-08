"""
retrieval/hybrid_retriever.py
Fuses results from the FAISS vector store and the Neo4j knowledge graph
into a single ranked context block for the LLM prompt.

Fusion strategy
───────────────
  final_score(chunk) = α × vector_score  +  β × graph_boost
  where graph_boost rewards chunks that mention graph-retrieved entities.

α = settings.vector_weight  (default 0.6)
β = settings.graph_weight   (default 0.4)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

from loguru import logger

from config.settings import settings
from embeddings.vector_store import VectorStore
from graph.kg_retriever import KGRetriever, GraphTriple
from ingestion.document_loader import DocumentChunk


@dataclass
class RetrievedContext:
    """Everything the LLM prompt builder needs."""
    query: str
    chunks: List[Tuple[DocumentChunk, float]]   # (chunk, fused_score)
    triples: List[GraphTriple]

    def to_prompt_block(self) -> str:
        """Render a structured context block for the system/user prompt."""
        lines = ["## Retrieved document context\n"]
        for i, (chunk, score) in enumerate(self.chunks, 1):
            lines.append(
                f"[{i}] (score={score:.3f}, source={chunk.source_file})\n"
                f"{chunk.text.strip()}\n"
            )

        if self.triples:
            lines.append("\n## Knowledge graph relationships\n")
            for t in self.triples:
                lines.append(f"  • {t.to_string()}")

        return "\n".join(lines)


class HybridRetriever:
    """
    Orchestrates vector + graph retrieval and returns a RetrievedContext.

    Usage:
        retriever = HybridRetriever(vector_store, kg_retriever)
        context   = retriever.retrieve("How does ROS talk to Arduino?")
        prompt    = context.to_prompt_block()
    """

    def __init__(self, vector_store: VectorStore, kg_retriever: KGRetriever):
        self.vs = vector_store
        self.kg = kg_retriever

    def retrieve(self, query: str) -> RetrievedContext:
        logger.debug(f"Hybrid retrieval for: {query!r}")

        # ── 1. Vector retrieval ──────────────────────────────────────────────
        vec_results: List[Tuple[DocumentChunk, float]] = self.vs.search(
            query, top_k=settings.top_k_vector
        )

        # ── 2. Graph retrieval ──────────────────────────────────────────────
        triples: List[GraphTriple] = self.kg.search(
            query, top_k=settings.top_k_graph
        )

        # ── 3. Graph-boost: reward chunks mentioning graph entities ─────────
        graph_entities = {
            name.lower()
            for t in triples
            for name in [t.source, t.target]
            if name is not None
        }

        def graph_boost(chunk: DocumentChunk) -> float:
            text_lower = chunk.text.lower()
            hits = sum(1 for e in graph_entities if e in text_lower)
            return min(hits / max(len(graph_entities), 1), 1.0)

        fused: List[Tuple[DocumentChunk, float]] = []
        for chunk, vscore in vec_results:
            boost = graph_boost(chunk)
            combined = (
                settings.vector_weight * vscore
                + settings.graph_weight * boost
            )
            fused.append((chunk, combined))

        # Sort by combined score descending
        fused.sort(key=lambda x: x[1], reverse=True)

        logger.debug(
            f"  → {len(fused)} chunks, {len(triples)} triples after fusion"
        )
        return RetrievedContext(query=query, chunks=fused, triples=triples)