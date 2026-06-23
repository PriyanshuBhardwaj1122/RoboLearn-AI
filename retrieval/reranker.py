"""
retrieval/reranker.py
Cross-encoder reranker — reorders retrieved chunks by joint (query, chunk) score.
Much more accurate than cosine similarity alone because it reads both together.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
Size:  ~80MB — fast on M2, no GPU needed
"""
from __future__ import annotations
from typing import List, Tuple

from loguru import logger
from sentence_transformers import CrossEncoder

from ingestion.document_loader import DocumentChunk


class Reranker:

    MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self):
        logger.info(f"Loading reranker: {self.MODEL}")
        self.model = CrossEncoder(self.MODEL, max_length=512)
        logger.info("Reranker ready")

    def rerank(
        self,
        query: str,
        chunks: List[Tuple[DocumentChunk, float]],
        top_k: int = 5,
        source_hints=None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Rerank chunks using cross-encoder scores.
        Returns top_k chunks sorted by reranker score (normalised 0-1).
        """
        if not chunks:
            return []

        # Build (query, chunk_text) pairs
        pairs  = [(query, chunk.text[:512]) for chunk, _ in chunks]
        scores = self.model.predict(pairs)
        # Lift hinted-source scores before normalisation so they can't be buried
        if source_hints:
           raw_min = min(scores)
           raw_max = max(scores)
           raw_rng = raw_max - raw_min if raw_max > raw_min else 1.0
           floor = raw_min + 0.3 * raw_rng
           scores = [
              max(s, floor) if any(h in chunks[i][0].source_file for h in source_hints)
            else s
            for i, s in enumerate(scores)
            ]
        # Normalise to 0-1
        min_s = min(scores)
        max_s = max(scores)
        rng   = max_s - min_s if max_s > min_s else 1.0
        norm  = [(s - min_s) / rng for s in scores]

        # Zip with chunks and sort
        reranked = sorted(
            zip([c for c, _ in chunks], norm),
            key=lambda x: x[1],
            reverse=True,
        )

        logger.debug(
            f"Reranker: {len(chunks)} → top {top_k} | "
            f"top score={reranked[0][1]:.4f} | "
            f"bottom score={reranked[-1][1]:.4f}"
        )
        return list(reranked[:top_k])