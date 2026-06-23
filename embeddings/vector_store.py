"""
embeddings/vector_store.py
Encodes DocumentChunks with a Sentence-Transformer model and stores /
queries a FAISS index for fast dense retrieval.

BGE models require a query prefix at search time but NOT at index time.
The prefix is applied automatically when the model name contains "bge".
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger

from config.settings import settings
from ingestion.document_loader import DocumentChunk


# BGE models need this prefix on queries (not on documents)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class VectorStore:
    """
    Wraps FAISS index + chunk lookup.  Supports incremental updates.

    Usage:
        store = VectorStore()
        store.build(chunks)          # first run
        results = store.search(query, top_k=5)

        # subsequent runs
        store = VectorStore.load()
        results = store.search(query)
    """

    def __init__(self, model_name: str | None = None):
        model_name = model_name or settings.embedding_model
        logger.info(f"Loading embedding model: {model_name}")
        self.model       = SentenceTransformer(model_name)
        self.model_name  = model_name
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: List[DocumentChunk] = []

        # BGE models need a query prefix — detect automatically
        self._use_bge_prefix = "bge" in model_name.lower()
        if self._use_bge_prefix:
            logger.info("BGE model detected — query prefix will be applied at search time")

    # ── Build ────────────────────────────────────────────────────────────────

    def build(self, chunks: List[DocumentChunk], batch_size: int = 64) -> None:
        """
        Encode all chunks and build an inner-product (cosine-equiv) FAISS index.
        Documents are encoded WITHOUT the BGE query prefix.
        """
        logger.info(f"Encoding {len(chunks)} chunks (batch={batch_size})…")
        texts = [c.text for c in chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,   # cosine via inner product
            convert_to_numpy=True,
        ).astype("float32")

        dim = embeddings.shape[1]
        self.index  = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.chunks = chunks

        logger.success(
            f"FAISS index built: {self.index.ntotal} vectors, dim={dim}"
        )

    # ── Persist ──────────────────────────────────────────────────────────────

    def save(self) -> None:
        settings.processed_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(settings.faiss_index_path) + ".bin")
        with open(str(settings.faiss_index_path) + "_meta.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        logger.success(f"Index saved → {settings.faiss_index_path}")

    @classmethod
    def load(cls, model_name: str | None = None) -> "VectorStore":
        store = cls(model_name)
        store.index = faiss.read_index(str(settings.faiss_index_path) + ".bin")
        with open(str(settings.faiss_index_path) + "_meta.pkl", "rb") as f:
            store.chunks = pickle.load(f)
        logger.info(f"Loaded FAISS index: {store.index.ntotal} vectors")
        return store

    # ── Search ───────────────────────────────────────────────────────────────

    def search(
        self, query: str, top_k: int | None = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Return (chunk, score) pairs sorted by relevance (highest first).
        BGE models: query is prefixed automatically for better retrieval quality.
        """
        top_k = top_k or settings.top_k_vector

        # Apply BGE query prefix if needed
        query_text = (
            f"{BGE_QUERY_PREFIX}{query}"
            if self._use_bge_prefix
            else query
        )

        q_vec = self.model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        scores, indices = self.index.search(q_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results