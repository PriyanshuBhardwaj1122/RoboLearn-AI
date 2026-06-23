"""
embeddings/bm25_store.py
BM25 sparse retrieval — keyword-based search that complements FAISS.
Especially useful for table content (pinout pages, spec sheets) that
embeds poorly but contains exact query tokens.
"""
from __future__ import annotations
import json
import pickle
import re
from pathlib import Path
from typing import List, Tuple

from loguru import logger
from rank_bm25 import BM25Okapi

from config.settings import settings


class BM25Store:

    def __init__(self):
        self.bm25   = None
        self.chunks = []   # parallel list to bm25 corpus

    # ── Build ────────────────────────────────────────────────────────────────

    def build(self, chunks: list) -> None:
        """Build BM25 index from document chunks."""
        self.chunks = chunks
        corpus = [self._tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(corpus)
        logger.info(f"BM25 index built: {len(chunks)} documents")

    def save(self) -> None:
        path = Path(settings.processed_dir) / "bm25_index.pkl"
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunk_ids": [c.chunk_id for c in self.chunks]}, f)
        logger.info(f"BM25 index saved → {path}")

    @classmethod
    def load(cls, chunks: list) -> "BM25Store":
        """Load saved BM25 index and reattach chunks."""
        path = Path(settings.processed_dir) / "bm25_index.pkl"
        store = cls()
        if not path.exists():
            logger.warning("BM25 index not found — building from chunks")
            store.build(chunks)
            store.save()
            return store
        with open(path, "rb") as f:
            data = pickle.load(f)
        store.bm25   = data["bm25"]
        store.chunks = chunks   # reattach full chunk objects
        logger.info(f"BM25 index loaded: {len(chunks)} documents")
        return store

    # ── Search ───────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> List[Tuple]:
        """
        Returns list of (chunk, bm25_score) sorted by score descending.
        Scores are normalised to [0, 1].
        """
        if self.bm25 is None:
            return []
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # Normalise
        max_score = max(scores) if max(scores) > 0 else 1.0
        norm = [s / max_score for s in scores]

        # Top-k
        top_idx = sorted(range(len(norm)), key=lambda i: norm[i], reverse=True)[:top_k]
        return [(self.chunks[i], norm[i]) for i in top_idx if norm[i] > 0]

    # ── Tokenise ─────────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple whitespace + punctuation tokeniser.
        Preserves technical terms like A4/SDA, I2C, PWM, D9.
        """
        # Split on whitespace and most punctuation but keep / in pin names
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9/]*", text.lower())
        # Filter very short tokens except known technical ones
        keep = {"i2c", "spi", "pwm", "ros", "sda", "scl", "vin",
                "a4", "a5", "d9", "d10", "d11", "ss", "rx", "tx"}
        return [t for t in tokens if len(t) > 2 or t in keep]