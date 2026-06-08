"""
ingestion/document_loader.py
Loads PDFs, Word docs, and plain-text training materials, then splits
them into overlapping chunks ready for embedding and graph extraction.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List

import pdfplumber
from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from config.settings import settings


@dataclass
class DocumentChunk:
    chunk_id: str
    source_file: str
    page_number: int
    text: str
    metadata: dict = field(default_factory=dict)


class DocumentLoader:
    """
    Ingests raw documents from data/raw/, extracts text per page/section,
    and returns a flat list of DocumentChunk objects.
    """

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def load_directory(self, directory: Path | None = None) -> List[DocumentChunk]:
        """Load all supported documents from *directory* (default: data/raw/)."""
        directory = directory or settings.raw_dir
        chunks: List[DocumentChunk] = []
        for path in Path(directory).rglob("*"):
            if path.suffix.lower() == ".pdf":
                chunks.extend(self._load_pdf(path))
            elif path.suffix.lower() in {".docx", ".doc"}:
                chunks.extend(self._load_docx(path))
            elif path.suffix.lower() == ".txt":
                chunks.extend(self._load_txt(path))
        logger.info(f"Loaded {len(chunks)} chunks from {directory}")
        return chunks

    def save_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Persist chunks to disk as JSON for inspection / reuse."""
        settings.processed_dir.mkdir(parents=True, exist_ok=True)
        with open(settings.chunk_store_path, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in chunks], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(chunks)} chunks → {settings.chunk_store_path}")

    def load_chunks(self) -> List[DocumentChunk]:
        """Reload previously saved chunks."""
        with open(settings.chunk_store_path, encoding="utf-8") as f:
            data = json.load(f)
        return [DocumentChunk(**d) for d in data]

    # ── Private helpers ─────────────────────────────────────────────────────

    def _split(self, text: str, source: str, page: int) -> List[DocumentChunk]:
        pieces = self.splitter.split_text(text)
        return [
            DocumentChunk(
                chunk_id=f"{source}__p{page}__c{i}",
                source_file=source,
                page_number=page,
                text=piece,
            )
            for i, piece in enumerate(pieces)
            if piece.strip()
        ]

    def _load_pdf(self, path: Path) -> List[DocumentChunk]:
        chunks = []
        try:
            with pdfplumber.open(path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ""
                    if text.strip():
                        chunks.extend(self._split(text, path.name, page_num))
        except Exception as e:
            logger.warning(f"Could not parse PDF {path}: {e}")
        return chunks

    def _load_docx(self, path: Path) -> List[DocumentChunk]:
        chunks = []
        try:
            doc = DocxDocument(path)
            full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            chunks.extend(self._split(full_text, path.name, 1))
        except Exception as e:
            logger.warning(f"Could not parse DOCX {path}: {e}")
        return chunks

    def _load_txt(self, path: Path) -> List[DocumentChunk]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return self._split(text, path.name, 1)
