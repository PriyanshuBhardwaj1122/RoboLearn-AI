"""
ingestion/document_loader.py
Phase 1 rebuild — structured extraction pipeline.

Extraction layers (applied per page in order):
  Layer 1: pdfplumber   — digital text + table extraction
  Layer 2: pymupdf      — heading detection via font size / bold flag
  Layer 3: unstructured — structural element classification
  Layer 4: tesseract    — OCR for low-text / scanned pages

Chunking strategy:
  Parent chunks (~1500 tokens) — full section under a heading
                                  stored in parents.json, sent to LLM
  Child  chunks (~256  tokens) — sub-splits of parent
                                  stored in chunks.json, used for FAISS/BM25

Retrieval flow:
  FAISS/BM25 finds child → parent_id lookup → parent text sent to LLM
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, asdict, field
from pathlib import Path
from statistics import mean
from typing import List, Dict, Optional, Tuple

import fitz                          # pymupdf
import pdfplumber
from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from PIL import Image

from config.settings import settings
import logging
logging.getLogger("pdfplumber").setLevel(logging.ERROR)
logging.getLogger("pdfminer").setLevel(logging.ERROR)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    chunk_id:      str
    source_file:   str
    page_number:   int
    text:          str
    metadata:      dict = field(default_factory=dict)

    # Phase 1 new fields
    chunk_type:    str  = "text"   # "text" | "table" | "figure" | "ocr"
    content_level: str  = "child"  # "parent" | "child"
    parent_id:     str  = ""       # child → its parent's chunk_id
    child_ids:     list = field(default_factory=list)   # parent → child ids
    heading_path:  list = field(default_factory=list)   # breadcrumb hierarchy
    bbox:          dict = field(default_factory=dict)   # page bounding box
    table_data:    dict = field(default_factory=dict)   # raw table structure


# ── Page element dataclass (internal) ─────────────────────────────────────────

@dataclass
class PageElement:
    """Raw extracted element before chunking."""
    element_type: str          # "heading" | "text" | "table" | "figure" | "ocr"
    text:         str
    page:         int
    bbox:         dict = field(default_factory=dict)
    table_data:   dict = field(default_factory=dict)
    font_size:    float = 0.0
    is_bold:      bool  = False


# ── Main loader ───────────────────────────────────────────────────────────────

class DocumentLoader:
    """
    Ingests PDFs and other documents.
    Produces two outputs:
      - child chunks  → chunks.json   (for FAISS + BM25)
      - parent chunks → parents.json  (for LLM answer generation)
    """

    def __init__(self):
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.child_chunk_size,
            chunk_overlap=settings.child_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.parent_chunk_size,
            chunk_overlap=128,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_directory(self, directory: Path | None = None) -> List[DocumentChunk]:
        """
        Load all documents. Returns child chunks (for FAISS/BM25).
        Parent chunks are saved separately to parents.json.
        """
        directory = directory or settings.raw_dir
        all_children: List[DocumentChunk] = []
        all_parents:  List[DocumentChunk] = []

        for path in sorted(Path(directory).rglob("*")):
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                children, parents = self._load_pdf(path)
            elif suffix in {".docx", ".doc"}:
                children, parents = self._load_docx(path)
            elif suffix == ".txt":
                children, parents = self._load_txt(path)
            else:
                continue

            all_children.extend(children)
            all_parents.extend(parents)
            logger.info(
                f"  {path.name}: {len(parents)} parents → "
                f"{len(children)} children"
            )

        logger.info(
            f"Total: {len(all_parents)} parent chunks, "
            f"{len(all_children)} child chunks"
        )

        # Save parents separately
        self._save_parents(all_parents)
        return all_children

    def save_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Save child chunks to chunks.json."""
        settings.processed_dir.mkdir(parents=True, exist_ok=True)
        with open(settings.chunk_store_path, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in chunks], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(chunks)} child chunks → {settings.chunk_store_path}")

    def load_chunks(self) -> List[DocumentChunk]:
        """Reload child chunks from chunks.json."""
        with open(settings.chunk_store_path, encoding="utf-8") as f:
            data = json.load(f)
        return [DocumentChunk(**d) for d in data]

    def load_parents(self) -> Dict[str, DocumentChunk]:
        """Load parent chunks as a dict keyed by chunk_id."""
        path = settings.parent_store_path
        if not path.exists():
            logger.warning("parents.json not found — run ingestion first")
            return {}
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return {d["chunk_id"]: DocumentChunk(**d) for d in data}

    # ── PDF extraction ─────────────────────────────────────────────────────────

    def _load_pdf(self, path: Path) -> Tuple[List[DocumentChunk], List[DocumentChunk]]:
        """
        Full PDF extraction using four layers.
        Returns (child_chunks, parent_chunks).
        """
        elements: List[PageElement] = []

        try:
            # Layer 1 + 2: pdfplumber (text/tables) + pymupdf (headings)
            elements = self._extract_pdf_elements(path)

            # Layer 3: unstructured for figure captions
            if settings.figure_caption_extraction:
                figure_elements = self._extract_figure_captions(path)
                elements.extend(figure_elements)

            # Sort by page then vertical position
            elements.sort(key=lambda e: (e.page, e.bbox.get("y0", 0)))

        except Exception as ex:
            logger.warning(f"PDF extraction failed for {path.name}: {ex}")
            return [], []

        # Build parent chunks from element stream
        parents = self._build_parents(elements, path.name)

        # Split parents into children
        children = self._build_children(parents)

        return children, parents

    def _extract_pdf_elements(self, path: Path) -> List[PageElement]:
        """
        Layer 1 (pdfplumber) + Layer 2 (pymupdf) + Layer 4 (OCR).
        Returns raw PageElement list for the whole PDF.
        """
        elements: List[PageElement] = []

        # Open both readers
        fitz_doc = fitz.open(str(path))
        avg_body_size = self._compute_avg_font_size(fitz_doc)

        with pdfplumber.open(path) as plumber_doc:
            for page_num, (plumber_page, fitz_page) in enumerate(
                zip(plumber_doc.pages, fitz_doc), start=1
            ):
                page_elements = []

                # ── Layer 1: tables via pdfplumber ────────────────────────────
                if settings.table_extraction:
                    table_bboxes = []
                    for table in plumber_page.find_tables():
                        table_data = table.extract()
                        if not table_data:
                            continue
                        headers = table_data[0] if table_data else []
                        rows    = table_data[1:] if len(table_data) > 1 else table_data
                        # Render table as markdown text for embedding
                        md = self._table_to_markdown(headers, rows)
                        bbox_dict = {
                            "x0": table.bbox[0], "y0": table.bbox[1],
                            "x1": table.bbox[2], "y1": table.bbox[3],
                        }
                        page_elements.append(PageElement(
                            element_type="table",
                            text=md,
                            page=page_num,
                            bbox=bbox_dict,
                            table_data={"headers": headers, "rows": rows},
                        ))
                        table_bboxes.append(table.bbox)

                # ── Layer 2: text + headings via pymupdf ──────────────────────
                blocks = fitz_page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
                for block in blocks.get("blocks", []):
                    if block.get("type") != 0:  # 0 = text block
                        continue
                    block_bbox = {
                        "x0": block["bbox"][0], "y0": block["bbox"][1],
                        "x1": block["bbox"][2], "y1": block["bbox"][3],
                    }
                    # Skip blocks that overlap with already-extracted tables
                    if settings.table_extraction and self._overlaps_table(
                        block["bbox"], table_bboxes
                    ):
                        continue

                    for line in block.get("lines", []):
                        line_text = ""
                        max_font  = 0.0
                        is_bold   = False
                        for span in line.get("spans", []):
                            line_text += span.get("text", "")
                            span_size  = span.get("size", 0)
                            if span_size > max_font:
                                max_font = span_size
                            if "bold" in span.get("font", "").lower():
                                is_bold = True

                        line_text = line_text.strip()
                        if not line_text:
                            continue

                        # Heading detection: font size > avg body OR bold + short
                        is_heading = False
                        if settings.heading_detection:
                            if max_font > avg_body_size * 1.15:
                                is_heading = True
                            elif is_bold and len(line_text) < 120:
                                is_heading = True

                        page_elements.append(PageElement(
                            element_type="heading" if is_heading else "text",
                            text=line_text,
                            page=page_num,
                            bbox=block_bbox,
                            font_size=max_font,
                            is_bold=is_bold,
                        ))

                # ── Layer 4: OCR for low-text pages ───────────────────────────
                total_text = " ".join(e.text for e in page_elements)
                if settings.ocr_enabled and len(total_text.strip()) < settings.ocr_min_chars:
                    ocr_text = self._run_ocr(fitz_page)
                    if ocr_text.strip():
                        page_elements.append(PageElement(
                            element_type="ocr",
                            text=ocr_text,
                            page=page_num,
                            bbox={},
                        ))
                        logger.debug(f"  OCR on page {page_num} of {path.name}: "
                                     f"{len(ocr_text)} chars")

                elements.extend(page_elements)

        fitz_doc.close()
        return elements

    def _extract_figure_captions(self, path: Path) -> List[PageElement]:
        """Layer 3: use unstructured to find figure captions."""
        captions: List[PageElement] = []
        try:
            from unstructured.partition.pdf import partition_pdf
            raw_elements = partition_pdf(
                filename=str(path),
                strategy="fast",
                infer_table_structure=False,
            )
            for el in raw_elements:
                cat = str(type(el).__name__)
                if "FigureCaption" in cat or "Caption" in cat:
                    captions.append(PageElement(
                        element_type="figure",
                        text=str(el).strip(),
                        page=getattr(el.metadata, "page_number", 0) or 0,
                        bbox={},
                    ))
        except Exception as ex:
            logger.debug(f"Figure caption extraction skipped for {path.name}: {ex}")
        return captions

    # ── Parent-child chunking ──────────────────────────────────────────────────

    def _build_parents(
        self, elements: List[PageElement], source: str
    ) -> List[DocumentChunk]:
        """
        Group elements by heading sections.
        Each section becomes one or more parent chunks.
        """
        parents: List[DocumentChunk] = []
        current_headings: List[str] = []
        current_buffer:   List[PageElement] = []
        current_page = 1

        def flush_section():
            nonlocal current_buffer, current_page
            if not current_buffer:
                return
            # Combine all element texts in this section
            section_parts = []
            table_data_list = []
            for el in current_buffer:
                if el.element_type == "table":
                    section_parts.append(el.text)
                    table_data_list.append(el.table_data)
                else:
                    section_parts.append(el.text)

            section_text = "\n\n".join(p for p in section_parts if p.strip())
            if not section_text.strip():
                return

            # Split long sections into multiple parents
            parent_texts = self.parent_splitter.split_text(section_text)
            for i, ptext in enumerate(parent_texts):
                if not ptext.strip():
                    continue
                pid = f"{source}__par{len(parents)}_{i}"
                chunk_type = "table" if table_data_list and i == 0 else "text"
                # Pick dominant element type
                types = [e.element_type for e in current_buffer]
                if "ocr" in types and types.count("ocr") > len(types) / 2:
                    chunk_type = "ocr"
                elif "figure" in types:
                    chunk_type = "figure"

                parents.append(DocumentChunk(
                    chunk_id=pid,
                    source_file=source,
                    page_number=current_page,
                    text=ptext,
                    chunk_type=chunk_type,
                    content_level="parent",
                    parent_id="",
                    child_ids=[],
                    heading_path=list(current_headings),
                    bbox=current_buffer[0].bbox if current_buffer else {},
                    table_data=table_data_list[0] if table_data_list else {},
                    metadata={
                        "heading": " > ".join(current_headings) if current_headings else "",
                        "section_index": i,
                    },
                ))
            current_buffer = []

        for el in elements:
            current_page = el.page
            if el.element_type == "heading":
                # Flush previous section before starting new one
                flush_section()
                # Update heading hierarchy
                # Simple heuristic: longer text = deeper heading
                heading_text = el.text.strip()
                if el.font_size > 0:
                    # Larger font = higher level heading
                    if el.font_size >= 14:
                        current_headings = [heading_text]
                    elif el.font_size >= 12:
                        current_headings = current_headings[:1] + [heading_text]
                    else:
                        current_headings = current_headings[:2] + [heading_text]
                else:
                    current_headings = current_headings[:1] + [heading_text]
            else:
                current_buffer.append(el)

        # Flush final section
        flush_section()

        logger.debug(f"  Built {len(parents)} parent chunks from {source}")
        return parents

    def _build_children(
        self, parents: List[DocumentChunk]
    ) -> List[DocumentChunk]:
        """Split each parent into child chunks."""
        children: List[DocumentChunk] = []

        for parent in parents:
            if parent.chunk_type == "table":
                # Tables stay as single child — don't split table text
                cid = f"{parent.chunk_id}__c0"
                child = DocumentChunk(
                    chunk_id=cid,
                    source_file=parent.source_file,
                    page_number=parent.page_number,
                    text=parent.text,
                    chunk_type="table",
                    content_level="child",
                    parent_id=parent.chunk_id,
                    child_ids=[],
                    heading_path=parent.heading_path,
                    bbox=parent.bbox,
                    table_data=parent.table_data,
                    metadata=parent.metadata,
                )
                children.append(child)
                parent.child_ids.append(cid)
                continue

            # Split text/ocr/figure parents into children
            pieces = self.child_splitter.split_text(parent.text)
            for i, piece in enumerate(pieces):
                if not piece.strip():
                    continue
                cid = f"{parent.chunk_id}__c{i}"
                child = DocumentChunk(
                    chunk_id=cid,
                    source_file=parent.source_file,
                    page_number=parent.page_number,
                    text=piece,
                    chunk_type=parent.chunk_type,
                    content_level="child",
                    parent_id=parent.chunk_id,
                    child_ids=[],
                    heading_path=parent.heading_path,
                    bbox=parent.bbox,
                    table_data={},
                    metadata=parent.metadata,
                )
                children.append(child)
                parent.child_ids.append(cid)

        logger.debug(
            f"  Built {len(children)} child chunks from {len(parents)} parents"
        )
        return children

    # ── Non-PDF loaders ────────────────────────────────────────────────────────

    def _load_docx(self, path: Path) -> Tuple[List[DocumentChunk], List[DocumentChunk]]:
        try:
            doc = DocxDocument(path)
            elements: List[PageElement] = []
            for para in doc.paragraphs:
                if not para.text.strip():
                    continue
                is_heading = para.style.name.startswith("Heading")
                elements.append(PageElement(
                    element_type="heading" if is_heading else "text",
                    text=para.text.strip(),
                    page=1,
                    is_bold=is_heading,
                ))
            parents  = self._build_parents(elements, path.name)
            children = self._build_children(parents)
            return children, parents
        except Exception as ex:
            logger.warning(f"DOCX load failed {path.name}: {ex}")
            return [], []

    def _load_txt(self, path: Path) -> Tuple[List[DocumentChunk], List[DocumentChunk]]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        elements = [PageElement(element_type="text", text=text, page=1)]
        parents  = self._build_parents(elements, path.name)
        children = self._build_children(parents)
        return children, parents

    # ── OCR helper ─────────────────────────────────────────────────────────────

    def _run_ocr(self, fitz_page) -> str:
        """Render page to image and run Tesseract OCR."""
        try:
            mat  = fitz.Matrix(2.0, 2.0)   # 2x zoom for better OCR accuracy
            pix  = fitz_page.get_pixmap(matrix=mat, alpha=False)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
                pix.save(tmp_path)
            result = subprocess.run(
                ["tesseract", tmp_path, "stdout", "-l", "eng", "--psm", "6"],
                capture_output=True, text=True, timeout=30,
            )
            Path(tmp_path).unlink(missing_ok=True)
            return result.stdout.strip()
        except Exception as ex:
            logger.debug(f"OCR failed: {ex}")
            return ""

    # ── Utility helpers ────────────────────────────────────────────────────────

    def _compute_avg_font_size(self, fitz_doc) -> float:
        """Compute average body font size across first 5 pages."""
        sizes = []
        for page in list(fitz_doc)[:5]:
            blocks = page.get_text("dict")
            for block in blocks.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        s = span.get("size", 0)
                        if s > 0:
                            sizes.append(s)
        return mean(sizes) if sizes else 10.0

    def _overlaps_table(self, block_bbox, table_bboxes: list) -> bool:
        """Check if a text block overlaps with any extracted table bbox."""
        bx0, by0, bx1, by1 = block_bbox
        for tx0, ty0, tx1, ty1 in table_bboxes:
            if bx0 < tx1 and bx1 > tx0 and by0 < ty1 and by1 > ty0:
                return True
        return False

    def _table_to_markdown(self, headers: list, rows: list) -> str:
        """Convert table to markdown string for embedding."""
        if not headers and not rows:
            return ""
        lines = []
        # Clean None values
        clean_headers = [str(h or "").strip() for h in headers]
        if clean_headers:
            lines.append("| " + " | ".join(clean_headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(clean_headers)) + " |")
        for row in rows:
            clean_row = [str(c or "").strip() for c in row]
            lines.append("| " + " | ".join(clean_row) + " |")
        return "\n".join(lines)

    # ── Save helpers ───────────────────────────────────────────────────────────

    def _save_parents(self, parents: List[DocumentChunk]) -> None:
        settings.processed_dir.mkdir(parents=True, exist_ok=True)
        with open(settings.parent_store_path, "w", encoding="utf-8") as f:
            json.dump([asdict(p) for p in parents], f, indent=2, ensure_ascii=False)
        logger.info(
            f"Saved {len(parents)} parent chunks → {settings.parent_store_path}"
        )