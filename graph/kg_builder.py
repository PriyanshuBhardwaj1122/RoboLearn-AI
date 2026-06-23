"""
graph/kg_builder.py
4-stage knowledge graph build pipeline:
  chunk → LLM extraction → validation → ontology mapper → Neo4j

Changes from v1:
  - Uses Claude Sonnet (via Anthropic API) for extraction — better JSON reliability
  - Fixed domain ontology: entity types constrained to 15 canonical types
  - Validation layer with retry-once strategy
  - Protected hardware identifier patterns (ADC0-5, Timer0-2, pins etc.)
  - Parent chunks only — richer context per extraction call
  - Progress tracking with resume support
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import List, Tuple, Optional

import anthropic
from neo4j import GraphDatabase
from loguru import logger

from config.settings import settings
from ingestion.document_loader import DocumentChunk
from graph.ontology import ENTITY_TYPES, RELATION_TYPES, OntologyMapper
from graph.kg_validator import KGValidator, ValidationResult


# ── Progress tracking ─────────────────────────────────────────────────────────

PROGRESS_PATH = Path("data/processed/kg_progress.json")


def _load_progress() -> set:
    if PROGRESS_PATH.exists():
        try:
            data = json.loads(PROGRESS_PATH.read_text())
            return set(data.get("processed_chunk_ids", []))
        except Exception:
            return set()
    return set()


def _save_progress(processed_ids: set) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(json.dumps(
        {"processed_chunk_ids": sorted(processed_ids)}, indent=2
    ))


# ── Extraction prompt ─────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """Extract technical entities and their relationships from the text below.

ENTITY TYPES — you MUST use ONLY these types (choose the closest match):
{entity_types}

If unsure, use "Component" as the default type.

CRITICAL RULES:
- ADC0, ADC1, ADC2, ADC3, ADC4, ADC5 are SEPARATE entities — never combine them
- Timer0, Timer1, Timer2 are SEPARATE entities — never combine them
- D0-D13 (digital pins), A0-A5 (analog pins) are SEPARATE entities
- OC0A, OC0B, OC1A, OC1B, OC2A, OC2B are SEPARATE entities
- SDA, SCL, MOSI, MISO, SCK, SS, TX, RX are SEPARATE entities
- Entity names must be specific (min 2 chars), not generic words like "system", "it", "the"

RELATION TYPES — use descriptive relation names like:
IS_TYPE, BELONGS_TO, PART_OF, HAS_PIN, CONNECTS_TO, COMMUNICATES_VIA,
CONTROLS, READS, WRITES, GENERATES, OPERATES_AT, REQUIRES, IMPLEMENTS,
PUBLISHES_TO, SUBSCRIBES_TO, CONFIGURED_BY, TRIGGERED_BY

Return ONLY valid JSON, no markdown, no explanation:
{{"entities": [{{"name": "...", "type": "..."}}], "relations": [{{"source": "...", "relation": "...", "target": "..."}}]}}

Text:
{text}"""

RETRY_PROMPT_SUFFIX = """

Your previous response was invalid JSON or contained errors.
Return ONLY a valid JSON object with this exact structure:
{{"entities": [{{"name": "EntityName", "type": "EntityType"}}], "relations": [{{"source": "SourceName", "relation": "RELATION_TYPE", "target": "TargetName"}}]}}

Entity types MUST be one of: {entity_types}
No markdown fences. No explanation. Just the JSON object."""


# ── Builder ───────────────────────────────────────────────────────────────────

class KnowledgeGraphBuilder:
    """
    4-stage KG build pipeline:
      1. LLM extraction (Claude Sonnet)
      2. Validation (KGValidator)
      3. Ontology mapping (OntologyMapper)
      4. Neo4j upsert
    """

    def __init__(self):
        # Anthropic client for KG extraction
        self.client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key
        )
        self.kg_model = settings.kg_llm_model

        # Neo4j
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

        # Validator + mapper
        self.validator = KGValidator()
        self.mapper    = OntologyMapper()

        self._ensure_constraints()

        # Build entity types string for prompt
        self.entity_types_str = ", ".join(sorted(ENTITY_TYPES))

    def close(self) -> None:
        self.driver.close()

    # ── Public ─────────────────────────────────────────────────────────────────

    def build_from_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Build KG from chunks. Uses parent chunks only if available.
        Supports resume via progress tracking.
        """
        # Filter to parent chunks only
        parent_chunks = [
            c for c in chunks
            if getattr(c, "content_level", "child") == "parent"
        ]
        if parent_chunks:
            chunks = parent_chunks
            logger.info(
                f"KG build: using {len(chunks)} parent chunks "
                f"(richer context per extraction)"
            )
        else:
            logger.info(
                f"KG build: no parent chunks found, "
                f"using all {len(chunks)} chunks (backward compat)"
            )

        # Resume support
        processed_ids = _load_progress()
        remaining = [c for c in chunks if c.chunk_id not in processed_ids]
        if processed_ids:
            logger.info(
                f"Resuming: {len(processed_ids)} already done, "
                f"{len(remaining)} remaining"
            )

        total_entities  = 0
        total_relations = 0
        skipped         = 0

        for i, chunk in enumerate(remaining):
            try:
                result = self._process_chunk(chunk)

                if result.skipped:
                    skipped += 1
                    logger.warning(
                        f"  Skipped chunk {chunk.chunk_id} after validation failure"
                    )
                else:
                    self._upsert_to_neo4j(result, chunk)
                    total_entities  += result.entity_count_kept
                    total_relations += result.relation_count_kept

                processed_ids.add(chunk.chunk_id)

                # Save progress every 10 chunks
                if (i + 1) % 10 == 0:
                    _save_progress(processed_ids)
                    logger.info(
                        f"  Progress: {i+1}/{len(remaining)} chunks | "
                        f"entities={total_entities} "
                        f"relations={total_relations} "
                        f"skipped={skipped}"
                    )

                # Rate limit safety — Claude allows high throughput but be safe
                time.sleep(0.1)

            except Exception as ex:
                logger.warning(
                    f"  Error on chunk {chunk.chunk_id}: {ex}"
                )
                continue

        _save_progress(processed_ids)
        logger.success(
            f"KG build complete: {total_entities} entities, "
            f"{total_relations} relations, {skipped} chunks skipped"
        )

    def clear(self) -> None:
        """Wipe the entire Neo4j graph."""
        with self.driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")
        logger.warning("Neo4j graph cleared")

    # ── Extraction ─────────────────────────────────────────────────────────────

    def _process_chunk(self, chunk: DocumentChunk) -> ValidationResult:
        """
        Run the 4-stage pipeline for one chunk.
        Retry once on validation failure.
        """
        # ── Stage 1: LLM extraction ───────────────────────────────────────────
        prompt = EXTRACTION_PROMPT.format(
            entity_types=self.entity_types_str,
            text=chunk.text[:3000],  # cap at 3000 chars for safety
        )
        raw = self._call_claude(prompt)

        # ── Stage 2: Validation ───────────────────────────────────────────────
        result = self.validator.validate(raw, chunk.chunk_id, chunk.source_file)

        # ── Retry once if needed ──────────────────────────────────────────────
        if self.validator.needs_retry(result):
            logger.debug(f"  [{chunk.chunk_id}] Retrying with stricter prompt")
            retry_prompt = prompt + RETRY_PROMPT_SUFFIX.format(
                entity_types=self.entity_types_str
            )
            raw2   = self._call_claude(retry_prompt)
            result = self.validator.validate(
                raw2, chunk.chunk_id, chunk.source_file
            )
            result.retried = True

            if result.skipped:
                logger.warning(
                    f"  [{chunk.chunk_id}] Failed after retry — skipping"
                )

        self.validator.log_summary(result, chunk.chunk_id)
        return result

    def _call_claude(self, prompt: str) -> str:
        """Call Claude Sonnet and return the text response."""
        try:
            msg = self.client.messages.create(
                model=self.kg_model,
                max_tokens=1024,
                temperature=0.0,  # deterministic for structured extraction
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text if msg.content else ""
        except anthropic.RateLimitError:
            logger.warning("Rate limit hit — waiting 30s")
            time.sleep(30)
            return self._call_claude(prompt)
        except Exception as ex:
            logger.warning(f"Claude API error: {ex}")
            return ""

    # ── Neo4j upsert ────────────────────────────────────────────────────────────

    def _upsert_to_neo4j(
        self,
        result: ValidationResult,
        chunk: DocumentChunk,
    ) -> None:
        """Write validated entities and relations to Neo4j."""
        with self.driver.session() as session:
            # Upsert entities
            for entity in result.entities:
                label = re.sub(r"[^A-Za-z0-9]", "", entity.type)
                if not label:
                    label = "Component"
                query = (
                    f"MERGE (n:{label} {{name: $name}}) "
                    "ON CREATE SET "
                    "  n.source = $source, "
                    "  n.entity_type = $etype, "
                    "  n.created = timestamp(), "
                    "  n.protected = $protected "
                    "ON MATCH SET "
                    "  n.source = $source, "
                    "  n.entity_type = $etype, "
                    "  n.protected = $protected"
                )
                from graph.ontology import is_protected
                session.run(
                    query,
                    name=entity.name,
                    source=entity.source,
                    etype=entity.type,
                    protected=is_protected(entity.name),
                )

            # Upsert relations
            for rel in result.relations:
                # Clean relation type for Neo4j
                rel_type = re.sub(
                    r"[^A-Z_]", "",
                    rel.relation.upper().replace(" ", "_")
                ) or "RELATED_TO"
                query = (
                    "MERGE (a {name: $source}) "
                    "MERGE (b {name: $target}) "
                    f"MERGE (a)-[r:{rel_type}]->(b) "
                    "ON CREATE SET r.chunk_id = $chunk_id"
                )
                session.run(
                    query,
                    source=rel.source,
                    target=rel.target,
                    chunk_id=rel.chunk_id,
                )

    # ── Neo4j setup ────────────────────────────────────────────────────────────

    def _ensure_constraints(self) -> None:
        """Create uniqueness constraints for core entity types."""
        with self.driver.session() as session:
            for etype in ["Component", "Peripheral", "Pin", "Register",
                          "Protocol", "Microcontroller", "Package", "Node"]:
                try:
                    label = re.sub(r"[^A-Za-z0-9]", "", etype)
                    session.run(
                        f"CREATE CONSTRAINT {label.lower()}_name IF NOT EXISTS "
                        f"FOR (n:{label}) REQUIRE n.name IS UNIQUE"
                    )
                except Exception:
                    pass  # constraint may already exist