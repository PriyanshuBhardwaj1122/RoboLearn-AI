"""
graph/kg_builder.py
Extracts entities and relationships from document chunks using an LLM
and stores them in Neo4j as a labelled property graph.
"""
from __future__ import annotations
import json
import re
from typing import List, Dict, Any

from neo4j import GraphDatabase
from loguru import logger

from config.settings import settings
from ingestion.document_loader import DocumentChunk
from generation.llm_client import LLMClient


# ── Data classes ──────────────────────────────────────────────────────────────

class KGEntity:
    def __init__(self, name: str, label: str = "Entity", source: str = ""):
        self.name = name.strip()
        # Neo4j labels: alphanumeric only — strip /, spaces, special chars
        clean = re.sub(r"[^A-Za-z0-9]", "", label.strip().title().replace(" ", ""))
        self.label = clean or "Entity"
        self.source = source


class KGRelation:
    def __init__(self, source: str, relation: str, target: str, chunk_id: str = ""):
        self.source = source.strip()
        # Neo4j relationship types: uppercase alphanumeric + underscore only
        self.relation = re.sub(r"[^A-Z_]", "", relation.upper().replace(" ", "_")) or "RELATED_TO"
        self.target = target.strip()
        self.chunk_id = chunk_id


# ── Builder ───────────────────────────────────────────────────────────────────

class KnowledgeGraphBuilder:
    """
    Two-phase pipeline:
      1. Extract entities + relations from each chunk via LLM.
      2. Upsert them into Neo4j.
    """

    def __init__(self):
        self.llm = LLMClient()
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        self._ensure_constraints()

    def close(self) -> None:
        self.driver.close()

    # ── Public ──────────────────────────────────────────────────────────────

    def build_from_chunks(self, chunks: List[DocumentChunk]) -> None:
        total_entities, total_relations = 0, 0
        for i, chunk in enumerate(chunks):
            try:
                entities, relations = self._extract(chunk)
                self._upsert_entities(entities)
                self._upsert_relations(relations)
                total_entities += len(entities)
                total_relations += len(relations)
                if (i + 1) % 10 == 0:
                    logger.info(f"  Processed {i+1}/{len(chunks)} chunks | "
                                f"entities={total_entities} relations={total_relations}")
            except Exception as e:
                logger.warning(f"Skipping chunk {chunk.chunk_id}: {e}")
                continue
        logger.success(f"KG built: {total_entities} entities, {total_relations} relations")

    def clear(self) -> None:
        with self.driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")
        logger.warning("Neo4j graph cleared")

    # ── Extraction ──────────────────────────────────────────────────────────

    def _extract(self, chunk: DocumentChunk) -> tuple[List[KGEntity], List[KGRelation]]:
        prompt = settings.entity_extraction_prompt.format(text=chunk.text)
        raw = self.llm.complete(prompt)
        return self._parse_extraction(raw, chunk)

    def _parse_extraction(self, raw: str, chunk: DocumentChunk) -> tuple[List[KGEntity], List[KGRelation]]:
        entities: List[KGEntity] = []
        relations: List[KGRelation] = []
        try:
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            data: Dict[str, Any] = json.loads(clean)

            for e in data.get("entities", []):
                if isinstance(e, dict) and e.get("name"):
                    entities.append(KGEntity(
                        name=e.get("name", ""),
                        label=e.get("type", "Entity"),
                        source=chunk.source_file,
                    ))
                elif isinstance(e, str) and e.strip():
                    entities.append(KGEntity(name=e, source=chunk.source_file))

            for r in data.get("relations", []):
                if isinstance(r, dict) and r.get("source") and r.get("target"):
                    relations.append(KGRelation(
                        source=r.get("source", ""),
                        relation=r.get("relation", "RELATED_TO"),
                        target=r.get("target", ""),
                        chunk_id=chunk.chunk_id,
                    ))
        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"Parse error on chunk {chunk.chunk_id}: {e}")
        return entities, relations

    # ── Neo4j writes ────────────────────────────────────────────────────────

    def _ensure_constraints(self) -> None:
        try:
            with self.driver.session() as s:
                s.run(
                    "CREATE CONSTRAINT entity_name IF NOT EXISTS "
                    "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
                )
        except Exception as e:
            logger.warning(f"Constraint creation skipped: {e}")

    def _upsert_entities(self, entities: List[KGEntity]) -> None:
        with self.driver.session() as s:
            for e in entities:
                if not e.name or not e.label:
                    continue
                query = (
                    f"MERGE (n:{e.label} {{name: $name}}) "
                    "ON CREATE SET n.source = $source, n.created = timestamp() "
                    "ON MATCH SET n.source = $source"
                )
                s.run(query, name=e.name, source=e.source)

    def _upsert_relations(self, relations: List[KGRelation]) -> None:
        with self.driver.session() as s:
            for r in relations:
                if not r.source or not r.target or not r.relation:
                    continue
                query = (
                    "MERGE (a {name: $source}) "
                    "MERGE (b {name: $target}) "
                    f"MERGE (a)-[rel:{r.relation}]->(b) "
                    "ON CREATE SET rel.chunk_id = $chunk_id"
                )
                s.run(query, source=r.source, target=r.target, chunk_id=r.chunk_id)