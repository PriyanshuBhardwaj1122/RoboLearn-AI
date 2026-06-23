"""
graph/kg_validator.py
Validation layer for LLM-extracted entities and relations.

Pipeline position:
  chunk → LLM extraction → [kg_validator] → ontology mapper → Neo4j

Validation steps (in order):
  1. JSON validity check — retry once with stricter prompt on failure
  2. Entity validation — name length, stopwords, type normalisation
  3. Relation validation — required fields, self-loops, type normalisation
  4. Protected pattern enforcement — hardware identifiers never merged
  5. Deduplication — remove duplicate entities and relations

Retry strategy:
  - First failure: retry with stricter prompt (one retry only)
  - Second failure: log and skip chunk (no infinite retries)
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

from loguru import logger

from graph.ontology import OntologyMapper, is_protected, ENTITY_TYPES


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ValidatedEntity:
    name:   str
    type:   str
    source: str = ""

    def __hash__(self):
        return hash((self.name.lower(), self.type))

    def __eq__(self, other):
        return (self.name.lower() == other.name.lower()
                and self.type == other.type)


@dataclass
class ValidatedRelation:
    source:   str
    relation: str
    target:   str
    chunk_id: str = ""

    def __hash__(self):
        return hash((self.source.lower(), self.relation, self.target.lower()))

    def __eq__(self, other):
        return (self.source.lower()   == other.source.lower()
                and self.relation     == other.relation
                and self.target.lower() == other.target.lower())


@dataclass
class ValidationResult:
    entities:          List[ValidatedEntity]  = field(default_factory=list)
    relations:         List[ValidatedRelation] = field(default_factory=list)
    parse_ok:          bool  = True
    entity_count_raw:  int   = 0
    entity_count_kept: int   = 0
    relation_count_raw:int   = 0
    relation_count_kept:int  = 0
    discarded_entities:List[str] = field(default_factory=list)
    discarded_relations:List[str] = field(default_factory=list)
    retried:           bool  = False
    skipped:           bool  = False


# ── Validator ─────────────────────────────────────────────────────────────────

class KGValidator:
    """
    Validates and normalises raw LLM extraction output.

    Usage:
        validator = KGValidator()
        result = validator.validate(raw_llm_text, chunk_id, source_file)
        if not result.skipped:
            # use result.entities and result.relations
    """

    # Strict retry prompt appended when first extraction fails validation
    RETRY_SUFFIX = (
        "\n\nIMPORTANT: Your previous response was invalid. "
        "Return ONLY a JSON object with this exact structure — "
        "no markdown, no explanation, no extra keys:\n"
        '{"entities": [{"name": "...", "type": "..."}], '
        '"relations": [{"source": "...", "relation": "...", "target": "..."}]}\n'
        "Entity types MUST be one of: "
        + ", ".join(sorted(ENTITY_TYPES))
        + "\nDo not invent new types."
    )

    def __init__(self):
        self.mapper = OntologyMapper()

    # ── Public ─────────────────────────────────────────────────────────────────

    def validate(
        self,
        raw_text:    str,
        chunk_id:    str = "",
        source_file: str = "",
    ) -> ValidationResult:
        """
        Validate raw LLM output. Returns a ValidationResult.
        Does NOT retry — caller is responsible for retry logic using
        needs_retry() and get_retry_suffix().
        """
        result = ValidationResult()

        # ── Step 1: Parse JSON ────────────────────────────────────────────────
        data = self._parse_json(raw_text)
        if data is None:
            result.parse_ok = False
            result.skipped  = True
            logger.debug(f"  [{chunk_id}] JSON parse failed — skipping")
            return result

        # ── Step 2: Validate entities ─────────────────────────────────────────
        raw_entities = data.get("entities", [])
        result.entity_count_raw = len(raw_entities)

        seen_entity_names: set[str] = set()
        for e in raw_entities:
            if not isinstance(e, dict):
                continue
            name = str(e.get("name", "")).strip()
            raw_type = str(e.get("type", "")).strip()

            # Name validation
            if not self.mapper.is_valid_entity_name(name):
                result.discarded_entities.append(f"invalid_name:{name!r}")
                continue

            # Normalise type via ontology mapper
            canonical_type = self.mapper.map_entity_type(raw_type)

            # Deduplicate (case-insensitive name)
            key = name.lower()
            if key in seen_entity_names:
                result.discarded_entities.append(f"duplicate:{name!r}")
                continue
            seen_entity_names.add(key)

            result.entities.append(ValidatedEntity(
                name=name,
                type=canonical_type,
                source=source_file,
            ))

        result.entity_count_kept = len(result.entities)

        # ── Step 3: Validate relations ────────────────────────────────────────
        raw_relations = data.get("relations", [])
        result.relation_count_raw = len(raw_relations)

        seen_relations: set = set()
        for r in raw_relations:
            if not isinstance(r, dict):
                continue
            src      = str(r.get("source",   "")).strip()
            rel      = str(r.get("relation", "")).strip()
            tgt      = str(r.get("target",   "")).strip()

            # Structural validation
            if not self.mapper.is_valid_relation(src, rel, tgt):
                result.discarded_relations.append(
                    f"invalid:{src!r}-{rel!r}-{tgt!r}"
                )
                continue

            # Source/target name validation
            if not self.mapper.is_valid_entity_name(src):
                result.discarded_relations.append(f"bad_source:{src!r}")
                continue
            if not self.mapper.is_valid_entity_name(tgt):
                result.discarded_relations.append(f"bad_target:{tgt!r}")
                continue

            # Protected pattern: prevent merging of hardware identifiers
            # If source or target is protected, ensure exact name is preserved
            if is_protected(src) or is_protected(tgt):
                # Keep as-is — don't normalise protected entity names
                pass

            # Normalise relation type
            canonical_rel = self.mapper.map_relation_type(rel)

            # Deduplicate triples (case-insensitive source/target)
            triple_key = (src.lower(), canonical_rel, tgt.lower())
            if triple_key in seen_relations:
                result.discarded_relations.append(
                    f"duplicate:{src!r}-{canonical_rel}-{tgt!r}"
                )
                continue
            seen_relations.add(triple_key)

            result.relations.append(ValidatedRelation(
                source=src,
                relation=canonical_rel,
                target=tgt,
                chunk_id=chunk_id,
            ))

        result.relation_count_kept = len(result.relations)
        result.skipped = False

        logger.debug(
            f"  [{chunk_id}] validated: "
            f"{result.entity_count_kept}/{result.entity_count_raw} entities, "
            f"{result.relation_count_kept}/{result.relation_count_raw} relations"
        )
        return result

    def needs_retry(self, result: ValidationResult) -> bool:
        """Return True if the result should trigger a retry."""
        if not result.parse_ok:
            return True
        # Retry if we got nothing at all from a non-empty response
        if result.entity_count_raw == 0 and result.relation_count_raw == 0:
            return True
        return False

    def get_retry_suffix(self) -> str:
        """Return the suffix to append to the prompt for retry."""
        return self.RETRY_SUFFIX

    def log_summary(self, result: ValidationResult, chunk_id: str) -> None:
        """Log a detailed validation summary for debugging."""
        if result.skipped:
            logger.warning(f"  [{chunk_id}] SKIPPED after validation failure")
            return
        if result.retried:
            logger.info(f"  [{chunk_id}] Recovered via retry")
        if result.discarded_entities:
            logger.debug(
                f"  [{chunk_id}] Discarded {len(result.discarded_entities)} "
                f"entities: {result.discarded_entities[:3]}"
            )
        if result.discarded_relations:
            logger.debug(
                f"  [{chunk_id}] Discarded {len(result.discarded_relations)} "
                f"relations: {result.discarded_relations[:3]}"
            )

    # ── Private ────────────────────────────────────────────────────────────────

    def _parse_json(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to parse JSON from raw LLM output.
        Strips markdown fences and leading/trailing whitespace.
        """
        if not raw_text:
            return None

        # Strip markdown code fences
        clean = re.sub(r"```(?:json)?|```", "", raw_text).strip()

        # Try direct parse
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            pass

        # Try extracting first JSON object from text
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return None
