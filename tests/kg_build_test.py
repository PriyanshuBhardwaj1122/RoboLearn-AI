"""
tests/kg_build_test.py
═══════════════════════════════════════════════════════════════
Tests ONLY the knowledge graph building pipeline:
  1. Load & chunk PDFs
  2. Send each chunk to Groq LLM
  3. Parse the JSON response (entities + relations)
  4. Write Cypher MERGE queries to Neo4j
  5. Verify the graph was built

Usage:
  python tests/kg_build_test.py --sample 5     # test with 5 chunks
  python tests/kg_build_test.py --sample 20    # test with 20 chunks
  python tests/kg_build_test.py --chunk-id "ARDUINO.pdf__p1__c0"  # one specific chunk
  python tests/kg_build_test.py --verify       # just check what is in Neo4j
═══════════════════════════════════════════════════════════════
"""

import argparse
import json
import sys
import os
import re
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.settings import settings
from ingestion.document_loader import DocumentLoader
from generation.llm_client import LLMClient

# ── colours ───────────────────────────────────────────────────────────────────
GRN = "\033[92m"
BLU = "\033[94m"
YLW = "\033[93m"
CYN = "\033[96m"
RED = "\033[91m"
GRY = "\033[90m"
BLD = "\033[1m"
RST = "\033[0m"
SEP = GRY + "─" * 70 + RST


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — LOAD AND CHUNK
# ══════════════════════════════════════════════════════════════════════════════

def step_chunk(sample: int = 5, chunk_id: str = None):
    """Load saved chunks or ingest fresh. Return selected chunks."""
    print(f"\n{BLD}STEP 1 — CHUNKING{RST}")
    print(SEP)

    loader = DocumentLoader()

    if settings.chunk_store_path.exists():
        print(f"{GRY}  Loading saved chunks from {settings.chunk_store_path}{RST}")
        chunks = loader.load_chunks()
    else:
        print(f"  Ingesting from {settings.raw_dir} ...")
        chunks = loader.load_directory()
        loader.save_chunks(chunks)

    print(f"  Total chunks available : {GRN}{len(chunks)}{RST}")
    print(f"  chunk_size             : {settings.chunk_size} tokens")
    print(f"  chunk_overlap          : {settings.chunk_overlap} tokens")

    # Select chunks to process
    if chunk_id:
        selected = [c for c in chunks if c.chunk_id == chunk_id]
        if not selected:
            print(f"\n  {RED}chunk_id not found: {chunk_id}{RST}")
            print(f"  Available chunk IDs (first 10):")
            for c in chunks[:10]:
                print(f"    {GRY}{c.chunk_id}{RST}")
            sys.exit(1)
    else:
        selected = chunks[:sample]

    print(f"\n  {YLW}Selected {len(selected)} chunk(s) for KG extraction:{RST}\n")
    for c in selected:
        print(f"  {BLU}chunk_id{RST} : {c.chunk_id}")
        print(f"  {BLU}source  {RST} : {c.source_file}  |  page {c.page_number}")
        preview = c.text[:180].replace("\n", " ").strip()
        print(f"  {BLU}text    {RST} : {preview}...")
        print()

    return selected


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — SEND TO LLM, GET JSON
# ══════════════════════════════════════════════════════════════════════════════

def step_llm_extract(chunks: list):
    """
    For each chunk:
      - Show the exact prompt sent to Groq
      - Show the raw JSON response
      - Parse entities and relations
    Returns list of (chunk, entities, relations) tuples.
    """
    print(f"\n{BLD}STEP 2 — LLM ENTITY + RELATION EXTRACTION{RST}")
    print(SEP)

    llm = LLMClient()
    results = []

    print(f"  Model    : {GRN}{settings.llm_model}{RST}")
    print(f"  Provider : {settings.llm_provider}")
    print(f"  Chunks   : {len(chunks)}\n")

    for i, chunk in enumerate(chunks, 1):
        print(f"\n{'━'*70}")
        print(f"  {BLD}Chunk {i}/{len(chunks)}{RST}  —  {chunk.chunk_id}")
        print(f"{'━'*70}")

        # ── Build prompt ──────────────────────────────────────────────────────
        prompt = settings.entity_extraction_prompt.format(text=chunk.text)

        print(f"\n  {YLW}PROMPT SENT TO GROQ:{RST}")
        print(f"  {GRY}{'─'*60}{RST}")
        # Show first 500 chars of prompt
        short_prompt = prompt[:500] + ("..." if len(prompt) > 500 else "")
        for line in short_prompt.split("\n"):
            print(f"  {GRY}{line}{RST}")
        print(f"  {GRY}{'─'*60}{RST}")
        print(f"  {GRY}(full text: {len(chunk.text)} chars, "
              f"~{len(chunk.text.split())} words){RST}\n")

        # ── Call LLM ─────────────────────────────────────────────────────────
        t0 = time.time()
        try:
            raw = llm.complete(prompt)
            ms = (time.time() - t0) * 1000
        except Exception as e:
            print(f"  {RED}LLM call failed: {e}{RST}")
            results.append((chunk, [], []))
            continue

        # ── Show raw response ─────────────────────────────────────────────────
        print(f"  {GRN}RAW RESPONSE FROM GROQ{RST}  ({ms:.0f}ms):")
        print(f"  {GRY}{'─'*60}{RST}")
        for line in raw.split("\n"):
            print(f"  {line}")
        print(f"  {GRY}{'─'*60}{RST}")

        # ── Parse JSON ────────────────────────────────────────────────────────
        entities, relations = parse_llm_response(raw, chunk.chunk_id)

        print(f"\n  {CYN}PARSED ENTITIES ({len(entities)}):{RST}")
        for e in entities:
            print(f"    name={BLD}{e['name']}{RST}  type={e['label']}")

        print(f"\n  {CYN}PARSED RELATIONS ({len(relations)}):{RST}")
        for r in relations:
            print(f"    {BLD}{r['source']}{RST} "
                  f"--[{GRN}{r['relation']}{RST}]--> "
                  f"{BLD}{r['target']}{RST}")

        results.append((chunk, entities, relations))

    return results


def parse_llm_response(raw: str, chunk_id: str):
    """Parse the LLM JSON response into entity and relation dicts."""
    entities = []
    relations = []
    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        data = json.loads(clean)

        for e in data.get("entities", []):
            if isinstance(e, dict) and e.get("name"):
                label = re.sub(
                    r"[^A-Za-z0-9]", "",
                    e.get("type", "Entity").strip().title().replace(" ", "")
                ) or "Entity"
                entities.append({"name": e["name"].strip(), "label": label})
            elif isinstance(e, str) and e.strip():
                entities.append({"name": e.strip(), "label": "Entity"})

        for r in data.get("relations", []):
            if isinstance(r, dict) and r.get("source") and r.get("target"):
                rel_type = re.sub(
                    r"[^A-Z_]", "",
                    r.get("relation", "RELATED_TO").upper().replace(" ", "_")
                ) or "RELATED_TO"
                relations.append({
                    "source":   r["source"].strip(),
                    "relation": rel_type,
                    "target":   r["target"].strip(),
                    "chunk_id": chunk_id,
                })
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  {RED}JSON parse error: {e}{RST}")
        print(f"  {GRY}Raw was: {raw[:300]}{RST}")

    return entities, relations


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — WRITE CYPHER TO NEO4J
# ══════════════════════════════════════════════════════════════════════════════

def step_write_neo4j(results: list):
    """
    For each (chunk, entities, relations):
      - Show the exact Cypher MERGE query
      - Execute it on Neo4j
      - Confirm what was written
    """
    print(f"\n{BLD}STEP 3 — CYPHER MERGE → NEO4J{RST}")
    print(SEP)

    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        driver.verify_connectivity()
        print(f"  {GRN}Neo4j connected ✓{RST}\n")
    except Exception as e:
        print(f"  {RED}Neo4j offline: {e}{RST}")
        print(f"  {GRY}Showing Cypher queries only (not executing){RST}\n")
        driver = None

    total_entities  = 0
    total_relations = 0

    with (driver.session() if driver else _dummy_session()) as session:
        for chunk, entities, relations in results:
            print(f"\n  {BLU}Chunk:{RST} {chunk.chunk_id}")

            # ── Entity MERGE queries ──────────────────────────────────────────
            for e in entities:
                cypher = (
                    f"MERGE (n:{e['label']} {{name: $name}}) "
                    f"ON CREATE SET n.source = $source, n.created = timestamp() "
                    f"ON MATCH SET n.source = $source"
                )
                print(f"\n    {YLW}Entity Cypher:{RST}")
                print(f"    {GRY}{cypher}{RST}")
                print(f"    params: name={BLD}{e['name']}{RST}  source={chunk.source_file}")

                if driver:
                    try:
                        session.run(cypher, name=e["name"], source=chunk.source_file)
                        print(f"    {GRN}✓ written{RST}")
                        total_entities += 1
                    except Exception as ex:
                        print(f"    {RED}✗ {ex}{RST}")

            # ── Relation MERGE queries ────────────────────────────────────────
            for r in relations:
                cypher = (
                    f"MERGE (a {{name: $source}}) "
                    f"MERGE (b {{name: $target}}) "
                    f"MERGE (a)-[rel:{r['relation']}]->(b) "
                    f"ON CREATE SET rel.chunk_id = $chunk_id"
                )
                print(f"\n    {YLW}Relation Cypher:{RST}")
                print(f"    {GRY}{cypher}{RST}")
                print(f"    params: source={BLD}{r['source']}{RST}  "
                      f"target={BLD}{r['target']}{RST}  "
                      f"rel={GRN}{r['relation']}{RST}")

                if driver:
                    try:
                        session.run(
                            cypher,
                            source=r["source"],
                            target=r["target"],
                            chunk_id=r["chunk_id"],
                        )
                        print(f"    {GRN}✓ written{RST}")
                        total_relations += 1
                    except Exception as ex:
                        print(f"    {RED}✗ {ex}{RST}")

    print(f"\n  {GRN}Done — {total_entities} entities, {total_relations} relations written{RST}")

    if driver:
        driver.close()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — VERIFY GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def step_verify():
    """Show what is currently in the Neo4j graph."""
    print(f"\n{BLD}STEP 4 — VERIFY GRAPH IN NEO4J{RST}")
    print(SEP)

    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        driver.verify_connectivity()
    except Exception as e:
        print(f"  {RED}Neo4j offline: {e}{RST}")
        return

    with driver.session() as session:
        # Counts
        nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rels  = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        print(f"  {GRN}Total nodes     : {nodes}{RST}")
        print(f"  {GRN}Total relations : {rels}{RST}")

        # Sample nodes
        print(f"\n  {YLW}Sample nodes (up to 15):{RST}")
        rows = session.run(
            "MATCH (n) RETURN n.name AS name, labels(n) AS labels LIMIT 15"
        )
        for row in rows:
            print(f"    {BLD}{row['name']}{RST}  {GRY}{row['labels']}{RST}")

        # Sample relations
        print(f"\n  {YLW}Sample relations (up to 15):{RST}")
        rows = session.run(
            "MATCH (a)-[r]->(b) "
            "RETURN a.name AS src, type(r) AS rel, b.name AS tgt LIMIT 15"
        )
        for row in rows:
            print(f"    {BLD}{row['src']}{RST} "
                  f"--[{GRN}{row['rel']}{RST}]--> "
                  f"{BLD}{row['tgt']}{RST}")

    driver.close()


# ── dummy context manager when Neo4j is offline ───────────────────────────────
class _dummy_session:
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def run(self, *a, **kw): pass


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="KG build test — chunk → LLM → Cypher → Neo4j",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/kg_build_test.py --sample 5
  python tests/kg_build_test.py --sample 20
  python tests/kg_build_test.py --chunk-id "ARDUINO.pdf__p1__c0"
  python tests/kg_build_test.py --verify
        """
    )
    parser.add_argument("--sample",   type=int, default=5,
                        help="Number of chunks to process (default: 5)")
    parser.add_argument("--chunk-id", type=str,
                        help="Process one specific chunk by ID")
    parser.add_argument("--verify",   action="store_true",
                        help="Just verify what is already in Neo4j")
    args = parser.parse_args()

    print(f"\n{BLD}{'═'*70}")
    print(f"  RoboLearn AI — KG Build Test")
    print(f"  chunk → LLM extraction → Cypher → Neo4j")
    print(f"{'═'*70}{RST}")

    if args.verify:
        step_verify()
        return

    # Step 1 — chunk
    chunks = step_chunk(sample=args.sample, chunk_id=args.chunk_id)

    # Step 2 — LLM extraction
    results = step_llm_extract(chunks)

    # Step 3 — write to Neo4j
    step_write_neo4j(results)

    # Step 4 — verify
    step_verify()


if __name__ == "__main__":
    main()