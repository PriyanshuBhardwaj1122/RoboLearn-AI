"""
pipeline.py
CLI script to (re-)build the full ingestion pipeline.

Usage:
    python pipeline.py --ingest              # full pipeline: FAISS + KG
    python pipeline.py --ingest-only         # FAISS only, no Neo4j
    python pipeline.py --kg-only             # KG from all existing chunks
    python pipeline.py --kg-only --sample 50 # KG from first 50 chunks (test)
"""
import argparse
from loguru import logger

from config.settings import settings
from ingestion.document_loader import DocumentLoader
from embeddings.vector_store import VectorStore
from graph.kg_builder import KnowledgeGraphBuilder


def run_full_pipeline(kg: bool = True, sample: int = 0) -> None:
    logger.info("── Step 1/3: Ingesting documents ──")
    loader = DocumentLoader()
    chunks = loader.load_directory(settings.raw_dir)
    loader.save_chunks(chunks)

    logger.info("── Step 2/3: Building vector index ──")
    store = VectorStore()
    store.build(chunks)
    store.save()

    if kg:
        logger.info("── Step 3/3: Building knowledge graph ──")
        kg_chunks = chunks[:sample] if sample > 0 else chunks
        if sample:
            logger.info(f"Sample mode: using first {sample} of {len(chunks)} chunks")
        builder = KnowledgeGraphBuilder()
        builder.build_from_chunks(kg_chunks)
        builder.close()

    logger.success("Pipeline complete ✓")


def run_kg_only(sample: int = 0) -> None:
    loader = DocumentLoader()
    
    # Load parent chunks for KG build — richer context per extraction call
    parents = loader.load_parents()
    chunks = list(parents.values())
    total = len(chunks)

    kg_chunks = chunks[:sample] if sample > 0 else chunks

    if sample:
        logger.info(f"Sample mode: building KG from first {sample} of {total} parent chunks")
    else:
        logger.info(f"Building KG from {total} parent chunks using Claude Sonnet")

    builder = KnowledgeGraphBuilder()
    builder.build_from_chunks(kg_chunks)
    builder.close()
    logger.success(f"KG built from {len(kg_chunks)} parent chunks ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG + KG ingestion pipeline")
    parser.add_argument("--ingest",      action="store_true", help="Full pipeline: FAISS + KG")
    parser.add_argument("--ingest-only", action="store_true", help="FAISS only, skip Neo4j")
    parser.add_argument("--kg-only",     action="store_true", help="Build KG from saved chunks")
    parser.add_argument("--sample",      type=int, default=0,
                        help="Only process first N chunks for KG (0 = all)")
    args = parser.parse_args()

    if args.ingest:
        run_full_pipeline(kg=True, sample=args.sample)
    elif args.ingest_only:
        run_full_pipeline(kg=False)
    elif args.kg_only:
        run_kg_only(sample=args.sample)
    else:
        parser.print_help()