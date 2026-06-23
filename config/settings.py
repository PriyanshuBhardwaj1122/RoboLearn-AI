from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # ── Paths ─────────────────────────────────────────────────────────────────
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    faiss_index_path: Path = Path("data/processed/faiss_index")
    chunk_store_path: Path = Path("data/processed/chunks.json")
    parent_store_path: Path = Path("data/processed/parents.json")  # NEW

    # ── Chunking ──────────────────────────────────────────────────────────────
    # Legacy flat chunking (kept for backward compat)
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Parent-child chunking (Phase 1 rebuild)
    parent_chunk_size: int = 1500   # full section → sent to LLM
    child_chunk_size: int = 256     # sub-split → used for FAISS/BM25 retrieval
    child_chunk_overlap: int = 32

    # ── Ingestion flags ───────────────────────────────────────────────────────
    ocr_enabled: bool = True            # run Tesseract on low-text pages
    ocr_min_chars: int = 50            # pages with fewer chars trigger OCR
    table_extraction: bool = True       # extract tables via pdfplumber
    heading_detection: bool = True      # detect headings via pymupdf font size
    figure_caption_extraction: bool = True  # extract figure captions

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k_vector: int = 8
    top_k_graph: int = 10
    vector_weight: float = 0.35
    graph_weight: float = 0.35

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:7b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    kg_llm_provider: str = "anthropic"
    kg_llm_model: str = "claude-sonnet-4-6"
    ollama_base_url: str = "http://localhost:11434/v1"
    groq_api_key: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    aws_bearer_token_bedrock: str = ""
    aws_default_region: str = "us-east-1"

    entity_extraction_prompt: str = (
        "Extract all technical entities (hardware, software, protocols, "
        "components, concepts) and their relationships from the text below.\n"
        "IMPORTANT: Extract register names, pin numbers, and peripheral names "
        "as SEPARATE entities. Never merge: ADC0/ADC1/ADC2/ADC3/ADC4/ADC5, "
        "Timer0/Timer1/Timer2, D9/D10/D11, A0/A1/A2/A3/A4/A5.\n"
        "Return ONLY valid JSON — no markdown fences, no preamble:\n"
        '{{"entities": [{{"name": "...", "type": "..."}}], '
        '"relations": [{{"source": "...", "relation": "...", "target": "..."}}]}}\n\n'
        "Text:\n{text}"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()