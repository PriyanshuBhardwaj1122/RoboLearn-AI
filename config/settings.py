from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    faiss_index_path: Path = Path("data/processed/faiss_index")
    chunk_store_path: Path = Path("data/processed/chunks.json")

    chunk_size: int = 512
    chunk_overlap: int = 64

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    top_k_vector: int = 5
    top_k_graph: int = 10
    vector_weight: float = 0.6
    graph_weight: float = 0.4

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    llm_provider: str = "groq"
    groq_api_key: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024

    # NOTE: {text} is the only real placeholder — all other braces are escaped
    entity_extraction_prompt: str = (
        "Extract all technical entities (hardware, software, protocols, "
        "components, concepts) and their relationships from the text below.\n"
        "Return ONLY valid JSON — no markdown fences, no preamble:\n"
        '{{"entities": [{{"name": "...", "type": "..."}}], '
        '"relations": [{{"source": "...", "relation": "...", "target": "..."}}]}}\n\n'
        "Text:\n{text}"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()