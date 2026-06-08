# RoboLearn AI — Hybrid RAG + Knowledge Graph Assistant

> A technical training assistant that combines dense vector search (FAISS)
> with a Neo4j knowledge graph to answer questions about robotics, electronics,
> embedded systems, and communication networks.

---

## Architecture

```
Documents (PDF/DOCX/TXT)
       │
       ▼
 Document Loader  →  Chunks  ─────────────────────────────────┐
       │                                                       │
       ├──[Embeddings]──→  FAISS Index                        │
       │                       │ vector retrieval             │
       └──[LLM Extraction]──→  Neo4j KG                       │
                                │ graph retrieval              │
                                ▼                              │
                     Hybrid Fusion (α·vec + β·graph)          │
                                │                              │
                                ▼                              │
                      LLM Answer Generation  ←────────────────┘
                                │
                                ▼
                        Streamlit Chatbot UI
```

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- [Neo4j Desktop](https://neo4j.com/download/) or Docker
- APOC plugin for Neo4j (optional, enables multi-hop traversal)

```bash
# Start Neo4j via Docker
docker run \
  --name neo4j-rag \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -e NEO4JPLUGINS='["apoc"]' \
  neo4j:5
```

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY and NEO4J_PASSWORD
```

### 4. Add training materials

Place your PDFs, DOCX files, or text files in `data/raw/`.

### 5. Run the ingestion pipeline

```bash
# Full pipeline: ingest + FAISS + knowledge graph
python pipeline.py --ingest

# FAISS only (no Neo4j required)
python pipeline.py --ingest-only

# Rebuild KG from existing chunks
python pipeline.py --kg-only
```

### 6. Launch the chatbot

```bash
streamlit run ui/app.py
```

Open http://localhost:8501 in your browser.

---

## Project Structure

```
rag_kg_assistant/
├── config/
│   └── settings.py          # All configuration (pydantic-settings)
├── data/
│   ├── raw/                 # ← drop your PDFs here
│   └── processed/           # FAISS index + chunk JSON (auto-generated)
├── ingestion/
│   └── document_loader.py   # PDF/DOCX/TXT → DocumentChunk list
├── embeddings/
│   └── vector_store.py      # Sentence-Transformers + FAISS
├── graph/
│   ├── kg_builder.py        # LLM entity extraction → Neo4j
│   └── kg_retriever.py      # Cypher queries + multi-hop traversal
├── retrieval/
│   └── hybrid_retriever.py  # Fuses vector + graph scores
├── generation/
│   ├── llm_client.py        # Anthropic/OpenAI wrapper
│   └── answer_generator.py  # Prompt assembly + LLM call
├── ui/
│   └── app.py               # Streamlit chatbot
├── pipeline.py              # CLI ingestion runner
└── requirements.txt
```

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | Fast, 384-dim, good for technical text |
| Vector index | FAISS `IndexFlatIP` | Exact cosine search; swap to `IndexIVFFlat` for >1M chunks |
| Graph DB | Neo4j | Cypher is expressive; APOC enables multi-hop traversal |
| Fusion | Weighted linear | Simple, tuneable via sidebar sliders |
| LLM | Claude/GPT (swappable) | `.env` controls provider |

---

## Extending the System

### Add a new document type
Implement `_load_<type>` in `DocumentLoader` following the PDF/DOCX pattern.

### Change the embedding model
Update `EMBEDDING_MODEL` in `.env` and rebuild the index.

### Add graph visualisation
```python
from pyvis.network import Network
net = Network()
for triple in triples:
    net.add_node(triple.source)
    net.add_node(triple.target)
    net.add_edge(triple.source, triple.target, label=triple.relation)
net.save_graph("graph.html")
```

### Enable multi-hop query expansion
In `KGRetriever._fetch_triples`, increase `hops` from 2 to 3 or implement
BFS over the Neo4j adjacency matrix with `networkx`.

---

## Roadmap

- [ ] Graph visualisation panel in Streamlit (pyvis)
- [ ] Query expansion via KG traversal before vector search
- [ ] IoT / 5G / NTN domain support
- [ ] Personalised learning recommendations (user interaction graph)
- [ ] Re-ranking with cross-encoder after initial retrieval
- [ ] Streaming LLM responses in the UI
# RoboLearnAI
