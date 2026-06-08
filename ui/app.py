"""
ui/app.py
Streamlit chatbot interface for the Hybrid RAG + Knowledge Graph assistant.
Run:  python -m streamlit run ui/app.py
"""
import streamlit as st
from loguru import logger

st.set_page_config(
    page_title="RoboLearn AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Casual / chitchat detection ───────────────────────────────────────────────
CASUAL_TRIGGERS = {
    "hi", "hello", "hey", "hiya", "howdy", "good morning", "good afternoon",
    "good evening", "what's up", "whats up", "sup", "how are you", "how r u",
    "who are you", "what are you", "what can you do", "help", "thanks",
    "thank you", "bye", "goodbye", "ok", "okay", "cool", "nice", "great",
}

CASUAL_RESPONSES = {
    "hi": "Hi! 👋 I'm RoboLearn AI — your robotics and electronics training assistant. Ask me anything about ROS, Arduino, DC motors, sensors, or embedded systems!",
    "hello": "Hello! 👋 I'm here to help with robotics and electronics questions. What would you like to learn today?",
    "hey": "Hey! Ready to talk robotics? Ask me about ROS, Arduino, sensors, motors — anything from your training materials.",
    "how are you": "I'm running great and ready to help! Ask me about ROS, Arduino, embedded systems, or anything from your training materials.",
    "who are you": "I'm RoboLearn AI — a hybrid RAG + Knowledge Graph assistant trained on your robotics and electronics PDFs. I can answer questions about ROS, Arduino, DC motors, sensors, communication protocols, and more.",
    "what are you": "I'm an AI assistant that combines vector search (FAISS) and a knowledge graph (Neo4j) to answer questions from your robotics and electronics training materials.",
    "what can you do": "I can answer technical questions about:\n- 🤖 ROS (Robot Operating System)\n- ⚡ Arduino & embedded systems\n- 🔌 Electronics & circuits\n- ⚙️ DC motors, sensors, actuators\n- 📡 Communication protocols\n\nJust ask away!",
    "thanks": "You're welcome! Ask me anything else about robotics or electronics.",
    "thank you": "Happy to help! Feel free to ask more questions.",
    "bye": "Goodbye! Come back when you have more robotics questions. 👋",
    "goodbye": "Goodbye! 👋",
    "help": "I can answer questions about your robotics and electronics training materials. Try asking:\n- *'How does ROS communicate with Arduino?'*\n- *'What is PWM and how is it used?'*\n- *'Explain ROS topics and nodes'*",
}

def is_casual(query: str) -> bool:
    q = query.strip().lower().rstrip("!?.").strip()
    return q in CASUAL_TRIGGERS

def get_casual_response(query: str) -> str:
    q = query.strip().lower().rstrip("!?.").strip()
    return CASUAL_RESPONSES.get(q,
        "I'm your robotics and electronics training assistant! "
        "Ask me technical questions about ROS, Arduino, sensors, motors, "
        "or anything from your training materials. 🤖"
    )

# ── Lazy-load heavy components ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading vector index…")
def load_vector_store():
    from embeddings.vector_store import VectorStore
    return VectorStore.load()

@st.cache_resource(show_spinner="Connecting to Neo4j…")
def load_kg_retriever():
    from graph.kg_retriever import KGRetriever
    return KGRetriever()

def get_retriever():
    from retrieval.hybrid_retriever import HybridRetriever
    return HybridRetriever(load_vector_store(), load_kg_retriever())

def get_generator():
    from generation.answer_generator import AnswerGenerator
    return AnswerGenerator()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 RoboLearn AI")
    st.caption("Hybrid RAG + Knowledge Graph assistant")
    st.divider()

    st.subheader("System status")
    try:
        kg = load_kg_retriever()
        stats = kg.summarize_graph()
        st.metric("Graph nodes",     stats["nodes"])
        st.metric("Graph relations", stats["relations"])
        st.success("Neo4j connected")
    except Exception as e:
        st.error(f"Neo4j offline")

    st.divider()
    st.subheader("Retrieval settings")
    top_k_vec    = st.slider("Vector top-k",   1, 10, 5)
    top_k_graph  = st.slider("Graph top-k",    1, 20, 10)
    vec_weight   = st.slider("Vector weight",  0.0, 1.0, 0.6, 0.05)
    show_sources = st.toggle("Show sources",   value=True)
    show_triples = st.toggle("Show graph triples", value=True)

    st.divider()
    if st.button("🗑 Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ── Main chat ─────────────────────────────────────────────────────────────────
st.title("Ask your training assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources") and show_sources:
            with st.expander("📄 Document sources"):
                for s in msg["sources"]:
                    st.caption(s)
        if msg.get("triples") and show_triples:
            with st.expander("🔗 Knowledge graph triples"):
                for t in msg["triples"]:
                    st.code(t, language=None)

# Input
if query := st.chat_input("Ask about robotics, electronics, ROS, Arduino… or just say hi!"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        # ── Casual / chitchat → no RAG needed ────────────────────────────────
        if is_casual(query):
            response = get_casual_response(query)
            st.markdown(response)
            st.session_state.messages.append({
                "role": "assistant", "content": response,
                "sources": [], "triples": [],
            })

        # ── Technical question → full RAG + KG pipeline ───────────────────────
        else:
            with st.spinner("Retrieving context and generating answer…"):
                try:
                    from config.settings import settings
                    settings.top_k_vector  = top_k_vec
                    settings.top_k_graph   = top_k_graph
                    settings.vector_weight = vec_weight
                    settings.graph_weight  = 1.0 - vec_weight

                    retriever = get_retriever()
                    generator = get_generator()

                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                        if m["role"] in {"user", "assistant"}
                    ]

                    context = retriever.retrieve(query)
                    answer  = generator.generate(query, context, history)

                    st.markdown(answer.answer)

                    if show_sources and answer.sources:
                        with st.expander("📄 Document sources"):
                            for s in set(answer.sources):
                                st.caption(s)

                    if show_triples and answer.graph_triples_used:
                        with st.expander("🔗 Knowledge graph triples"):
                            for t in answer.graph_triples_used:
                                st.code(t, language=None)

                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": answer.answer,
                        "sources": list(set(answer.sources)),
                        "triples": answer.graph_triples_used,
                    })

                except FileNotFoundError:
                    err = "⚠️ Vector index not found. Run `python pipeline.py --ingest-only` first."
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
                except Exception as e:
                    logger.exception(e)
                    err = f"⚠️ Error: {e}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})